//! Finding, and then loading, an ONNX Runtime shared library -- without linking against one and
//! without asking Python where it is.
//!
//! WHY DISCOVERY IS A SEPARATE, PURE FUNCTION
//! -------------------------------------------
//! "Which ORT did this number come from" is a provenance question, and the honest answers include
//! *two different ones are on this machine*. So discovery collects candidates from every source it
//! knows, labels each one with the source that produced it, canonicalises them, and then rules:
//!
//!   * an explicit `--ort-lib` wins outright, and an explicit path that does not exist is an
//!     error rather than a reason to go looking elsewhere;
//!   * zero candidates is `ERROR(instrument=ort_library_missing)`, and the message lists every
//!     place that was looked at;
//!   * two or more *distinct* canonical paths is `ERROR(instrument=ort_library_ambiguous)` and
//!     names them all. It is not resolved by priority order, because "the first one on PATH" is
//!     exactly the silent choice that makes one machine's numbers unreproducible on another.
//!
//! [`Search`] holds the raw material and nothing reads the environment inside [`discover`], so the
//! ambiguity and missing arms are unit-testable against planted directories.
//!
//! WHY THE LIBRARY IS LEAKED
//! --------------------------
//! `OrtApi` is a table of function pointers *into the loaded image*, and `OrtEnv`/`OrtSession`
//! own memory allocated by it. Unloading while any of that is alive is undefined behaviour with a
//! plausible-looking stack trace somewhere else -- the same class of bug `rust/src/sys.rs` refuses
//! to risk with hand-written bindings. The process exits soon after; the OS reclaims the mapping.

use std::ffi::CStr;
use std::path::{Path, PathBuf};

use onnxruntime_vulkan_ep::sys::ort;
use onnxruntime_vulkan_ep::sys::{ORT_API_VERSION_EXPECTED, ORT_API_VERSION_MIN};

use crate::error::{Failure, Result};
use crate::sha256;

pub const ORT_LIB_ENV: &str = "ORT_MODEL_RUNNER_ORT_LIB";

/// Environment variables that name an ORT *installation root* (as CI's `ORT_HOME` does), rather
/// than the library file itself.
const ORT_ROOT_ENVS: [&str; 4] = ["ORT_HOME", "ORT_DIR", "ONNXRUNTIME_DIR", "ORT_LIB_DIR"];

pub fn library_file_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["onnxruntime.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libonnxruntime.dylib"]
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    {
        &["libonnxruntime.so"]
    }
}

/// A candidate location, and the reason it is a candidate. The label survives into the evidence
/// artifact: "which ORT" is answered with a path, a digest, *and* how it was found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub path: PathBuf,
    pub source: String,
}

/// The raw material for [`discover`]. Built from the process environment by
/// [`Search::from_environment`], or by hand in tests.
#[derive(Debug, Clone, Default)]
pub struct Search {
    pub explicit: Option<PathBuf>,
    pub explicit_source: String,
    /// Directories that may *contain* a library file, with the source that suggested them.
    pub dirs: Vec<(String, PathBuf)>,
}

impl Search {
    pub fn from_environment(repo: Option<&Path>, explicit: Option<PathBuf>) -> Self {
        let mut search = Search {
            explicit: explicit.clone(),
            explicit_source: if explicit.is_some() {
                "--ort-lib".to_string()
            } else {
                String::new()
            },
            dirs: Vec::new(),
        };
        if search.explicit.is_none() {
            if let Ok(p) = std::env::var(ORT_LIB_ENV) {
                if !p.trim().is_empty() {
                    search.explicit = Some(PathBuf::from(p));
                    search.explicit_source = format!("${ORT_LIB_ENV}");
                }
            }
        }
        for key in ORT_ROOT_ENVS {
            if let Ok(root) = std::env::var(key) {
                if root.trim().is_empty() {
                    continue;
                }
                let root = PathBuf::from(root);
                for sub in ["", "lib", "bin", "lib64"] {
                    let dir = if sub.is_empty() {
                        root.clone()
                    } else {
                        root.join(sub)
                    };
                    search.dirs.push((format!("${key}"), dir));
                }
            }
        }
        // A virtualenv's `onnxruntime/capi/` holds a real ORT shared library. Loading that file is
        // loading a shared library; it is not running Python, and issue #5's constraint is about
        // the *runtime*, not about which installer once put a `.dll` on the disk. It is listed
        // after the explicit roots so a deliberate ORT_HOME is never shadowed by a stray venv.
        let mut venv_roots: Vec<(String, PathBuf)> = Vec::new();
        if let Ok(v) = std::env::var("VIRTUAL_ENV") {
            if !v.trim().is_empty() {
                venv_roots.push(("$VIRTUAL_ENV".to_string(), PathBuf::from(v)));
            }
        }
        if let Some(repo) = repo {
            venv_roots.push(("<repo>/.venv".to_string(), repo.join(".venv")));
        }
        for (label, root) in venv_roots {
            for dir in site_packages_capi_dirs(&root) {
                search.dirs.push((label.clone(), dir));
            }
        }
        for key in ["PATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"] {
            if let Ok(value) = std::env::var(key) {
                for dir in std::env::split_paths(&value) {
                    if dir.as_os_str().is_empty() {
                        continue;
                    }
                    search.dirs.push((format!("${key}"), dir));
                }
            }
        }
        search
    }
}

/// `<venv>/Lib/site-packages/onnxruntime/capi` on Windows; `<venv>/lib/python*/site-packages/...`
/// elsewhere. Globbing one directory level by hand beats taking a dependency for it.
fn site_packages_capi_dirs(venv: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let windows = venv.join("Lib").join("site-packages");
    if windows.is_dir() {
        out.push(windows.join("onnxruntime").join("capi"));
    }
    let lib = venv.join("lib");
    if let Ok(entries) = std::fs::read_dir(&lib) {
        for entry in entries.flatten() {
            let p = entry
                .path()
                .join("site-packages")
                .join("onnxruntime")
                .join("capi");
            if p.is_dir() {
                out.push(p);
            }
        }
    }
    out
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Discovered {
    pub path: PathBuf,
    pub source: String,
    pub sha256: String,
    /// Every distinct candidate seen, in the order found. Recorded even on success: a run that
    /// had one choice and a run that had one *left* after a rule are different runs.
    pub considered: Vec<Candidate>,
}

pub fn discover(search: &Search) -> Result<Discovered> {
    if let Some(explicit) = &search.explicit {
        if !explicit.is_file() {
            return Err(Failure::instrument(
                "ort_library_missing",
                format!(
                    "{} names {}, which is not a file. An explicit override that does not resolve \
                     is an error: falling back to a search here would run against a library the \
                     caller did not ask for.",
                    search.explicit_source,
                    explicit.display()
                ),
            ));
        }
        let sha = hash_of(explicit)?;
        return Ok(Discovered {
            path: canonical(explicit),
            source: search.explicit_source.clone(),
            sha256: sha,
            considered: vec![Candidate {
                path: canonical(explicit),
                source: search.explicit_source.clone(),
            }],
        });
    }

    let mut considered: Vec<Candidate> = Vec::new();
    for (source, dir) in &search.dirs {
        for name in library_file_names() {
            let candidate = dir.join(name);
            if !candidate.is_file() {
                continue;
            }
            let path = canonical(&candidate);
            if considered.iter().any(|c| c.path == path) {
                continue;
            }
            considered.push(Candidate {
                path,
                source: source.clone(),
            });
        }
    }

    match considered.len() {
        0 => Err(Failure::instrument(
            "ort_library_missing",
            format!(
                "no {names:?} found. Looked in {count} director{plural}: {dirs:?}. Set \
                 --ort-lib <path>, or ${ORT_LIB_ENV}, or $ORT_HOME.",
                names = library_file_names(),
                count = search.dirs.len(),
                plural = if search.dirs.len() == 1 { "y" } else { "ies" },
                dirs = search
                    .dirs
                    .iter()
                    .map(|(s, d)| format!("{s} -> {}", d.display()))
                    .collect::<Vec<_>>(),
            ),
        )),
        1 => {
            let only = considered[0].clone();
            let sha = hash_of(&only.path)?;
            Ok(Discovered {
                path: only.path.clone(),
                source: only.source.clone(),
                sha256: sha,
                considered,
            })
        }
        _ => Err(Failure::instrument(
            "ort_library_ambiguous",
            format!(
                "{} distinct ONNX Runtime libraries are visible and this runner will not pick one \
                 for you: {:?}. Two ORT builds do not have to agree, and a number attributed to \
                 `whichever came first on PATH` is not attributed. Pass --ort-lib <path>.",
                considered.len(),
                considered
                    .iter()
                    .map(|c| format!("{} (via {})", c.path.display(), c.source))
                    .collect::<Vec<_>>()
            ),
        )),
    }
}

fn canonical(path: &Path) -> PathBuf {
    std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf())
}

fn hash_of(path: &Path) -> Result<String> {
    sha256::sha256_file(path).map_err(|e| {
        Failure::instrument("ort_library_unreadable", format!("{}: {e}", path.display()))
    })
}

/// A loaded ORT, with the negotiated API table.
#[derive(Debug)]
pub struct LoadedOrt {
    pub api: &'static ort::OrtApi,
    pub api_version: u32,
    pub version_string: String,
    pub library: Discovered,
}

// SAFETY (the whole reason this type exists): `api` points into a library that is deliberately
// never unloaded, so the reference is valid for the process lifetime. ORT's own documentation
// makes the API table immutable and callable from any thread.
unsafe impl Send for LoadedOrt {}
unsafe impl Sync for LoadedOrt {}

pub fn load(discovered: Discovered) -> Result<LoadedOrt> {
    // On Windows, ORT's dll pulls in siblings from its own directory (providers_shared, DirectML
    // where present). Prepending its directory to PATH is the portable way to say "look there
    // first" without reaching for SetDllDirectoryW.
    //
    // This prepends the *resolved library's own* directory and nothing else. It is not a search
    // fallback: `discover` has already refused anything ambiguous or absent, so this only tells
    // the loader where the siblings of an already-chosen file live. No directory the caller did
    // not effectively name is ever added.
    #[cfg(target_os = "windows")]
    if let Some(dir) = discovered.path.parent() {
        let current = std::env::var("PATH").unwrap_or_default();
        let joined = format!("{};{current}", dir.display());
        // SAFETY: called before any session exists and before any thread is spawned by this
        // process; ORT itself has not been loaded yet, so nothing is reading PATH concurrently.
        unsafe { std::env::set_var("PATH", joined) };
    }
    // The deliberate other arm. `dlopen` resolves `DT_NEEDED` siblings via the library's own
    // RUNPATH and the loader cache, so there is nothing to do -- but an absent arm and an
    // intentionally empty one look identical in a diff, and portability rule P2 exists because
    // that ambiguity is how a platform ends up with a hole. Saying it costs three lines.
    #[cfg(not(target_os = "windows"))]
    {
        // Nothing: the ELF loader already looks beside the library it is loading.
    }

    // SAFETY: loading an arbitrary shared library runs its initialisers. The path is either an
    // explicit operator choice or the single unambiguous candidate found above; that is the same
    // trust boundary every ORT host has.
    let library = unsafe { libloading::Library::new(&discovered.path) }.map_err(|e| {
        Failure::instrument(
            "ort_library_unloadable",
            format!(
                "{} could not be loaded: {e}. On Windows this is usually a missing sibling DLL or \
                 an architecture mismatch (x64 host vs arm64 library).",
                discovered.path.display()
            ),
        )
    })?;
    let library: &'static libloading::Library = Box::leak(Box::new(library));

    // SAFETY: `OrtGetApiBase` is ORT's single documented entry point and has had this signature
    // since 1.0. A library that does not export it is not ONNX Runtime, and the error says so.
    let get_api_base: libloading::Symbol<
        'static,
        unsafe extern "C" fn() -> *const ort::OrtApiBase,
    > = unsafe { library.get(b"OrtGetApiBase\0") }.map_err(|e| {
        Failure::instrument(
            "ort_library_not_onnxruntime",
            format!(
                "{} exports no OrtGetApiBase: {e}",
                discovered.path.display()
            ),
        )
    })?;

    // SAFETY: calling the resolved entry point; ORT returns a pointer to a static table.
    let base = unsafe { get_api_base() };
    if base.is_null() {
        return Err(Failure::instrument(
            "ort_library_not_onnxruntime",
            format!("{}: OrtGetApiBase returned null", discovered.path.display()),
        ));
    }
    // SAFETY: non-null by the check above, and points at ORT's process-static API base.
    let base = unsafe { &*base };

    let version_string = match base.GetVersionString {
        // SAFETY: ORT guarantees a NUL-terminated static string here.
        Some(f) => unsafe { CStr::from_ptr(f()) }
            .to_string_lossy()
            .into_owned(),
        None => String::new(),
    };

    let get_api = base.GetApi.ok_or_else(|| {
        Failure::instrument(
            "ort_library_not_onnxruntime",
            "OrtApiBase::GetApi is null".to_string(),
        )
    })?;

    // The version gate is the EP crate's, not a second copy of it: ORT_API_VERSION_EXPECTED is
    // what the vendored headers (and therefore these bindings) describe, and ORT_API_VERSION_MIN
    // is the oldest host the EP will run against. Walking down from EXPECTED to MIN mirrors what
    // the EP does on its side of the same boundary.
    let mut negotiated = 0u32;
    let mut api_ptr: *const ort::OrtApi = std::ptr::null();
    let mut version = ORT_API_VERSION_EXPECTED;
    loop {
        // SAFETY: ORT's documented contract -- returns null for an unsupported version.
        let candidate = unsafe { get_api(version) };
        if !candidate.is_null() {
            negotiated = version;
            api_ptr = candidate;
            break;
        }
        if version <= ORT_API_VERSION_MIN {
            break;
        }
        version -= 1;
    }
    if api_ptr.is_null() {
        return Err(Failure::instrument(
            "ort_api_version_unsupported",
            format!(
                "{} (version string {version_string:?}, discovered via {}) serves no API between \
                 {} and {}. This runner uses the same gate as the plugin: below {} the struct \
                 layouts this crate was compiled against are not a prefix of what the host \
                 serves.\nA stale system-wide onnxruntime on PATH is the usual cause. Point at \
                 the right one with --ort-lib <file>, or set ORT_MODEL_RUNNER_ORT_LIB / ORT_HOME.",
                discovered.path.display(),
                discovered.source,
                ORT_API_VERSION_MIN,
                ORT_API_VERSION_EXPECTED,
                ORT_API_VERSION_MIN,
            ),
        ));
    }
    // SAFETY: non-null, and ORT's API table is process-static and immutable for the lifetime of
    // the (never-unloaded) library.
    let api: &'static ort::OrtApi = unsafe { &*api_ptr };

    Ok(LoadedOrt {
        api,
        api_version: negotiated,
        version_string,
        library: discovered,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(tag: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("ort-model-runner-lib-{}-{tag}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plant(dir: &Path, contents: &[u8]) -> PathBuf {
        fs::create_dir_all(dir).unwrap();
        let p = dir.join(library_file_names()[0]);
        fs::write(&p, contents).unwrap();
        p
    }

    #[test]
    fn nothing_found_is_a_missing_error_that_lists_where_it_looked() {
        let dir = scratch("empty");
        let search = Search {
            explicit: None,
            explicit_source: String::new(),
            dirs: vec![("$ORT_HOME".into(), dir.join("nowhere"))],
        };
        let err = discover(&search).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=ort_library_missing)");
        assert!(err.message.contains("nowhere"), "{}", err.message);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn two_distinct_libraries_are_ambiguous_not_first_wins() {
        let dir = scratch("ambiguous");
        let a = dir.join("a");
        let b = dir.join("b");
        plant(&a, b"library A");
        plant(&b, b"library B");
        let search = Search {
            explicit: None,
            explicit_source: String::new(),
            dirs: vec![("$PATH".into(), a), ("$ORT_HOME".into(), b)],
        };
        let err = discover(&search).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=ort_library_ambiguous)");
        assert!(err.message.contains("--ort-lib"), "{}", err.message);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_same_library_reached_by_two_paths_is_not_ambiguous() {
        let dir = scratch("dedupe");
        let a = dir.join("only");
        plant(&a, b"library A");
        let search = Search {
            explicit: None,
            explicit_source: String::new(),
            dirs: vec![("$PATH".into(), a.clone()), ("$ORT_HOME".into(), a)],
        };
        let found = discover(&search).unwrap();
        assert_eq!(found.considered.len(), 1);
        assert_eq!(found.sha256, sha256::sha256_hex(b"library A"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_explicit_override_wins_over_everything_visible() {
        let dir = scratch("explicit");
        let chosen = plant(&dir.join("chosen"), b"the chosen one");
        let other = dir.join("other");
        plant(&other, b"not this one");
        let search = Search {
            explicit: Some(chosen.clone()),
            explicit_source: "--ort-lib".into(),
            dirs: vec![("$PATH".into(), other)],
        };
        let found = discover(&search).unwrap();
        assert_eq!(found.sha256, sha256::sha256_hex(b"the chosen one"));
        assert_eq!(found.source, "--ort-lib");
        assert_eq!(found.considered.len(), 1);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_explicit_override_that_does_not_exist_never_falls_back() {
        let dir = scratch("explicit-missing");
        let present = dir.join("present");
        plant(&present, b"a perfectly good library");
        let search = Search {
            explicit: Some(dir.join("absent").join("onnxruntime.dll")),
            explicit_source: "--ort-lib".into(),
            dirs: vec![("$PATH".into(), present)],
        };
        let err = discover(&search).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=ort_library_missing)");
        assert!(err.message.contains("not a file"), "{}", err.message);
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_library_that_is_not_ort_is_rejected_by_name_not_by_crash() {
        let dir = scratch("not-ort");
        // A file that is not a loadable image at all: the error must be the instrument's, with a
        // path in it, rather than a process abort.
        let planted = plant(&dir, b"this is not a shared object");
        let err = load(Discovered {
            path: planted,
            source: "--ort-lib".into(),
            sha256: String::new(),
            considered: Vec::new(),
        })
        .unwrap_err();
        assert_eq!(err.severity, crate::error::Severity::Instrument);
        assert!(
            err.cause == "ort_library_unloadable" || err.cause == "ort_library_not_onnxruntime",
            "{}",
            err.token()
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn the_version_gate_is_the_plugins_own_constants() {
        // Not a re-declaration: if the EP bumps its ABI pin, this runner moves with it.
        // black_box keeps the comparison a comparison rather than a constant clippy folds away.
        let min = std::hint::black_box(ORT_API_VERSION_MIN);
        let expected = std::hint::black_box(ORT_API_VERSION_EXPECTED);
        assert!(min <= expected, "{min} > {expected}");
        assert_eq!(expected, 28);
    }
}
