//! Foundry Local model discovery, without Python.
//!
//! This is a deliberately partial port of `rust/tools/foundry_discovery.py`. That module has two
//! strategies: ask the `foundry` CLI for its own cache manifest, then fall back to a constrained
//! filesystem search. **Only the filesystem strategy is implemented here**, and the reason is
//! recorded rather than hidden: the CLI strategy needs a JSON manifest whose schema is Foundry's,
//! not ours, and a second, silently-diverging reader of someone else's schema is a worse liability
//! than a documented gap. When this runner cannot resolve a model it says exactly that, and the
//! Python resolver -- which does ask the CLI -- remains the authority for the Foundry path.
//!
//! What *is* preserved exactly, because these are the properties that make the resolver safe:
//!
//! * the cache root and its two environment overrides;
//! * the `<root>/Microsoft/<variant>[-<revision>]/**/<onnx filename>` shape, matched by exact name
//!   or exact name plus a `-` suffix -- never a fuzzy or wildcard family match;
//! * **exactly one** match is required. Zero and two are different errors with different remedies,
//!   and neither is ever resolved by picking the first.
//!
//! NO CLOCK, NO ARBITRARY CHOICE. Identity and presence only.

use std::path::{Path, PathBuf};

use crate::error::{Failure, Result};

/// The exact model identity to resolve -- never a family name alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FoundryModelSpec {
    /// Foundry's own `variantName`, which is the on-disk directory name minus any catalog-revision
    /// suffix.
    pub variant_name: String,
    /// Foundry's own `executionProvider` string, recorded in the evidence so a run against the
    /// wrong build is visible.
    pub execution_provider: String,
    pub onnx_filename: String,
    /// What a human types to fetch it. Used only in error messages, never in discovery.
    pub download_alias: String,
}

impl FoundryModelSpec {
    /// The Phi-3.5 identity this repository already depends on, spelled the same way
    /// `rust/tools/foundry_discovery.py` and its callers spell it.
    pub fn phi35_cuda_gpu() -> Self {
        Self {
            variant_name: "Phi-3.5-mini-instruct-cuda-gpu".to_string(),
            execution_provider: "CUDAExecutionProvider".to_string(),
            onnx_filename: "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx".to_string(),
            download_alias: "phi-3.5-mini".to_string(),
        }
    }
}

/// The Foundry Local model-cache root, overridable for tests and non-default installs.
pub fn default_cache_root() -> Result<PathBuf> {
    for key in ["ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE", "FOUNDRY_CACHE_DIR"] {
        if let Ok(value) = std::env::var(key) {
            if !value.trim().is_empty() {
                return Ok(PathBuf::from(value));
            }
        }
    }
    let home = crate::provenance::home_dir().ok_or_else(|| {
        Failure::instrument(
            "foundry_cache_unresolvable",
            "neither ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE, FOUNDRY_CACHE_DIR, nor a home \
             directory (USERPROFILE/HOME) is set, so there is no Foundry cache root to look in.",
        )
    })?;
    Ok(home.join(".foundry").join("cache").join("models"))
}

/// One resolved candidate, kept as a value so selection is testable without a filesystem.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Candidate {
    pub variant_id: String,
    pub onnx_path: PathBuf,
}

/// Recursively collect files named `filename` under `dir`.
///
/// Bounded by `MAX_DEPTH` because a symlink loop in a model cache should surface as a bounded
/// search that found nothing, not as a runner that never returns.
fn find_named(dir: &Path, filename: &str, depth: usize, out: &mut Vec<PathBuf>) {
    const MAX_DEPTH: usize = 8;
    if depth > MAX_DEPTH {
        return;
    }
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    // Sorted so the *reported* order of an ambiguity is stable across machines; the ambiguity is
    // still an error, this only makes its message reproducible.
    children.sort();
    for child in children {
        if child.is_dir() {
            find_named(&child, filename, depth + 1, out);
        } else if child.file_name().map(|n| n == filename).unwrap_or(false) {
            out.push(child);
        }
    }
}

/// The version-tolerant filesystem search: exact variant name, or exact name plus a `-` revision.
pub fn candidates_from_filesystem(cache_root: &Path, spec: &FoundryModelSpec) -> Vec<Candidate> {
    let family_root = cache_root.join("Microsoft");
    if !family_root.is_dir() {
        return Vec::new();
    }
    let Ok(entries) = std::fs::read_dir(&family_root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = entries.filter_map(|e| e.ok()).map(|e| e.path()).collect();
    dirs.sort();
    let prefix = format!("{}-", spec.variant_name);
    let mut out = Vec::new();
    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        let Some(name) = dir.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if name != spec.variant_name && !name.starts_with(&prefix) {
            continue;
        }
        let mut found = Vec::new();
        find_named(&dir, &spec.onnx_filename, 0, &mut found);
        for onnx_path in found {
            out.push(Candidate {
                variant_id: name.to_string(),
                onnx_path,
            });
        }
    }
    out
}

/// Pure selection over an already-collected candidate list -- no filesystem, no subprocess.
///
/// Separate from [`resolve_model_path`] for the same reason the Python is: the negative cases must
/// be unit-testable without a real Foundry install.
pub fn select(candidates: &[Candidate], spec: &FoundryModelSpec) -> Result<Candidate> {
    match candidates.len() {
        0 => Err(Failure::unsupported(
            "foundry_model_missing",
            format!(
                "no cached Foundry variant named {:?} containing {:?} was found. Run \
                 `foundry model download {}` to fetch it. (This Rust resolver searches the \
                 filesystem cache only; rust/tools/foundry_discovery.py additionally asks the \
                 foundry CLI.)",
                spec.variant_name, spec.onnx_filename, spec.download_alias
            ),
        )),
        1 => Ok(candidates[0].clone()),
        n => {
            let listed = candidates
                .iter()
                .map(|c| format!("{} -> {}", c.variant_id, c.onnx_path.display()))
                .collect::<Vec<_>>()
                .join("; ");
            Err(Failure::instrument(
                "foundry_model_ambiguous",
                format!(
                    "ambiguous: {n} cached entries match {:?} ({}): {listed}. Refusing to choose \
                     arbitrarily -- remove the stale entries with `foundry cache clear <variant>` \
                     (keeping the one you want) before running this again.",
                    spec.variant_name, spec.execution_provider
                ),
            ))
        }
    }
}

/// Resolve the one cached ONNX file for `spec`, or explain precisely why it could not.
pub fn resolve_model_path(spec: &FoundryModelSpec, cache_root: Option<&Path>) -> Result<Candidate> {
    let root = match cache_root {
        Some(r) => r.to_path_buf(),
        None => default_cache_root()?,
    };
    if !root.is_dir() {
        return Err(Failure::unsupported(
            "foundry_cache_absent",
            format!(
                "the Foundry model cache root {} does not exist. Set \
                 ONNXRUNTIME_EP_VULKAN_FOUNDRY_CACHE or FOUNDRY_CACHE_DIR, or run \
                 `foundry model download {}`.",
                root.display(),
                spec.download_alias
            ),
        ));
    }
    select(&candidates_from_filesystem(&root, spec), spec)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> FoundryModelSpec {
        FoundryModelSpec {
            variant_name: "Phi-3.5-mini-instruct-cuda-gpu".into(),
            execution_provider: "CUDAExecutionProvider".into(),
            onnx_filename: "model.onnx".into(),
            download_alias: "phi-3.5-mini".into(),
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ort_model_runner_foundry_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn plant(root: &Path, variant: &str, rel: &str) {
        let dir = root.join("Microsoft").join(variant).join(rel);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("model.onnx"), b"not a real model").unwrap();
    }

    #[test]
    fn exactly_one_cached_variant_resolves() {
        let root = scratch("one");
        plant(&root, "Phi-3.5-mini-instruct-cuda-gpu-2", "v2");
        let found = resolve_model_path(&spec(), Some(&root)).unwrap();
        assert_eq!(found.variant_id, "Phi-3.5-mini-instruct-cuda-gpu-2");
        assert!(found.onnx_path.ends_with("model.onnx"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn the_exact_variant_name_without_a_revision_suffix_also_resolves() {
        let root = scratch("exact");
        plant(&root, "Phi-3.5-mini-instruct-cuda-gpu", "v1");
        assert!(resolve_model_path(&spec(), Some(&root)).is_ok());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn two_cached_variants_are_ambiguous_rather_than_first_wins() {
        // The defect this guards: picking `[0]` here resolves to whichever directory the
        // filesystem happened to list first, so the run's model identity depends on the machine.
        let root = scratch("two");
        plant(&root, "Phi-3.5-mini-instruct-cuda-gpu-2", "v2");
        plant(&root, "Phi-3.5-mini-instruct-cuda-gpu-3", "v3");
        let err = resolve_model_path(&spec(), Some(&root)).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=foundry_model_ambiguous)");
        assert!(err.message.contains("gpu-2"), "{}", err.message);
        assert!(err.message.contains("gpu-3"), "{}", err.message);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_missing_variant_names_the_remedy() {
        let root = scratch("none");
        std::fs::create_dir_all(root.join("Microsoft")).unwrap();
        let err = resolve_model_path(&spec(), Some(&root)).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=foundry_model_missing)");
        assert!(
            err.message.contains("foundry model download phi-3.5-mini"),
            "{}",
            err.message
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn a_similar_but_different_family_is_not_matched() {
        // "Phi-3.5-mini-instruct-cuda-gpu" must not match "Phi-3.5-mini-instruct-cpu" or
        // "Phi-3.5-mini-instruct-cuda-gpuX": the prefix rule requires an exact `-` boundary.
        let root = scratch("similar");
        plant(&root, "Phi-3.5-mini-instruct-cpu", "v1");
        plant(&root, "Phi-3.5-mini-instruct-cuda-gpuX", "v1");
        let err = resolve_model_path(&spec(), Some(&root)).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=foundry_model_missing)");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn an_absent_cache_root_is_its_own_error() {
        let missing = std::env::temp_dir().join("ort_model_runner_foundry_definitely_absent");
        let _ = std::fs::remove_dir_all(&missing);
        let err = resolve_model_path(&spec(), Some(&missing)).unwrap_err();
        assert_eq!(err.token(), "UNSUPPORTED(reason=foundry_cache_absent)");
    }

    #[test]
    fn selection_is_pure_and_testable_without_a_filesystem() {
        let one = vec![Candidate {
            variant_id: "v".into(),
            onnx_path: PathBuf::from("/a/model.onnx"),
        }];
        assert_eq!(select(&one, &spec()).unwrap().variant_id, "v");
        assert!(select(&[], &spec()).is_err());
    }

    #[test]
    fn the_phi35_identity_matches_the_repository_spelling() {
        let s = FoundryModelSpec::phi35_cuda_gpu();
        assert_eq!(s.variant_name, "Phi-3.5-mini-instruct-cuda-gpu");
        assert_eq!(s.execution_provider, "CUDAExecutionProvider");
        assert_eq!(
            s.onnx_filename,
            "phi-3.5-mini-instruct-cuda-int4-rtn-block-32.onnx"
        );
    }
}
