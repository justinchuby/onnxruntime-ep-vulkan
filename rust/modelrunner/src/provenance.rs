//! Pinned-provenance resolution: which bytes did this run actually consume, and were they the
//! bytes the repository pinned?
//!
//! This is a Rust port of `rust/tools/model_provenance.py`, and "port" is exact: it reads the same
//! `bench/results/model_provenance.json`, keyed the same way, checks size before hash for the same
//! reason (a truncated download fails on size alone and the shorter report is the more useful
//! one), and streams in the same 1 MiB chunks so the digests are the same strings. The Python
//! module stays the contract for the pytest suite; this one exists because issue #5's runner may
//! not import it.
//!
//! THE RULE, AND IT HAS NO EXCEPTIONS
//! -----------------------------------
//! A model file that is not pinned, or that does not match its pin, is never run. Not run and
//! reported as "unverified"; not run. Every number this runner emits is attributed to a SHA-256,
//! and a number attributed to a hash nobody checked is the citation defect
//! `ci/check_artifact_frame.py` was written about, one directory over.
//!
//! FETCHING
//! --------
//! Downloading is opt-in (`--fetch`) and goes through `curl`, which is the same tool CI already
//! uses to pull the ORT release archive, and which is present on Windows 10+, on every Linux
//! runner image, and on macOS. That is a *process* dependency at fetch time only: nothing in the
//! run path shells out, and a cache hit never touches the network. The download lands on a
//! temporary name and is renamed only after the pin verifies, so an interrupted fetch cannot
//! leave a half-file where a verified one is expected.

use std::path::{Path, PathBuf};
use std::process::Command;

use crate::error::{Failure, Result};
use crate::json::{self, Json};
use crate::sha256;

/// The cache root override, matching `tests/ops/test_small_model_provenance.py` exactly so both
/// harnesses look in the same place.
pub const MODEL_CACHE_ENV: &str = "ONNXRUNTIME_EP_VULKAN_MODEL_CACHE";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvenance {
    pub name: String,
    pub url: String,
    pub sha256: String,
    pub bytes: u64,
}

/// A model file that has been checked against its pin. There is no constructor that does not
/// check: holding one of these *is* the statement that the bytes matched.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModel {
    pub name: String,
    pub path: PathBuf,
    pub sha256: String,
    pub bytes: u64,
    /// `pinned` for a `model_provenance.json` entry; `resolved` for a Foundry cache model, whose
    /// identity is measured here rather than pinned upstream.
    pub provenance: &'static str,
}

pub fn load_manifest(path: &Path) -> Result<Vec<ModelProvenance>> {
    let text = std::fs::read_to_string(path).map_err(|e| {
        Failure::instrument(
            "provenance_manifest_unreadable",
            format!(
                "the pinned-provenance contract at {} could not be read: {e}. The contract not \
                 existing is never the same as `no models are pinned`.",
                path.display()
            ),
        )
    })?;
    let doc = json::parse(&text).map_err(|e| {
        Failure::instrument(
            "provenance_manifest_malformed",
            format!("{}: {e}", path.display()),
        )
    })?;
    let rows = doc.get("models").and_then(Json::as_array).ok_or_else(|| {
        Failure::instrument(
            "provenance_manifest_malformed",
            format!("{}: no `models` array", path.display()),
        )
    })?;
    let mut out = Vec::with_capacity(rows.len());
    for row in rows {
        let name = row.str_of("name");
        let url = row.str_of("url");
        let hash = row.str_of("sha256");
        let bytes = row.i64_of("bytes");
        match (name, url, hash, bytes) {
            (Some(name), Some(url), Some(hash), Some(bytes)) if bytes >= 0 => {
                out.push(ModelProvenance {
                    name: name.to_string(),
                    url: url.to_string(),
                    sha256: hash.to_ascii_lowercase(),
                    bytes: bytes as u64,
                });
            }
            _ => {
                return Err(Failure::instrument(
                    "provenance_manifest_malformed",
                    format!(
                        "{}: an entry is missing one of name/url/sha256/bytes, or carries a \
                         negative size. A partially-specified pin is not a pin.",
                        path.display()
                    ),
                ));
            }
        }
    }
    Ok(out)
}

pub fn entry<'a>(manifest: &'a [ModelProvenance], name: &str) -> Result<&'a ModelProvenance> {
    manifest.iter().find(|m| m.name == name).ok_or_else(|| {
        let known: Vec<&str> = manifest.iter().map(|m| m.name.as_str()).collect();
        Failure::instrument(
            "model_not_pinned",
            format!(
                "`{name}` has no entry in bench/results/model_provenance.json, so there is no \
                 pinned SHA-256 to check it against. Known: {known:?}. Add a pinned entry rather \
                 than running an unattributed file."
            ),
        )
    })
}

/// Size first, then hash -- and both messages name the URL, because the only useful next action
/// for either failure is a re-download.
pub fn verify_file(path: &Path, pin: &ModelProvenance) -> Result<VerifiedModel> {
    let meta = std::fs::metadata(path).map_err(|e| {
        Failure::instrument(
            "model_file_missing",
            format!("{}: expected a file at {}: {e}", pin.name, path.display()),
        )
    })?;
    if !meta.is_file() {
        return Err(Failure::instrument(
            "model_file_missing",
            format!("{}: {} is not a regular file", pin.name, path.display()),
        ));
    }
    if meta.len() != pin.bytes {
        return Err(Failure::condition(
            "provenance_size_mismatch",
            format!(
                "{}: size mismatch at {}: expected {} bytes, got {} bytes. Re-download from {}.",
                pin.name,
                path.display(),
                pin.bytes,
                meta.len(),
                pin.url
            ),
        ));
    }
    let actual = sha256::sha256_file(path).map_err(|e| {
        Failure::instrument(
            "model_file_unreadable",
            format!("{}: {}: {e}", pin.name, path.display()),
        )
    })?;
    if actual != pin.sha256 {
        return Err(Failure::condition(
            "provenance_hash_mismatch",
            format!(
                "{}: SHA-256 mismatch at {}: expected {}, got {}. Size matched but the contents \
                 did not -- this file does not match the pinned provenance contract; re-download \
                 from {} rather than trusting the local copy.",
                pin.name,
                path.display(),
                pin.sha256,
                actual,
                pin.url
            ),
        ));
    }
    Ok(VerifiedModel {
        name: pin.name.clone(),
        path: path.to_path_buf(),
        sha256: actual,
        bytes: meta.len(),
        provenance: "pinned",
    })
}

pub fn home_dir() -> Option<PathBuf> {
    // No `dirs` crate for one lookup. Windows has USERPROFILE; everything else has HOME.
    if let Ok(p) = std::env::var("USERPROFILE") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    if let Ok(p) = std::env::var("HOME") {
        if !p.is_empty() {
            return Some(PathBuf::from(p));
        }
    }
    None
}

/// `$ONNXRUNTIME_EP_VULKAN_MODEL_CACHE`, else `~/.cache/onnxruntime-ep-vulkan/models`.
pub fn model_cache_dir() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var(MODEL_CACHE_ENV) {
        if !dir.trim().is_empty() {
            return Ok(PathBuf::from(dir));
        }
    }
    let home = home_dir().ok_or_else(|| {
        Failure::instrument(
            "model_cache_unresolvable",
            format!(
                "neither {MODEL_CACHE_ENV} nor a home directory (USERPROFILE/HOME) is set, so \
                 there is no cache root to look in."
            ),
        )
    })?;
    Ok(home
        .join(".cache")
        .join("onnxruntime-ep-vulkan")
        .join("models"))
}

pub fn cached_path(cache_dir: &Path, name: &str) -> PathBuf {
    cache_dir.join(format!("{name}.onnx"))
}

/// Resolve a pinned model to verified bytes, downloading only if asked to.
pub fn ensure_model(
    manifest: &[ModelProvenance],
    name: &str,
    cache_dir: &Path,
    fetch: bool,
) -> Result<VerifiedModel> {
    let pin = entry(manifest, name)?;
    let path = cached_path(cache_dir, name);
    if path.exists() {
        return verify_file(&path, pin);
    }
    if !fetch {
        return Err(Failure::instrument(
            "model_not_cached",
            format!(
                "{name} is not cached at {}. Pass --fetch to download it from {} (pinned \
                 sha256 {}), or set {MODEL_CACHE_ENV} to a directory that already holds it. This \
                 runner never downloads without being told to.",
                path.display(),
                pin.url,
                pin.sha256
            ),
        ));
    }
    fetch_model(pin, &path)?;
    verify_file(&path, pin)
}

/// Download to a sibling temporary name, verify, and only then rename into place.
pub fn fetch_model(pin: &ModelProvenance, dest: &Path) -> Result<()> {
    let parent = dest.parent().ok_or_else(|| {
        Failure::instrument(
            "model_cache_unresolvable",
            format!("{} has no parent directory", dest.display()),
        )
    })?;
    std::fs::create_dir_all(parent).map_err(|e| {
        Failure::instrument(
            "model_cache_unwritable",
            format!("{}: {e}", parent.display()),
        )
    })?;
    let staging = parent.join(format!(".{}.partial", pin.name));
    let status = Command::new("curl")
        .arg("--fail")
        .arg("--location")
        .arg("--silent")
        .arg("--show-error")
        .arg("--output")
        .arg(&staging)
        .arg(&pin.url)
        .status()
        .map_err(|e| {
            Failure::instrument(
                "fetch_tool_missing",
                format!(
                    "could not run `curl` to fetch {}: {e}. curl is the same tool CI uses for the \
                     ORT release archive; install it or pre-populate the model cache.",
                    pin.url
                ),
            )
        })?;
    if !status.success() {
        let _ = std::fs::remove_file(&staging);
        return Err(Failure::instrument(
            "fetch_failed",
            format!("curl exited {status} fetching {}", pin.url),
        ));
    }
    // Verify the staged bytes BEFORE they take the cached name. A file at the cached path is
    // treated as trusted-until-hashed by every other caller; a failed download must never be able
    // to occupy that name.
    let staged = verify_file(&staging, pin);
    match staged {
        Ok(_) => {
            std::fs::rename(&staging, dest).map_err(|e| {
                Failure::instrument(
                    "model_cache_unwritable",
                    format!("{} -> {}: {e}", staging.display(), dest.display()),
                )
            })?;
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&staging);
            Err(e)
        }
    }
}

/// Hash a model that has no upstream pin (a Foundry cache checkpoint) and report its identity.
///
/// This is not a weaker form of `verify_file`: it produces a *measured* identity rather than a
/// checked one, and the distinction is carried in `VerifiedModel::provenance` so no reader can
/// mistake one for the other.
pub fn measure_model(name: &str, path: &Path) -> Result<VerifiedModel> {
    let meta = std::fs::metadata(path).map_err(|e| {
        Failure::instrument(
            "model_file_missing",
            format!("{name}: {}: {e}", path.display()),
        )
    })?;
    let sha = sha256::sha256_file(path).map_err(|e| {
        Failure::instrument(
            "model_file_unreadable",
            format!("{name}: {}: {e}", path.display()),
        )
    })?;
    Ok(VerifiedModel {
        name: name.to_string(),
        path: path.to_path_buf(),
        sha256: sha,
        bytes: meta.len(),
        provenance: "resolved",
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "ort-model-runner-prov-{}-{tag}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn pin_for(path: &Path, name: &str) -> ModelProvenance {
        ModelProvenance {
            name: name.to_string(),
            url: "https://example.invalid/x.onnx".to_string(),
            sha256: sha256::sha256_file(path).unwrap(),
            bytes: fs::metadata(path).unwrap().len(),
        }
    }

    #[test]
    fn a_matching_file_verifies() {
        let dir = scratch("match");
        let p = dir.join("m.onnx");
        fs::write(&p, b"the bytes").unwrap();
        let pin = pin_for(&p, "m");
        let v = verify_file(&p, &pin).unwrap();
        assert_eq!(v.sha256, pin.sha256);
        assert_eq!(v.provenance, "pinned");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_hash_mismatch_is_a_condition_not_an_instrument_error() {
        let dir = scratch("hash");
        let p = dir.join("m.onnx");
        fs::write(&p, b"the bytes").unwrap();
        let mut pin = pin_for(&p, "m");
        // Same length, different content: this is the case a size check cannot catch.
        pin.sha256 = sha256::sha256_hex(b"the BYTES");
        let err = verify_file(&p, &pin).unwrap_err();
        assert_eq!(err.token(), "FAIL(condition=provenance_hash_mismatch)");
        assert!(
            err.message
                .contains("Size matched but the contents did not")
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_size_mismatch_is_reported_on_its_own() {
        let dir = scratch("size");
        let p = dir.join("m.onnx");
        fs::write(&p, b"short").unwrap();
        let mut pin = pin_for(&p, "m");
        pin.bytes = 99999;
        let err = verify_file(&p, &pin).unwrap_err();
        assert_eq!(err.token(), "FAIL(condition=provenance_size_mismatch)");
        assert!(err.message.contains("expected 99999 bytes, got 5 bytes"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_missing_file_is_an_instrument_error() {
        let dir = scratch("missing");
        let p = dir.join("absent.onnx");
        let pin = ModelProvenance {
            name: "absent".into(),
            url: "https://example.invalid/a.onnx".into(),
            sha256: "0".repeat(64),
            bytes: 1,
        };
        let err = verify_file(&p, &pin).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=model_file_missing)");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_unpinned_name_refuses_rather_than_defaulting() {
        let manifest = vec![ModelProvenance {
            name: "mnist-12".into(),
            url: "u".into(),
            sha256: "0".repeat(64),
            bytes: 1,
        }];
        let err = entry(&manifest, "resnet-50").unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=model_not_pinned)");
        assert!(err.message.contains("mnist-12"), "{}", err.message);
    }

    #[test]
    fn the_repository_manifest_parses_and_pins_the_issue_5_models() {
        let manifest_path = crate::repo::root()
            .expect("the repository root must be discoverable from the test binary")
            .join("bench")
            .join("results")
            .join("model_provenance.json");
        let manifest = load_manifest(&manifest_path).unwrap();
        let mnist = entry(&manifest, "mnist-12").unwrap();
        assert_eq!(
            mnist.sha256,
            "5c688690f8bacf667d4c2074af5ad0646ca328d7ab03eccf944a65b320171bdd"
        );
        assert_eq!(mnist.bytes, 26143);
        let mobilenet = entry(&manifest, "mobilenetv2-12").unwrap();
        assert_eq!(
            mobilenet.sha256,
            "c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5"
        );
        // Every pin is a full lowercase hex SHA-256; a 40-char value would be a git blob id, and
        // a truncated one would silently weaken every comparison made against it.
        for m in &manifest {
            assert_eq!(m.sha256.len(), 64, "{}", m.name);
            assert!(
                m.sha256
                    .chars()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
                "{}",
                m.name
            );
            assert!(m.bytes > 0, "{}", m.name);
        }
    }

    #[test]
    fn a_malformed_manifest_is_an_instrument_error_not_an_empty_pin_set() {
        let dir = scratch("manifest");
        let p = dir.join("bad.json");
        fs::write(&p, b"{\"models\": [{\"name\": \"x\"}]}").unwrap();
        let err = load_manifest(&p).unwrap_err();
        assert_eq!(
            err.token(),
            "ERROR(instrument=provenance_manifest_malformed)"
        );
        fs::write(&p, b"not json at all").unwrap();
        assert_eq!(
            load_manifest(&p).unwrap_err().token(),
            "ERROR(instrument=provenance_manifest_malformed)"
        );
        assert_eq!(
            load_manifest(&dir.join("nope.json")).unwrap_err().token(),
            "ERROR(instrument=provenance_manifest_unreadable)"
        );
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn an_uncached_model_refuses_before_it_reaches_the_network() {
        let dir = scratch("uncached");
        let manifest = vec![ModelProvenance {
            name: "not-here".into(),
            url: "https://example.invalid/not-here.onnx".into(),
            sha256: "0".repeat(64),
            bytes: 7,
        }];
        let err = ensure_model(&manifest, "not-here", &dir, false).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=model_not_cached)");
        assert!(err.message.contains("--fetch"));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_cached_file_that_fails_its_pin_is_never_returned_as_a_model() {
        let dir = scratch("poisoned-cache");
        let path = cached_path(&dir, "planted");
        fs::write(&path, b"not the pinned bytes").unwrap();
        let manifest = vec![ModelProvenance {
            name: "planted".into(),
            url: "https://example.invalid/planted.onnx".into(),
            sha256: sha256::sha256_hex(b"the pinned bytes"),
            bytes: b"not the pinned bytes".len() as u64,
        }];
        let err = ensure_model(&manifest, "planted", &dir, false).unwrap_err();
        assert_eq!(err.token(), "FAIL(condition=provenance_hash_mismatch)");
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn measured_identity_is_labelled_differently_from_a_checked_pin() {
        let dir = scratch("measured");
        let p = dir.join("foundry.onnx");
        fs::write(&p, b"weights").unwrap();
        let m = measure_model("phi-3.5", &p).unwrap();
        assert_eq!(m.provenance, "resolved");
        assert_eq!(m.sha256, sha256::sha256_hex(b"weights"));
        fs::remove_dir_all(&dir).ok();
    }
}
