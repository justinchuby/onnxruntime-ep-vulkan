//! The evidence document, and the artifact frame that makes it checkable.
//!
//! WHAT AN EVIDENCE FILE IS FOR
//! ---------------------------
//! Not for the person who ran the command -- they watched it happen. It is for the reviewer six
//! months later who has to decide whether a green result is still true. That reader needs the
//! things that could have differed: which model file (path *and* SHA-256), which ONNX Runtime
//! (path, SHA-256, API version), which plugin build (path, SHA-256), which device, which seed,
//! which tolerance and where it came from, and which of the five guards actually held.
//!
//! So this module writes all of them, unconditionally, on success *and* on failure. An evidence
//! file that only exists when the run passed is a record of nothing.
//!
//! IDENTITY IS TOP-LEVEL
//! ---------------------
//! `onnx_file` and `onnx_sha256` sit at the top level of the document, matching the convention the
//! provenance tooling and `ci/check_hardcoded_foundry_paths.py` readers already expect, and the
//! one issue #19 had to retrofit into two tools that had dropped them. They are written from the
//! file that was *actually opened*, never from the argument that was requested.

use std::path::Path;

use crate::error::{Failure, Result};
use crate::json::Json;

/// Where a run ended up, in one word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    Pass,
    Fail,
    Error,
    Unsupported,
}

impl Outcome {
    pub fn as_str(self) -> &'static str {
        match self {
            Outcome::Pass => "PASS",
            Outcome::Fail => "FAIL",
            Outcome::Error => "ERROR",
            Outcome::Unsupported => "UNSUPPORTED",
        }
    }
}

/// One named guard and whether it held. A guard that could not be evaluated is `held: false` with
/// its reason, never absent -- a missing guard reads as a passed one.
#[derive(Debug, Clone)]
pub struct Guard {
    pub name: String,
    pub held: bool,
    pub detail: String,
}

impl Guard {
    pub fn new(name: &str, held: bool, detail: impl Into<String>) -> Self {
        Self {
            name: name.to_string(),
            held,
            detail: detail.into(),
        }
    }

    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("name", Json::s(self.name.as_str())),
            ("held", Json::Bool(self.held)),
            ("detail", Json::s(self.detail.as_str())),
        ])
    }
}

/// A file this run depended on, identified the only way that survives a rebuild.
#[derive(Debug, Clone)]
pub struct FileIdentity {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}

impl FileIdentity {
    pub fn of(path: &Path) -> Result<Self> {
        let bytes = std::fs::metadata(path)
            .map_err(|e| {
                Failure::instrument(
                    "identity_unreadable",
                    format!("cannot stat {} for identity: {e}", path.display()),
                )
            })?
            .len();
        Ok(Self {
            path: path.display().to_string(),
            sha256: crate::sha256::sha256_file(path).map_err(|e| {
                Failure::instrument(
                    "identity_unreadable",
                    format!("cannot hash {} for identity: {e}", path.display()),
                )
            })?,
            bytes,
        })
    }

    pub fn to_json(&self) -> Json {
        Json::obj(vec![
            ("path", Json::s(self.path.as_str())),
            ("sha256", Json::s(self.sha256.as_str())),
            ("bytes", Json::int(self.bytes as i64)),
        ])
    }
}

/// Write a JSON document to `path`, creating parent directories.
pub fn write_json(path: &Path, doc: &Json) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).map_err(|e| {
                Failure::instrument(
                    "evidence_unwritable",
                    format!("cannot create {}: {e}", parent.display()),
                )
            })?;
        }
    }
    // Trailing newline: these files are read by line-oriented tools and diffed by humans.
    let text = format!("{}\n", crate::json::to_string_pretty(doc));
    std::fs::write(path, text).map_err(|e| {
        Failure::instrument(
            "evidence_unwritable",
            format!("cannot write {}: {e}", path.display()),
        )
    })
}

/// Read the EP's counters snapshot and pull out the headline observables.
///
/// A missing or unparsable counters file is *not* fatal here: it is reported as `null`, and the
/// guard that requires `dispatches_executed > 0` then fails with that as its reason. Conflating
/// "the instrument did not report" with "the instrument reported zero" is exactly the confusion
/// this repository's counters module was written to prevent.
#[derive(Debug, Clone, Default)]
pub struct Counters {
    pub present: bool,
    pub dispatches_executed: Option<i64>,
    pub claimed_nodes: Option<i64>,
    pub islands_offered: Option<i64>,
    pub compute_calls: Option<i64>,
    pub note: String,
}

impl Counters {
    pub fn read(path: &Path) -> Self {
        let Ok(text) = std::fs::read_to_string(path) else {
            return Self {
                present: false,
                note: format!(
                    "no counters snapshot at {} -- the EP writes one at teardown only when \
                     ONNXRUNTIME_EP_VULKAN_COUNTERS_FILE is set and the EP was actually loaded",
                    path.display()
                ),
                ..Default::default()
            };
        };
        let doc = match crate::json::parse(&text) {
            Ok(d) => d,
            Err(e) => {
                return Self {
                    present: true,
                    note: format!("counters snapshot at {} is not JSON: {e}", path.display()),
                    ..Default::default()
                };
            }
        };
        let get = |key: &str| doc.get(key).and_then(|v| v.as_i64());
        Self {
            present: true,
            dispatches_executed: get("dispatches_executed"),
            claimed_nodes: get("claimed_nodes"),
            islands_offered: get("islands_offered"),
            compute_calls: get("compute_calls"),
            note: String::new(),
        }
    }

    pub fn to_json(&self) -> Json {
        let opt = |v: Option<i64>| match v {
            Some(n) => Json::int(n),
            None => Json::Null,
        };
        Json::obj(vec![
            ("present", Json::Bool(self.present)),
            ("dispatches_executed", opt(self.dispatches_executed)),
            ("claimed_nodes", opt(self.claimed_nodes)),
            ("islands_offered", opt(self.islands_offered)),
            ("compute_calls", opt(self.compute_calls)),
            ("note", Json::s(self.note.as_str())),
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("ort_model_runner_evidence_{tag}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn a_missing_counters_file_is_absent_not_zero() {
        // The distinction that matters: `dispatches_executed: 0` is a claim about the run;
        // `null` with `present: false` is a claim about the instrument. Collapsing them would let
        // "the counters file was never written" read as "the EP ran nothing", or worse, the
        // reverse.
        let c = Counters::read(Path::new("definitely-not-a-file.json"));
        assert!(!c.present);
        assert_eq!(c.dispatches_executed, None);
        assert!(c.note.contains("COUNTERS_FILE"), "{}", c.note);
        assert_eq!(c.to_json().get("dispatches_executed"), Some(&Json::Null));
    }

    #[test]
    fn counters_are_read_from_the_snapshot_the_ep_writes() {
        let dir = scratch("counters");
        let path = dir.join("counters.json");
        std::fs::write(
            &path,
            r#"{"dispatches_executed": 12, "claimed_nodes": 3, "islands_offered": 1,
                "compute_calls": 3, "leak_verdict": "CLEAN"}"#,
        )
        .unwrap();
        let c = Counters::read(&path);
        assert!(c.present);
        assert_eq!(c.dispatches_executed, Some(12));
        assert_eq!(c.claimed_nodes, Some(3));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupt_counters_file_reports_the_corruption_rather_than_a_number() {
        let dir = scratch("corrupt");
        let path = dir.join("counters.json");
        std::fs::write(&path, "{not json").unwrap();
        let c = Counters::read(&path);
        assert!(c.present);
        assert_eq!(c.dispatches_executed, None);
        assert!(c.note.contains("not JSON"), "{}", c.note);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn file_identity_is_the_hash_of_the_file_that_was_opened() {
        let dir = scratch("identity");
        let path = dir.join("thing.bin");
        std::fs::write(&path, b"abc").unwrap();
        let id = FileIdentity::of(&path).unwrap();
        assert_eq!(
            id.sha256,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
        assert_eq!(id.bytes, 3);
        assert!(id.path.ends_with("thing.bin"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn identity_of_a_missing_file_is_an_error_not_a_blank_hash() {
        // Issue #19's defect in one line: a tool that writes an empty `onnx_sha256` on a
        // successful-looking run has produced a document that cannot be checked.
        let err = FileIdentity::of(Path::new("no-such-file.bin")).unwrap_err();
        assert_eq!(err.token(), "ERROR(instrument=identity_unreadable)");
    }

    #[test]
    fn evidence_is_written_with_its_parent_directories_and_a_trailing_newline() {
        let dir = scratch("write");
        let path = dir.join("nested/deeper/result.json");
        write_json(
            &path,
            &Json::obj(vec![("pass", Json::Bool(true)), ("n", Json::int(3))]),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.ends_with("\n"));
        let parsed = crate::json::parse(&text).unwrap();
        assert_eq!(parsed.get("pass"), Some(&Json::Bool(true)));
        // Integers must not acquire a decimal point on the way out: `3.0` is a different token to
        // every consumer that reads these files with a strict schema.
        assert!(text.contains("\"n\": 3"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_guard_records_its_reason_even_when_it_held() {
        let g = Guard::new("vulkan_ep_selected", true, "OrtEpDevice[0] present");
        let j = g.to_json();
        assert_eq!(j.get("held"), Some(&Json::Bool(true)));
        assert!(!j.get("detail").unwrap().as_str().unwrap().is_empty());
    }
}
