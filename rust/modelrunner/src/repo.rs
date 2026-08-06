//! Where is the repository, from inside a binary that may have been copied anywhere?
//!
//! Every other path this runner resolves (the provenance manifest, the built plugin, the evidence
//! directory) hangs off one answer, and getting it wrong silently is the failure mode that makes a
//! runner "work on my machine". So the answer is derived from a landmark rather than from
//! `cwd`-relative guesswork, is overridable, and is an explicit error when it cannot be found --
//! never a silent fallback to `.`.

use std::path::{Path, PathBuf};

use crate::error::{Failure, Result};

pub const REPO_ENV: &str = "ORT_MODEL_RUNNER_REPO";

/// The landmark. Not `.git`: a git worktree's `.git` is a *file*, `git worktree list` shows five
/// of them on the machine this was written on, and a checkout exported without git history is
/// still a checkout. The provenance contract is the thing this runner actually needs, so it is
/// the thing it looks for.
const LANDMARK: [&str; 3] = ["bench", "results", "model_provenance.json"];

fn has_landmark(dir: &Path) -> bool {
    let mut p = dir.to_path_buf();
    for part in LANDMARK {
        p.push(part);
    }
    p.is_file()
}

/// `$ORT_MODEL_RUNNER_REPO`, else the nearest ancestor of the current directory or of this
/// executable that carries the landmark.
pub fn root() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var(REPO_ENV) {
        if !dir.trim().is_empty() {
            let p = PathBuf::from(dir);
            if has_landmark(&p) {
                return Ok(p);
            }
            return Err(Failure::instrument(
                "repo_root_unresolvable",
                format!(
                    "{REPO_ENV} points at {} but there is no bench/results/model_provenance.json \
                     under it. An override that does not resolve is an error, not a hint.",
                    p.display()
                ),
            ));
        }
    }
    let mut searched = Vec::new();
    for start in starting_points() {
        searched.push(start.display().to_string());
        let mut cursor: Option<&Path> = Some(start.as_path());
        while let Some(dir) = cursor {
            if has_landmark(dir) {
                return Ok(dir.to_path_buf());
            }
            cursor = dir.parent();
        }
    }
    Err(Failure::instrument(
        "repo_root_unresolvable",
        format!(
            "no ancestor of {searched:?} carries bench/results/model_provenance.json. Set \
             {REPO_ENV} to the checkout root."
        ),
    ))
}

fn starting_points() -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        out.push(cwd);
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            out.push(dir.to_path_buf());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_landmark_resolves_from_the_test_binary() {
        let repo = root().expect("cargo test runs inside the checkout");
        assert!(repo.join("bench/results/model_provenance.json").is_file());
        assert!(repo.join("rust/modelrunner/Cargo.toml").is_file());
    }

    #[test]
    fn a_directory_without_the_landmark_is_not_a_root() {
        assert!(!has_landmark(&std::env::temp_dir()));
    }
}
