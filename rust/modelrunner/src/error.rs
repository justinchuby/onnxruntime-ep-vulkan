//! One failure type, three severities, and an exit code you can read off the token.
//!
//! The repository's screens already speak a fixed vocabulary -- `ERROR(instrument=...)` for "the
//! instrument could not take the reading", `FAIL(condition=...)` for "the reading was taken and
//! it is red", and a third state for "this subject is outside what this instrument can represent"
//! -- and the distinction between the first two is load-bearing everywhere it appears. A runner
//! that could not find ONNX Runtime and a runner that found a numerical disagreement must not
//! share an exit code, because only one of them is a statement about the EP.
//!
//! `UNSUPPORTED` is the third state and it is deliberately **not** a pass. Issue #5 names the
//! case: a Foundry Phi-3.5 checkpoint the runner can identify exactly but cannot execute (external
//! data, a tokenizer, custom ops). Reporting that as a green because "nothing went wrong" is the
//! vacuous pass this whole runner exists to make impossible.

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    /// The instrument could not take a reading. Says nothing about the EP. Exit 2.
    Instrument,
    /// A reading was taken and it is red. Exit 1.
    Condition,
    /// The subject is outside what this runner can represent, declared rather than faked. Exit 3.
    Unsupported,
}

impl Severity {
    pub fn exit_code(self) -> i32 {
        match self {
            Severity::Condition => 1,
            Severity::Instrument => 2,
            Severity::Unsupported => 3,
        }
    }

    pub fn keyword(self) -> &'static str {
        match self {
            Severity::Instrument => "instrument",
            Severity::Condition => "condition",
            Severity::Unsupported => "reason",
        }
    }

    pub fn token(self) -> &'static str {
        match self {
            Severity::Instrument => "ERROR",
            Severity::Condition => "FAIL",
            Severity::Unsupported => "UNSUPPORTED",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Failure {
    pub severity: Severity,
    pub cause: String,
    pub message: String,
}

impl Failure {
    pub fn instrument(cause: &str, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Instrument,
            cause: cause.to_string(),
            message: message.into(),
        }
    }

    pub fn condition(cause: &str, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Condition,
            cause: cause.to_string(),
            message: message.into(),
        }
    }

    pub fn unsupported(cause: &str, message: impl Into<String>) -> Self {
        Self {
            severity: Severity::Unsupported,
            cause: cause.to_string(),
            message: message.into(),
        }
    }

    /// The one-line token a screen greps for, e.g. `ERROR(instrument=ort_library_missing)`.
    pub fn token(&self) -> String {
        format!(
            "{}({}={})",
            self.severity.token(),
            self.severity.keyword(),
            self.cause
        )
    }

    pub fn exit_code(&self) -> i32 {
        self.severity.exit_code()
    }
}

impl fmt::Display for Failure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.token(), self.message)
    }
}

impl std::error::Error for Failure {}

pub type Result<T> = std::result::Result<T, Failure>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_three_severities_do_not_share_an_exit_code() {
        let codes = [
            Severity::Instrument.exit_code(),
            Severity::Condition.exit_code(),
            Severity::Unsupported.exit_code(),
        ];
        let mut sorted = codes.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "{codes:?}");
        assert!(!codes.contains(&0), "no failure may exit 0: {codes:?}");
    }

    #[test]
    fn tokens_are_greppable_and_name_their_cause() {
        assert_eq!(
            Failure::instrument("ort_library_missing", "x").token(),
            "ERROR(instrument=ort_library_missing)"
        );
        assert_eq!(
            Failure::condition("vacuous_no_dispatch", "x").token(),
            "FAIL(condition=vacuous_no_dispatch)"
        );
        assert_eq!(
            Failure::unsupported("external_data", "x").token(),
            "UNSUPPORTED(reason=external_data)"
        );
    }
}
