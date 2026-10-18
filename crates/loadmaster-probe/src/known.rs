//! Known upstream issues: when a failure matches a curated signature, the probe prints the
//! issue URL. The curated list is `known_issues.toml` next to the fixtures.

use serde::{Deserialize, Serialize};

use crate::detect::Detected;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KnownIssue {
    /// Failure class, e.g. `call_in_content`.
    pub failure: String,
    /// Runtime it applies to; any when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime: Option<String>,
    /// Case-insensitive substring of the model name; any when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    pub url: String,
    pub note: String,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct File {
    #[serde(default)]
    issue: Vec<KnownIssue>,
}

pub fn parse(text: &str) -> Result<Vec<KnownIssue>, String> {
    toml::from_str::<File>(text)
        .map(|f| f.issue)
        .map_err(|e| e.to_string())
}

/// The curated list shipped with loadmaster.
pub fn builtin() -> Vec<KnownIssue> {
    parse(include_str!("../fixtures/known_issues.toml")).expect("built-in known_issues.toml parses")
}

pub fn matches(k: &KnownIssue, failure: &str, d: &Detected, model: &str) -> bool {
    k.failure == failure
        && k.runtime.as_ref().is_none_or(|r| *r == d.runtime)
        && k.model
            .as_ref()
            .is_none_or(|m| model.to_ascii_lowercase().contains(&m.to_ascii_lowercase()))
}
