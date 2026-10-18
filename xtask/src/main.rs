//! Repository tasks: `cargo xtask <task>`.

use std::collections::{BTreeMap, BTreeSet};
use std::process::Command;

use anyhow::{Context, bail};

mod fixtures;
mod hf_fixtures;

/// Workspace crates each crate may depend on (normal and dev). Anything not listed is a
/// boundary violation; in particular serve and probe must never see each other.
const ALLOWED: &[(&str, &[&str])] = &[
    ("loadmaster-model", &[]),
    ("loadmaster-hw", &["loadmaster-model"]),
    ("loadmaster-store", &["loadmaster-model"]),
    ("loadmaster-plan", &["loadmaster-model", "loadmaster-hw"]),
    (
        "loadmaster-lock",
        &["loadmaster-model", "loadmaster-hw", "loadmaster-plan"],
    ),
    (
        "loadmaster-serve",
        &[
            "loadmaster-model",
            "loadmaster-hw",
            "loadmaster-plan",
            "loadmaster-lock",
        ],
    ),
    (
        "loadmaster-probe",
        &[
            "loadmaster-model",
            "loadmaster-hw",
            "loadmaster-plan",
            "loadmaster-lock",
        ],
    ),
    (
        "loadmaster",
        &[
            "loadmaster-model",
            "loadmaster-hw",
            "loadmaster-plan",
            "loadmaster-store",
            "loadmaster-lock",
            "loadmaster-serve",
            "loadmaster-probe",
        ],
    ),
];

fn main() -> anyhow::Result<()> {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("check-deps") => check_deps(),
        Some("gen-fixtures") => {
            let check = std::env::args().nth(2).as_deref() == Some("--check");
            fixtures::generate(&workspace_root().join("fixtures"), check)
        }
        _ => bail!("usage: cargo xtask <check-deps | gen-fixtures [--check]>"),
    }
}

fn check_deps() -> anyhow::Result<()> {
    let out = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()))
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .output()
        .context("running cargo metadata")?;
    if !out.status.success() {
        bail!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let meta: serde_json::Value = serde_json::from_slice(&out.stdout)?;
    let allowed: BTreeMap<&str, BTreeSet<&str>> = ALLOWED
        .iter()
        .map(|(k, v)| (*k, v.iter().copied().collect()))
        .collect();

    let mut violations = Vec::new();
    for pkg in meta["packages"].as_array().into_iter().flatten() {
        let name = pkg["name"].as_str().unwrap_or_default();
        if name == "xtask" {
            continue;
        }
        let Some(ok) = allowed.get(name) else {
            violations.push(format!("{name}: not listed in xtask ALLOWED table"));
            continue;
        };
        for dep in pkg["dependencies"].as_array().into_iter().flatten() {
            let dep_name = dep["name"].as_str().unwrap_or_default();
            if dep_name.starts_with("loadmaster") && !ok.contains(dep_name) {
                violations.push(format!("{name} -> {dep_name} is not allowed"));
            }
        }
    }
    if violations.is_empty() {
        println!("crate boundaries ok");
        Ok(())
    } else {
        bail!("crate boundary violations:\n  {}", violations.join("\n  "))
    }
}

fn workspace_root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives in the workspace root")
        .to_path_buf()
}
