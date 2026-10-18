pub mod gpu;
pub mod plan;
pub mod scan;

use std::path::Path;

use anyhow::Context;
use loadmaster_hw::{HwProvider, HwSnapshot, SystemHw};

/// The hardware snapshot: replayed from `--hw-file`, or detected.
pub fn snapshot(hw_file: Option<&Path>) -> anyhow::Result<HwSnapshot> {
    match hw_file {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("reading {}", path.display()))?;
            serde_json::from_str(&text).with_context(|| {
                format!(
                    "{}: not a hardware snapshot; write one with `loadmaster gpu --json`",
                    path.display()
                )
            })
        }
        None => Ok(SystemHw.snapshot()?),
    }
}

/// Store roots from the environment, plus `--path` directories.
pub fn roots(extra: Vec<std::path::PathBuf>) -> loadmaster_store::Roots {
    let home = std::env::var_os("HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_default();
    let mut roots = loadmaster_store::Roots::defaults(&home, |k| std::env::var(k).ok());
    roots.extra = extra;
    roots
}

/// A model by path, or by name from the scanned stores.
pub fn open_model(arg: &str) -> anyhow::Result<loadmaster_model::ModelInfo> {
    let path = Path::new(arg);
    if path.exists() {
        return Ok(loadmaster_model::ModelInfo::open(path)?);
    }
    let scan = loadmaster_store::scan(&roots(Vec::new()));
    let entry = loadmaster_store::resolve(&scan.entries, arg).map_err(|e| anyhow::anyhow!(e))?;
    match &entry.info {
        Ok(info) => Ok((**info).clone()),
        Err(e) => anyhow::bail!("{}: {e}", entry.path.display()),
    }
}
