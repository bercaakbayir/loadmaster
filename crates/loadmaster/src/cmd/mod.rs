pub mod gpu;
pub mod plan;

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
