pub mod fake;
pub mod gpu;
pub mod import;
pub mod lock;
pub mod plan;
pub mod probe;
pub mod scan;
pub mod serve;
pub mod verify;

use std::path::Path;

use anyhow::Context;
use loadmaster_hw::{HwProvider, HwSnapshot, SystemHw};

/// The hardware snapshot: `--pretend`, replayed from `--hw-file`, or detected.
pub fn snapshot(hw: &crate::HwArgs) -> anyhow::Result<HwSnapshot> {
    if let Some(spec) = &hw.pretend {
        return Ok(loadmaster_hw::pretend(spec)?);
    }
    match hw.hw_file.as_deref() {
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

/// `$LOADMASTER_HOME`, or `~/.loadmaster`: logs, probe captures, eval database.
pub fn state_dir() -> std::path::PathBuf {
    std::env::var_os("LOADMASTER_HOME")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::var_os("HOME")
                .map(std::path::PathBuf::from)
                .unwrap_or_default()
                .join(".loadmaster")
        })
}

/// The runtime to plan a model with: the requested one, or the natural one for its format.
pub fn default_runtime(m: &loadmaster_model::ModelInfo) -> loadmaster_plan::Runtime {
    use loadmaster_model::{Format, Quant};
    match (&m.format, &m.quant) {
        (Format::Gguf, _) => loadmaster_plan::Runtime::LlamaCpp,
        (_, Quant::Exl3 { .. }) => loadmaster_plan::Runtime::Exl3,
        _ => loadmaster_plan::Runtime::Vllm,
    }
}

/// The GPU to plan for: `auto` (most usable memory), an index, or part of a name.
pub fn pick_gpu(
    snap: &loadmaster_hw::HwSnapshot,
    query: &str,
) -> anyhow::Result<loadmaster_hw::Gpu> {
    use anyhow::Context;
    let gpu =
        match query {
            "auto" => snap.best_gpu(),
            q => Some(snap.find_gpu(q).with_context(|| {
                format!("no GPU matches `{q}`; run `loadmaster gpu` to list them")
            })?),
        };
    Ok(gpu.ok_or(loadmaster_plan::Error::NoGpu)?.clone())
}
