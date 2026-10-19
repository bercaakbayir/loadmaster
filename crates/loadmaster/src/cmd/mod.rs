pub mod fake;
pub mod gpu;
pub mod import;
pub mod lock;
pub mod plan;
pub mod probe;
pub mod report;
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

/// What `plan` was pointed at.
pub enum Target {
    Model(Box<loadmaster_model::ModelInfo>),
    /// A Hub repo with no file named: the caller compares its GGUF files.
    Repo {
        spec: loadmaster_store::remote::Spec,
        files: Vec<loadmaster_store::remote::RemoteFile>,
    },
}

/// A model on disk or in a local store, else, for `hf://` specs only, from the Hugging Face
/// Hub (headers only; `offline` uses the header cache alone). A repo with no file named is
/// a GGUF repo to compare when planning for llama.cpp (`gguf`), else a safetensors model.
pub fn open_target(arg: &str, offline: bool, gguf: bool) -> anyhow::Result<Target> {
    use loadmaster_store::remote::Spec;
    let local = open_model(arg);
    let spec = match (local, Spec::parse(arg)) {
        (Ok(m), _) => return Ok(Target::Model(Box::new(m))),
        (Err(e), None) => return Err(e),
        (Err(_), Some(spec)) => spec,
    };
    let hub = hub(offline);
    let files = hub.list(&spec.repo, &spec.rev)?;
    match &spec.file {
        Some(file) => Ok(Target::Model(Box::new(
            hub.open_gguf(&spec.repo, &spec.rev, file, &files)?,
        ))),
        None => {
            let has_gguf = files.iter().any(|f| f.path.ends_with(".gguf"));
            let has_st = files.iter().any(|f| f.path == "config.json")
                && files.iter().any(|f| f.path.ends_with(".safetensors"));
            match (has_gguf, has_st) {
                (true, _) if gguf || !has_st => Ok(Target::Repo { spec, files }),
                (_, true) => Ok(Target::Model(Box::new(
                    hub.open_safetensors(&spec.repo, &spec.rev, &files)?,
                ))),
                _ => anyhow::bail!(
                    "hf://{}: no .gguf files and no config.json with safetensors",
                    spec.repo
                ),
            }
        }
    }
}

/// The Hub client: `$HF_ENDPOINT`, a token from `$HF_TOKEN` or the `huggingface-cli login`
/// file, and the header cache under the state directory.
pub fn hub(offline: bool) -> loadmaster_store::remote::Hub {
    use loadmaster_store::remote::{DEFAULT_ENDPOINT, Hub};
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let endpoint = env("HF_ENDPOINT").unwrap_or_else(|| DEFAULT_ENDPOINT.into());
    let token = env("HF_TOKEN").or_else(|| {
        let hf_home = env("HF_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::path::PathBuf::from(env("HOME").unwrap_or_default()).join(".cache/huggingface")
            });
        std::fs::read_to_string(hf_home.join("token"))
            .ok()
            .map(|t| t.trim().to_owned())
            .filter(|t| !t.is_empty())
    });
    Hub::new(&endpoint, token, state_dir().join("hf"), offline)
}

/// `$LOADMASTER_HOME`, or `~/.loadmaster`: logs, probe captures, eval database, Hub headers.
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

/// The GPUs to plan for: `auto` or `all` (every NVIDIA GPU, the way llama.cpp splits by
/// default; otherwise the one with most usable memory), a list of indices `0,1`, or one
/// GPU by index or part of its name. The first is the primary; the rest are split onto.
pub fn pick_gpus(
    snap: &loadmaster_hw::HwSnapshot,
    query: &str,
) -> anyhow::Result<(loadmaster_hw::Gpu, Vec<loadmaster_hw::Gpu>)> {
    use loadmaster_hw::{MemoryKind, Vendor};
    let splittable =
        |g: &&loadmaster_hw::Gpu| g.vendor == Vendor::Nvidia && g.memory == MemoryKind::Dedicated;
    let mut chosen: Vec<loadmaster_hw::Gpu> = match query {
        "auto" | "all" => {
            let many: Vec<_> = snap.gpus.iter().filter(splittable).cloned().collect();
            if many.len() > 1 {
                many
            } else {
                vec![pick_gpu(snap, "auto")?]
            }
        }
        q if q.contains(',') => q
            .split(',')
            .map(|i| pick_gpu(snap, i.trim()))
            .collect::<anyhow::Result<_>>()?,
        q => vec![pick_gpu(snap, q)?],
    };
    if chosen.len() > 1 && !chosen.iter().all(|g| splittable(&g)) {
        anyhow::bail!("splitting across GPUs is supported for NVIDIA cards only");
    }
    chosen.sort_by_key(|g| g.index);
    chosen.dedup_by_key(|g| g.index);
    let first = chosen.remove(0);
    Ok((first, chosen))
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
