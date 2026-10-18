//! `init`, `add`, `plan --update`: write models.lock from plans.

use std::io::Read;
use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, bail};
use loadmaster_lock::{Host, Lockfile, Model, diff, gb, source_string};
use loadmaster_model::{ModelInfo, Source};
use loadmaster_plan::{Error as PlanError, PlanRequest, explain};
use loadmaster_store::Scan;

use crate::{AddArgs, InitArgs, PlanArgs};

/// Lazily scanned stores, for resolving lockfile sources.
#[derive(Default)]
pub struct Stores(Option<Scan>);

impl Stores {
    fn scan(&mut self) -> &Scan {
        self.0
            .get_or_insert_with(|| loadmaster_store::scan(&super::roots(Vec::new())))
    }

    /// The model a lockfile `source` names.
    pub fn resolve(&mut self, source: &str) -> anyhow::Result<ModelInfo> {
        let by_name = |stores: &mut Stores, name: &str| -> anyhow::Result<ModelInfo> {
            let scan = stores.scan();
            let e =
                loadmaster_store::resolve(&scan.entries, name).map_err(|e| anyhow::anyhow!(e))?;
            match &e.info {
                Ok(m) => Ok((**m).clone()),
                Err(err) => bail!("{}: {err}", e.path.display()),
            }
        };
        if let Some(rest) = source.strip_prefix("hf://") {
            return by_name(self, rest);
        }
        if let Some(rest) = source.strip_prefix("ollama://") {
            return by_name(self, rest);
        }
        let path = Path::new(source);
        if path.exists() {
            return Ok(ModelInfo::open(path)?);
        }
        // `org/repo` as vLLM writes it.
        if !source.starts_with('/') && source.matches('/').count() == 1 {
            if let Ok(m) = by_name(self, source) {
                return Ok(m);
            }
        }
        bail!("model source `{source}` not found on disk or in the scanned stores")
    }
}

/// The model's sha256 when the store already names the file by it, else hashed (unless
/// `no_hash`, or the model is a directory).
fn sha256(m: &ModelInfo, no_hash: bool) -> anyhow::Result<Option<String>> {
    if let Source::Ollama { digest, .. } = &m.source {
        return Ok(digest.strip_prefix("sha256:").map(str::to_owned));
    }
    if m.path.is_dir() || m.files.len() != 1 {
        return Ok(None);
    }
    // HF cache blobs of LFS files are named by the content's sha256.
    if let Ok(real) = std::fs::canonicalize(&m.path)
        && let Some(name) = real.file_name().and_then(|n| n.to_str())
        && name.len() == 64
        && name.bytes().all(|b| b.is_ascii_hexdigit())
    {
        return Ok(Some(name.to_owned()));
    }
    if no_hash {
        return Ok(None);
    }
    use sha2::{Digest, Sha256};
    let mut f =
        std::fs::File::open(&m.path).with_context(|| format!("hashing {}", m.path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(Some(
        h.finalize().iter().map(|b| format!("{b:02x}")).collect(),
    ))
}

/// A served name for a model: the Ollama name, the HF repo, or the file stem.
fn default_name(m: &ModelInfo) -> String {
    let raw = match &m.source {
        Source::Ollama { name, .. } => name.clone(),
        Source::HfCache { repo, .. } if m.path.is_dir() => {
            repo.rsplit('/').next().unwrap_or(repo).to_owned()
        }
        _ => m
            .path
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_else(|| "model".into()),
    };
    raw.to_ascii_lowercase().replace(char::is_whitespace, "-")
}

fn unique(lock: &Lockfile, base: String) -> String {
    let taken = |n: &str| {
        lock.models
            .iter()
            .any(|m| m.name == n || m.aliases.iter().any(|a| a == n))
    };
    if !taken(&base) {
        return base;
    }
    (2..)
        .map(|i| format!("{base}-{i}"))
        .find(|n| !taken(n))
        .expect("some suffix is free")
}

fn host(gpu: &loadmaster_hw::Gpu) -> Host {
    Host {
        gpu: gpu.name.clone(),
        vram_gb: gb(gpu.total_bytes),
        headroom_gb: gb(loadmaster_plan::headroom_for(gpu, None).bytes),
    }
}

/// Plan `m` and build its lockfile entry; `None` (with a message) if it cannot fit.
fn entry(
    m: &ModelInfo,
    name: String,
    snap: &loadmaster_hw::HwSnapshot,
    gpu: &loadmaster_hw::Gpu,
    req: &PlanRequest,
    no_hash: bool,
) -> anyhow::Result<Option<Model>> {
    match loadmaster_plan::plan(m, snap, gpu, req) {
        Ok(p) => {
            let source = source_string(&m.source, &m.path);
            Ok(Some(Model::from_plan(
                &name,
                source,
                sha256(m, no_hash)?,
                &p,
            )))
        }
        Err(PlanError::CannotFit(n)) => {
            eprint!("skipping {name}: {}", explain::no_fit(&n, false));
            Ok(None)
        }
        Err(e) => {
            eprintln!("skipping {name}: {e}");
            Ok(None)
        }
    }
}

pub fn init(args: InitArgs) -> anyhow::Result<ExitCode> {
    if args.lock.lock.exists() && !args.force {
        bail!(
            "{} already exists; pass --force to overwrite it, or use `loadmaster add`",
            args.lock.lock.display()
        );
    }
    if args.names.is_empty() && !args.all {
        bail!("name the models to include (see `loadmaster scan`), or pass --all");
    }
    let snap = super::snapshot(args.hw.hw_file.as_deref())?;
    let gpu = super::pick_gpu(&snap, "auto")?;
    let scan = loadmaster_store::scan(&super::roots(Vec::new()));
    let chosen: Vec<&ModelInfo> = if args.all {
        scan.entries
            .iter()
            .filter_map(|e| e.info.as_deref().ok())
            .collect()
    } else {
        args.names
            .iter()
            .map(|n| {
                let e =
                    loadmaster_store::resolve(&scan.entries, n).map_err(|e| anyhow::anyhow!(e))?;
                e.info
                    .as_deref()
                    .map_err(|err| anyhow::anyhow!("{n}: {err}"))
            })
            .collect::<anyhow::Result<_>>()?
    };
    let mut lock = Lockfile::new(host(&gpu));
    lock.proxy.listen = args.listen;
    for m in chosen {
        let req = PlanRequest {
            ctx: args.ctx,
            runtime: super::default_runtime(m),
            ..PlanRequest::default()
        };
        let name = unique(&lock, default_name(m));
        if let Some(e) = entry(m, name, &snap, &gpu, &req, args.no_hash)? {
            println!(
                "{:<32} ctx {:>6}  {:>6.2} GiB  {}",
                e.name,
                e.ctx,
                e.footprint_gb.unwrap_or(0.0),
                e.runtime
            );
            lock.models.push(e);
        }
    }
    if lock.models.is_empty() {
        bail!("no model fits; nothing written");
    }
    lock.save(&args.lock.lock, args.force)?;
    println!(
        "wrote {} ({} models); start them with `loadmaster up`",
        args.lock.lock.display(),
        lock.models.len()
    );
    Ok(ExitCode::SUCCESS)
}

pub fn add(args: AddArgs) -> anyhow::Result<ExitCode> {
    let mut lock = Lockfile::load(&args.lock.lock)
        .with_context(|| "run `loadmaster init` first to create the lockfile".to_string())?;
    let m = super::open_model(&args.model)?;
    let snap = super::snapshot(args.hw.hw_file.as_deref())?;
    let gpu = super::pick_gpu(&snap, "auto")?;
    let name = match args.name {
        Some(n) => {
            if lock.route(&n).is_some() {
                bail!(
                    "`{n}` is already a model name or alias in {}",
                    args.lock.lock.display()
                );
            }
            n
        }
        None => unique(&lock, default_name(&m)),
    };
    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args
            .runtime
            .map_or_else(|| super::default_runtime(&m), Into::into),
        ..PlanRequest::default()
    };
    let Some(mut e) = entry(&m, name, &snap, &gpu, &req, args.no_hash)? else {
        return Ok(ExitCode::from(1));
    };
    e.aliases = args.aliases;
    e.keep_warm = args.keep_warm;
    lock.models.push(e.clone());
    let problems = lock.problems();
    if !problems.is_empty() {
        bail!("{}", problems.join("; "));
    }
    lock.save(&args.lock.lock, true)?;
    println!(
        "added {} (ctx {}, {:.2} GiB) to {}",
        e.name,
        e.ctx,
        e.footprint_gb.unwrap_or(0.0),
        args.lock.lock.display()
    );
    Ok(ExitCode::SUCCESS)
}

/// `plan --update NAME`: re-plan an entry, keeping what the user chose (name, source,
/// aliases, keep_warm, env, runtime_bin, idle timeout).
pub fn update(args: PlanArgs) -> anyhow::Result<ExitCode> {
    let old = Lockfile::load(&args.lock)?;
    let entry = old
        .model(&args.model)
        .with_context(|| format!("no model `{}` in {}", args.model, args.lock.display()))?
        .clone();
    let m = Stores::default().resolve(&entry.source)?;
    let snap = super::snapshot(args.hw.hw_file.as_deref())?;
    let gpu = super::pick_gpu(&snap, &args.gpu)?;
    let runtime =
        loadmaster_lock::runtime(&entry.runtime).context("unknown runtime in lockfile")?;
    let req = PlanRequest {
        ctx: Some(args.ctx.unwrap_or(entry.ctx)),
        runtime,
        ..PlanRequest::default()
    };
    let p = match loadmaster_plan::plan(&m, &snap, &gpu, &req) {
        Ok(p) => p,
        Err(PlanError::CannotFit(n)) => {
            print!("{}", explain::no_fit(&n, args.explain));
            return Ok(ExitCode::from(1));
        }
        Err(e) => return Err(e.into()),
    };
    let fresh = Model::from_plan(&entry.name, entry.source.clone(), entry.sha256.clone(), &p);
    let mut new = old.clone();
    let slot = new
        .models
        .iter_mut()
        .find(|x| x.name == entry.name)
        .expect("found above");
    slot.ctx = fresh.ctx;
    slot.footprint_gb = fresh.footprint_gb;
    slot.flags = fresh.flags;
    slot.mmproj = fresh.mmproj.or(slot.mmproj.take());
    let changes = diff(&old, &new);
    if changes.is_empty() {
        println!("{}: up to date", entry.name);
        return Ok(ExitCode::SUCCESS);
    }
    for c in &changes {
        println!("{c}");
    }
    new.save(&args.lock, true)?;
    println!("updated {} in {}", entry.name, args.lock.display());
    Ok(ExitCode::SUCCESS)
}
