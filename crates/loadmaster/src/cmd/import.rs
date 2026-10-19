//! `import llama-swap`: convert a llama-swap config into models.lock.

use std::path::Path;
use std::process::ExitCode;

use anyhow::{Context, bail};
use loadmaster_lock::import::llama_swap;
use loadmaster_lock::{Host, Lockfile, Model, gb};
use loadmaster_plan::{PlanRequest, Runtime};

use super::lock::Stores;

pub fn llama_swap(
    config: &Path,
    lock_path: &Path,
    force: bool,
    hw: &crate::HwArgs,
) -> anyhow::Result<ExitCode> {
    if lock_path.exists() && !force {
        bail!(
            "{} already exists; pass --force to overwrite it",
            lock_path.display()
        );
    }
    let text =
        std::fs::read_to_string(config).with_context(|| format!("reading {}", config.display()))?;
    let imported = llama_swap::parse(&text, |k| std::env::var(k).ok())
        .map_err(|e| anyhow::anyhow!("{}: {e}", config.display()))?;
    let snap = super::snapshot(hw)?;
    let gpu = snap.best_gpu().cloned();
    let mut lock = Lockfile::new(match &gpu {
        Some(g) => Host {
            gpu: g.name.clone(),
            vram_gb: gb(g.total_bytes),
            headroom_gb: gb(loadmaster_plan::headroom_for(g, None).bytes),
        },
        None => Host {
            gpu: "none".into(),
            vram_gb: 0.0,
            headroom_gb: 0.0,
        },
    });
    if let Some(ttl) = imported.global_ttl_s {
        lock.proxy.idle_unload_s = ttl;
    }

    let mut stores = Stores::default();
    let mut unplanned = Vec::new();
    for im in &imported.models {
        let mut m: Model = im.model.clone();
        // vLLM names models by repo id; point at the HF cache when we can.
        if m.runtime == "vllm"
            && !Path::new(&m.source).exists()
            && m.source.matches('/').count() == 1
        {
            m.source = format!("hf://{}", m.source);
        }
        let planned = match (&gpu, stores.resolve(&m.source)) {
            (Some(g), Ok(info)) => {
                let req = PlanRequest {
                    ctx: im.ctx_given.then_some(m.ctx),
                    runtime: loadmaster_lock::runtime(&m.runtime).unwrap_or(Runtime::LlamaCpp),
                    ..PlanRequest::default()
                };
                match loadmaster_plan::plan(&info, &snap, g, &req) {
                    Ok(p) => {
                        m.ctx = p.ctx;
                        m.footprint_gb = Some(gb(p.footprint.gpu_bytes));
                        true
                    }
                    Err(e) => {
                        unplanned.push(format!("{}: {e}", m.name));
                        false
                    }
                }
            }
            (None, _) => {
                unplanned.push(format!("{}: no GPU detected to plan on", m.name));
                false
            }
            (_, Err(e)) => {
                unplanned.push(format!("{}: {e}", m.name));
                false
            }
        };
        println!(
            "imported {:<28} {:<9} ctx {:>6}  {}",
            m.name,
            m.runtime,
            m.ctx,
            if planned {
                format!("{:.2} GiB", m.footprint_gb.unwrap_or(0.0))
            } else {
                "footprint unknown".into()
            }
        );
        lock.models.push(m);
    }
    for s in &imported.skipped {
        println!("skipped  {:<28} {}", s.id, s.reason);
    }
    for w in &imported.warnings {
        println!("note: {w}");
    }
    for u in &unplanned {
        println!("not planned: {u}; set the path and run `loadmaster plan --update`");
    }
    let problems = lock.problems();
    if !problems.is_empty() {
        bail!(
            "the imported config is not a valid lockfile:\n  {}",
            problems.join("\n  ")
        );
    }
    lock.save(lock_path, force)?;
    println!(
        "wrote {} ({} models)",
        lock_path.display(),
        lock.models.len()
    );
    Ok(ExitCode::SUCCESS)
}
