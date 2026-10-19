//! `up`, `down`, `swap`, `status`: run and talk to the supervisor.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, bail};
use loadmaster_hw::MemoryKind;
use loadmaster_lock::{Lockfile, Model, gb_to_bytes};
use loadmaster_plan::gib;
use loadmaster_serve::process::LaunchSpec;
use loadmaster_serve::{ModelSpec, Options, Routes, Supervisor};
use serde_json::Value;

use super::lock::Stores;
use crate::UpArgs;

fn runtime() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

/// The full command line for one entry, minus `--host`/`--port` (the supervisor adds them).
fn launch_argv(m: &Model, path: &Path, mmproj: Option<PathBuf>) -> Vec<String> {
    let p = path.display().to_string();
    let mut a = vec![m.runtime_bin.clone()];
    match m.runtime.as_str() {
        "vllm" => {
            a.extend([
                "serve".into(),
                p,
                "--max-model-len".into(),
                m.ctx.to_string(),
            ]);
            a.extend(["--served-model-name".into(), m.name.clone()]);
            a.extend(m.flags.iter().cloned());
        }
        "exl3" => {
            a.extend(m.flags.iter().cloned());
            let dir = path
                .parent()
                .map_or(".".into(), |d| d.display().to_string());
            let name = path
                .file_name()
                .map_or(p.clone(), |n| n.to_string_lossy().into_owned());
            a.extend(["--model-dir".into(), dir, "--model-name".into(), name]);
            a.extend(["--max-seq-len".into(), m.ctx.to_string()]);
        }
        _ => {
            a.extend(["-m".into(), p]);
            if let Some(mm) = mmproj {
                a.extend(["--mmproj".into(), mm.display().to_string()]);
            }
            a.extend(["-c".into(), m.ctx.to_string()]);
            a.extend(m.flags.iter().cloned());
        }
    }
    a
}

pub fn up(args: UpArgs) -> anyhow::Result<ExitCode> {
    if args.hw.pretend.is_some() {
        anyhow::bail!(
            "--pretend is for planning; `up` runs models on this machine's real hardware"
        );
    }
    let lock = Lockfile::load(&args.lock.lock)?;
    let snap = super::snapshot(&args.hw)?;
    let gpu = super::pick_gpu(&snap, "auto")?;
    if gpu.name != lock.host.gpu {
        eprintln!(
            "warning: models.lock was planned on {} but this machine has {}; footprints may not hold",
            lock.host.gpu, gpu.name
        );
    }
    let headroom = gb_to_bytes(lock.host.headroom_gb);
    // On unified memory the GPU side is RAM too: the budget is also capped by free RAM.
    let usable = match gpu.memory {
        MemoryKind::Unified if snap.host.available_bytes > 0 => {
            gpu.usable_bytes.min(snap.host.available_bytes + headroom)
        }
        _ => gpu.usable_bytes,
    };
    let budget = usable.saturating_sub(headroom);

    let logs = super::state_dir().join("logs");
    let mut stores = Stores::default();
    let mut specs = Vec::new();
    let mut targets = Vec::new();
    let mut problems = Vec::new();
    for m in &lock.models {
        let Some(footprint) = m.footprint_gb else {
            problems.push(format!(
                "{}: no footprint_gb; run `loadmaster plan --update {}`",
                m.name, m.name
            ));
            continue;
        };
        let info = match stores.resolve(&m.source) {
            Ok(i) => i,
            Err(e) => {
                problems.push(format!("{}: {e}", m.name));
                continue;
            }
        };
        let mmproj = match &m.mmproj {
            Some(src) if Path::new(src).exists() => Some(PathBuf::from(src)),
            Some(_) => info.mmproj.as_ref().map(|mm| mm.path.clone()),
            None => None,
        };
        targets.push(super::verify::Target {
            name: m.name.clone(),
            template_blake3: info.chat_template.as_ref().map(|t| t.blake3.clone()),
            quant: info.quant.to_string(),
        });
        let env = m
            .env
            .iter()
            .filter_map(|e| e.split_once('=').map(|(k, v)| (k.to_owned(), v.to_owned())))
            .collect();
        specs.push(ModelSpec {
            name: m.name.clone(),
            launch: LaunchSpec {
                argv: launch_argv(m, &info.path, mmproj),
                env,
                log: logs.join(format!("{}.log", m.name)),
                ready_path: "/health".into(),
            },
            bytes: gb_to_bytes(footprint).saturating_sub(headroom),
            keep_warm: m.keep_warm,
            idle_unload: m.idle_unload_s.map(Duration::from_secs),
        });
    }
    if !problems.is_empty() {
        bail!("cannot start:\n  {}", problems.join("\n  "));
    }

    let routes = Routes {
        by_name: lock
            .models
            .iter()
            .flat_map(|m| {
                std::iter::once(&m.name)
                    .chain(&m.aliases)
                    .map(move |n| (n.clone(), m.name.clone()))
            })
            .collect(),
        listed: lock.models.iter().map(|m| m.name.clone()).collect(),
    };
    let opts = Options {
        budget,
        load_timeout: Duration::from_secs(lock.proxy.load_timeout_s),
        idle_unload: Duration::from_secs(lock.proxy.idle_unload_s),
        max_restarts: 3,
        backoff: Duration::from_secs(1),
        stop_grace: Duration::from_secs(10),
    };
    let preload: Vec<String> = lock
        .models
        .iter()
        .filter(|m| m.keep_warm)
        .map(|m| m.name.clone())
        .collect();
    let listen = lock.proxy.listen.clone();
    let verify = args.verify;
    let lock_path = args.lock.lock.clone();

    runtime()?.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&listen)
            .await
            .with_context(|| format!("binding {listen} (is another `loadmaster up` running?)"))?;
        let sup = Supervisor::new(opts, specs);
        eprintln!(
            "loadmaster: serving {} models on http://{listen} ({} budget on {}); logs in {}",
            lock.models.len(),
            gib(budget),
            gpu.name,
            logs.display()
        );
        if verify {
            tokio::spawn(super::verify::run(
                std::sync::Arc::clone(&sup),
                targets,
                listen.clone(),
                lock_path,
            ));
        }
        loadmaster_serve::serve(listener, sup, routes, preload).await?;
        anyhow::Ok(())
    })?;
    Ok(ExitCode::SUCCESS)
}

fn base_url(lock_path: &Path) -> anyhow::Result<String> {
    let lock = Lockfile::load(lock_path)?;
    Ok(format!("http://{}", lock.proxy.listen))
}

async fn admin(lock_path: &Path, method: reqwest::Method, path: &str) -> anyhow::Result<Value> {
    let url = format!("{}{path}", base_url(lock_path)?);
    let resp = reqwest::Client::new()
        .request(method, &url)
        .send()
        .await
        .with_context(|| {
            format!("no supervisor answering at {url}; is `loadmaster up` running?")
        })?;
    let status = resp.status();
    let body: Value = resp.json().await.unwrap_or(Value::Null);
    if !status.is_success() {
        bail!(
            "{}",
            body["error"]["message"].as_str().unwrap_or(status.as_str())
        );
    }
    Ok(body)
}

pub fn down(lock_path: &Path) -> anyhow::Result<ExitCode> {
    runtime()?.block_on(admin(
        lock_path,
        reqwest::Method::POST,
        "/loadmaster/shutdown",
    ))?;
    println!("stopping");
    Ok(ExitCode::SUCCESS)
}

pub fn swap(lock_path: &Path, name: &str) -> anyhow::Result<ExitCode> {
    let v = runtime()?.block_on(admin(
        lock_path,
        reqwest::Method::POST,
        &format!("/loadmaster/swap/{name}"),
    ))?;
    println!("{} is ready", v["ready"].as_str().unwrap_or(name));
    Ok(ExitCode::SUCCESS)
}

pub fn status(lock_path: &Path, json: bool) -> anyhow::Result<ExitCode> {
    let lock = Lockfile::load(lock_path)?;
    let v = runtime()?.block_on(admin(lock_path, reqwest::Method::GET, "/loadmaster/status"))?;
    if json {
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(ExitCode::SUCCESS);
    }
    let verdicts: HashMap<&str, &loadmaster_lock::Verify> = lock
        .models
        .iter()
        .filter_map(|m| m.verify.as_ref().map(|v| (m.name.as_str(), v)))
        .collect();
    // A verdict holds for the runtime version it was measured on; for running models,
    // compare with what the backend reports now.
    let rt = runtime()?;
    let verdict_cell = |name: &str, port: Option<u64>| -> String {
        let Some(v) = verdicts.get(name) else {
            return "-".into();
        };
        let now = port.and_then(|p| {
            let c = super::probe::client(&format!("http://127.0.0.1:{p}"), None);
            let d = rt.block_on(loadmaster_probe::detect::detect(&c));
            (d.runtime != "unknown").then(|| super::verify::runtime_label(&d))
        });
        match (now, &v.runtime_version) {
            (Some(now), Some(then)) if now != *then => {
                format!(
                    "{} (stale: verified on {then}, running {now}; re-run `up --verify`)",
                    v.verdict
                )
            }
            _ => v.verdict.clone(),
        }
    };
    println!(
        "{:<28} {:<9} {:>7} {:>6} {:>10} {:>8}  {:<8}",
        "NAME", "STATE", "PID", "PORT", "RESIDENT", "RESTARTS", "VERIFY"
    );
    for s in v.as_array().into_iter().flatten() {
        let name = s["name"].as_str().unwrap_or("?");
        println!(
            "{:<28} {:<9} {:>7} {:>6} {:>10} {:>8}  {:<8}",
            name,
            s["state"].as_str().unwrap_or("?"),
            s["pid"].as_u64().map_or("-".into(), |p| p.to_string()),
            s["port"].as_u64().map_or("-".into(), |p| p.to_string()),
            gib(s["resident_bytes"].as_u64().unwrap_or(0)),
            s["restarts"].as_u64().unwrap_or(0),
            verdict_cell(name, s["port"].as_u64()),
        );
        if s["state"] == "failed" {
            println!(
                "  {}; log: {}",
                s["error"].as_str().unwrap_or("failed"),
                s["log"].as_str().unwrap_or("?")
            );
        }
    }
    Ok(ExitCode::SUCCESS)
}
