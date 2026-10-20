//! `loadmaster report`: run a model once (or read a `loadmaster up` log), record what it
//! really allocated and how fast it generated, next to what loadmaster predicted, and write
//! a validation case to review and submit. Nothing is uploaded; the user opens a prefilled
//! GitHub issue in their browser if they want to share it.

pub mod llama_log;

use std::io::{BufRead, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitCode, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use loadmaster_hw::{HwProvider, MemoryKind, SystemHw};
use loadmaster_model::{ModelInfo, Source};
use loadmaster_plan::{Error as PlanError, Plan, PlanRequest, UNIFIED_RAM_MARGIN_PCT};
use serde::Serialize;

use llama_log::{Kind, LoadLog, Outcome};

/// Prompt and generation lengths of the one timed request.
const PROMPT_WORDS: usize = 512;
const GEN_TOKENS: u32 = 128;
/// URLs longer than this are not reliably accepted by browsers and GitHub.
const MAX_URL: usize = 7500;

pub struct Args {
    pub model: Option<String>,
    pub from_log: Option<String>,
    pub ctx: Option<u32>,
    pub runtime_bin: String,
    pub yes: bool,
    pub unsafe_unified: bool,
    pub try_anyway: bool,
    pub timeout_s: u64,
    pub out: PathBuf,
    pub show: bool,
    pub lock: PathBuf,
    pub hw: crate::HwArgs,
}

#[derive(Serialize)]
struct Report {
    report: Meta,
    model: ModelPart,
    hardware: HardwarePart,
    prediction: Prediction,
    observed: Observed,
}

#[derive(Serialize)]
struct Meta {
    loadmaster: &'static str,
    date: String,
    os: String,
    runtime: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    runtime_build: Option<String>,
    /// The command that ran, with paths replaced by `<model>` and `<mmproj>`.
    command: String,
    /// `run` (started by `loadmaster report`) or `log` (read from a `loadmaster up` log).
    mode: &'static str,
}

#[derive(Serialize)]
struct ModelPart {
    name: String,
    arch: String,
    quant: String,
    n_layer: u32,
    weight_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    sha256: Option<String>,
    /// `hf://org/repo/file.gguf` or `ollama://name:tag` when the model came from a store.
    #[serde(skip_serializing_if = "Option::is_none")]
    source: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    mmproj_mib: Option<f64>,
}

#[derive(Serialize)]
struct HardwarePart {
    gpu: String,
    vendor: &'static str,
    memory: &'static str,
    total_mib: u64,
    usable_mib: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    bandwidth_gbs: Option<f64>,
    host_total_mib: u64,
    host_available_mib: u64,
}

#[derive(Serialize)]
struct Prediction {
    /// `fits` or `cannot_fit`.
    result: &'static str,
    ctx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    kv: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_gpu_layers: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    n_cpu_moe: Option<u32>,
    /// GPU footprint including headroom.
    gpu_mib: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_weights_mib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_kv_mib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    gpu_compute_mib: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tok_s: Option<[f64; 2]>,
}

#[derive(Serialize)]
struct Observed {
    /// `fits`, `oom`, `failed`, `timeout`, `aborted` or `unknown`.
    result: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    load_seconds: Option<f64>,
    buffers: Vec<DeviceBuffers>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vision: Option<VisionBuffers>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<SpeedPart>,
}

#[derive(Serialize)]
struct DeviceBuffers {
    device: String,
    model: f64,
    kv: f64,
    compute: f64,
    output: f64,
}

#[derive(Serialize)]
struct VisionBuffers {
    #[serde(skip_serializing_if = "Option::is_none")]
    projector: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_mib: Option<f64>,
    compute_mib: f64,
}

#[derive(Serialize)]
struct SpeedPart {
    tg_tok_s: f64,
    pp_tok_s: f64,
    prompt_tokens: u64,
    gen_tokens: u64,
}

fn mib(bytes: u64) -> f64 {
    (bytes as f64 / (1u64 << 20) as f64 * 100.0).round() / 100.0
}

/// Replace absolute paths (and the model's own paths) so a report never names a user,
/// a host or a directory layout.
fn redact(s: &str, model: &ModelInfo) -> String {
    let mut s = s.to_owned();
    for p in &model.files {
        s = s.replace(&p.display().to_string(), "<model>");
    }
    s = s.replace(&model.path.display().to_string(), "<model>");
    if let Some(m) = &model.mmproj {
        s = s.replace(&m.path.display().to_string(), "<mmproj>");
    }
    s.split(' ')
        .map(|w| {
            let bare = w.trim_matches(|c| c == '\'' || c == '"' || c == ',');
            if (bare.starts_with('/') && bare.len() > 1)
                || bare.starts_with("~/")
                || bare.get(1..3) == Some(":\\")
            {
                w.replace(bare, "<path>")
            } else {
                w.to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn source_string(m: &ModelInfo) -> Option<String> {
    match &m.source {
        Source::Loose => None,
        s => Some(loadmaster_lock::source_string(s, &m.path)),
    }
}

fn free_port() -> anyhow::Result<u16> {
    Ok(std::net::TcpListener::bind("127.0.0.1:0")?
        .local_addr()?
        .port())
}

fn confirm(question: &str) -> anyhow::Result<bool> {
    if !std::io::stdin().is_terminal() {
        bail!("{question} needs an answer; rerun with --yes to start without asking");
    }
    eprint!("{question} [y/N] ");
    std::io::stderr().flush()?;
    let mut line = String::new();
    std::io::stdin().lock().read_line(&mut line)?;
    Ok(matches!(line.trim(), "y" | "Y" | "yes"))
}

/// Why a run on unified memory is refused, or `None` when it may go ahead.
fn unified_refusal(plan: &Plan) -> Option<String> {
    if plan.gpu.memory != MemoryKind::Unified {
        return None;
    }
    plan.warnings
        .iter()
        .any(|w| w.is(loadmaster_plan::WarningKind::TightRam))
        .then(|| {
            format!(
                "this plan leaves under {UNIFIED_RAM_MARGIN_PCT}% of free RAM unused; on unified memory running out can freeze or restart the machine. Use a smaller --ctx, close apps, or pass --unsafe-unified"
            )
        })
}

struct Run {
    log: LoadLog,
    observed: &'static str,
    reason: Option<String>,
    load_seconds: Option<f64>,
    speed: Option<SpeedPart>,
}

fn stop(child: &mut Child) {
    let _ = child.kill();
    let _ = child.wait();
}

/// Start the runtime, wait for ready or failure, time one request, stop it.
fn run_once(
    argv: &[String],
    log_path: &Path,
    timeout: Duration,
    watch_ram: Option<u64>,
) -> anyhow::Result<Run> {
    let port = free_port()?;
    let log = std::fs::File::create(log_path)
        .with_context(|| format!("creating {}", log_path.display()))?;
    let mut child = Command::new(&argv[0])
        .args(&argv[1..])
        .args(["--host", "127.0.0.1", "--port", &port.to_string()])
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .spawn()
        .with_context(|| format!("starting {}", argv[0]))?;
    let http = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?;
    let base = format!("http://127.0.0.1:{port}");
    let start = Instant::now();
    let read_log = || llama_log::parse(&std::fs::read_to_string(log_path).unwrap_or_default());
    let ended = |observed, reason: Option<String>| -> anyhow::Result<Run> {
        Ok(Run {
            log: read_log(),
            observed,
            reason,
            load_seconds: None,
            speed: None,
        })
    };
    loop {
        if let Some(status) = child.try_wait()? {
            let log = read_log();
            let (observed, reason) = match log.outcome() {
                Outcome::Oom(r) => ("oom", Some(r)),
                Outcome::Failed(r) => ("failed", Some(r)),
                _ => ("failed", Some(format!("exited before ready ({status})"))),
            };
            return Ok(Run {
                log,
                observed,
                reason,
                load_seconds: None,
                speed: None,
            });
        }
        if start.elapsed() > timeout {
            stop(&mut child);
            return ended(
                "timeout",
                Some(format!("not ready after {}s", timeout.as_secs())),
            );
        }
        if let Some(floor) = watch_ram
            && let Ok(snap) = SystemHw.snapshot()
            && snap.host.available_bytes < floor
        {
            stop(&mut child);
            return ended(
                "aborted",
                Some(format!(
                    "stopped: free RAM fell to {} MiB, under the safety floor of {} MiB",
                    snap.host.available_bytes >> 20,
                    floor >> 20
                )),
            );
        }
        if http
            .get(format!("{base}/health"))
            .timeout(Duration::from_secs(2))
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(250));
    }
    let load_seconds = (start.elapsed().as_secs_f64() * 10.0).round() / 10.0;
    let body = serde_json::json!({
        "prompt": vec!["hello"; PROMPT_WORDS].join(" "),
        "n_predict": GEN_TOKENS,
        "temperature": 0,
        "seed": 42,
        "cache_prompt": false,
        "ignore_eos": true,
    });
    let speed = http
        .post(format!("{base}/completion"))
        .json(&body)
        .send()
        .ok()
        .and_then(|r| r.json::<serde_json::Value>().ok())
        .and_then(|v| {
            let t = &v["timings"];
            Some(SpeedPart {
                tg_tok_s: (t["predicted_per_second"].as_f64()? * 100.0).round() / 100.0,
                pp_tok_s: (t["prompt_per_second"].as_f64()? * 100.0).round() / 100.0,
                prompt_tokens: t["prompt_n"].as_f64()? as u64,
                gen_tokens: t["predicted_n"].as_f64()? as u64,
            })
        });
    stop(&mut child);
    Ok(Run {
        log: read_log(),
        observed: "fits",
        reason: None,
        load_seconds: Some(load_seconds),
        speed,
    })
}

fn observed_part(run: Run) -> Observed {
    let mut buffers: Vec<DeviceBuffers> = run
        .log
        .devices()
        .into_iter()
        .chain(std::iter::once("CPU".to_owned()))
        .map(|d| DeviceBuffers {
            model: run.log.mib(&d, Kind::Model),
            kv: run.log.mib(&d, Kind::Kv) + run.log.mib(&d, Kind::Rs),
            compute: run.log.mib(&d, Kind::Compute),
            output: run.log.mib(&d, Kind::Output),
            device: d,
        })
        .collect();
    buffers.retain(|b| b.model + b.kv + b.compute + b.output > 0.0);
    let vision_compute: f64 = run
        .log
        .buffers
        .iter()
        .filter(|b| b.kind == Kind::VisionCompute)
        .map(|b| b.mib)
        .sum::<f64>()
        + 0.0;
    let vision =
        (vision_compute > 0.0 || run.log.vision_model_mib.is_some()).then(|| VisionBuffers {
            projector: run.log.projector.clone(),
            model_mib: run.log.vision_model_mib,
            compute_mib: vision_compute,
        });
    Observed {
        result: run.observed,
        reason: run.reason,
        load_seconds: run.load_seconds,
        buffers,
        vision,
        speed: run.speed,
    }
}

fn prediction_part(plan: Result<&Plan, &loadmaster_plan::NoFit>, ctx: u32) -> Prediction {
    match plan {
        Ok(p) => {
            let term = |k: &str| p.footprint.term(k).map(|t| mib(t.bytes));
            Prediction {
                result: "fits",
                ctx: p.ctx,
                kv: Some(p.kv.flag()),
                n_gpu_layers: Some(p.placement.n_gpu_layers),
                n_cpu_moe: Some(p.placement.n_cpu_moe),
                gpu_mib: mib(p.footprint.gpu_bytes),
                gpu_weights_mib: term("weights"),
                gpu_kv_mib: term("kv"),
                gpu_compute_mib: term("compute"),
                tok_s: p.speed.as_ref().and_then(|s| s.at.first()).map(|a| {
                    [
                        f64::from(a.tok_s_x10.0) / 10.0,
                        f64::from(a.tok_s_x10.1) / 10.0,
                    ]
                }),
            }
        }
        Err(n) => Prediction {
            result: "cannot_fit",
            ctx,
            kv: None,
            n_gpu_layers: None,
            n_cpu_moe: None,
            gpu_mib: mib(n.closest.gpu_bytes),
            gpu_weights_mib: None,
            gpu_kv_mib: None,
            gpu_compute_mib: None,
            tok_s: None,
        },
    }
}

fn percent_encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len() * 3);
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// The prefilled "new issue" link; `false` when the report is too long to prefill.
fn issue_url(title: &str, body: &str) -> (String, bool) {
    let base = format!(
        "{}/issues/new?template=validation.yml&labels=validation&title={}",
        env!("CARGO_PKG_REPOSITORY"),
        percent_encode(title)
    );
    let full = format!("{base}&report={}", percent_encode(body));
    if full.len() <= MAX_URL {
        (full, true)
    } else {
        (base, false)
    }
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_owned()
}

fn hardware_part(gpu: &loadmaster_hw::Gpu, host: &loadmaster_hw::Host) -> HardwarePart {
    HardwarePart {
        gpu: gpu.name.clone(),
        vendor: match gpu.vendor {
            loadmaster_hw::Vendor::Nvidia => "nvidia",
            loadmaster_hw::Vendor::Apple => "apple",
        },
        memory: match gpu.memory {
            MemoryKind::Dedicated => "dedicated",
            MemoryKind::Unified => "unified",
        },
        total_mib: gpu.total_bytes >> 20,
        usable_mib: gpu.usable_bytes >> 20,
        bandwidth_gbs: gpu.bandwidth_bytes_s.map(|b| b as f64 / 1e9),
        host_total_mib: host.total_bytes >> 20,
        host_available_mib: host.available_bytes >> 20,
    }
}

fn model_part(m: &ModelInfo) -> anyhow::Result<ModelPart> {
    Ok(ModelPart {
        name: m.name.clone().unwrap_or_else(|| "unnamed".into()),
        arch: m.arch.clone(),
        quant: m.quant.to_string(),
        n_layer: m.n_layer,
        weight_mib: mib(m.weight_bytes),
        sha256: super::lock::sha256(m, false)?,
        source: source_string(m),
        mmproj_mib: m.mmproj.as_ref().map(|p| mib(p.weight_bytes)),
    })
}

/// Write the report, print a predicted-vs-observed summary (the whole file with `--show`)
/// and the submit link. The issue page shows the full file before anything is sent.
fn finish(report: &Report, out_dir: &Path, title: &str, show: bool) -> anyhow::Result<ExitCode> {
    let text = format!(
        "# loadmaster report: review before sharing. It names no paths, hosts or users.\n{}",
        toml::to_string(report)?
    );
    let file = out_dir.join(format!("loadmaster-report-{}.toml", slug(title)));
    std::fs::write(&file, &text).with_context(|| format!("writing {}", file.display()))?;
    let style = crate::ui::style();
    let mut out = String::new();
    if show {
        out.push_str(&text);
        out.push('\n');
    }
    out.push_str(&summary(report, style));
    out.push_str(&format!(
        "\nwrote {} {}\n",
        file.display(),
        style.dim("(no paths, hosts or users; --show prints it)")
    ));
    let (url, prefilled) = issue_url(title, &text);
    if prefilled {
        out.push_str("review and submit it here (loadmaster sends nothing):\n");
    } else {
        out.push_str(
            "open this link and paste the file (too long to prefill; loadmaster sends nothing):\n",
        );
    }
    out.push_str(&format!("  {url}\n"));
    crate::ui::print(&out);
    Ok(ExitCode::SUCCESS)
}

/// `✓ Prediction held` and a predicted / observed table.
fn summary(r: &Report, style: crate::ui::Style) -> String {
    let (p, o) = (&r.prediction, &r.observed);
    let gpu = r
        .hardware
        .gpu
        .trim_start_matches("NVIDIA GeForce ")
        .trim_start_matches("NVIDIA ");
    let verdict = match (p.result, o.result) {
        ("fits", "fits") => style.green(&format!("✓ Prediction held: fits on {gpu}")),
        ("cannot_fit", "oom") => style.green(&format!("✓ Prediction held: does not fit on {gpu}")),
        (_, "fits" | "oom") => style.red(&format!(
            "✗ Prediction was wrong: predicted {}, observed {}",
            p.result.replace('_', " "),
            o.result
        )),
        (_, other) => style.yellow(&format!("! No verdict: the run ended with `{other}`")),
    };
    let gib = |mib: f64| format!("{:.1} GiB", mib / 1024.0);
    let predicted_buffers = match (p.gpu_weights_mib, p.gpu_kv_mib, p.gpu_compute_mib) {
        (Some(w), Some(k), Some(c)) => gib(w + k + c),
        _ => "-".into(),
    };
    let on_gpu: Vec<&DeviceBuffers> = o
        .buffers
        .iter()
        .filter(|b| !b.device.starts_with("CPU") && !b.device.ends_with("_Host"))
        .collect();
    let observed_buffers = if on_gpu.is_empty() {
        "-".into()
    } else {
        gib(on_gpu
            .iter()
            .map(|b| b.model + b.kv + b.compute + b.output)
            .sum())
    };
    let rows = [
        ("GPU buffers", predicted_buffers, observed_buffers),
        (
            "speed",
            p.tok_s
                .map_or("-".into(), |[lo, hi]| format!("{lo:.0}-{hi:.0} tok/s")),
            o.speed
                .as_ref()
                .map_or("-".into(), |s| format!("{:.1} tok/s", s.tg_tok_s)),
        ),
        (
            "load time",
            "-".into(),
            o.load_seconds.map_or("-".into(), |t| format!("{t:.1} s")),
        ),
    ];
    // Pad by the visible width, then style the words.
    let mut out = format!(
        "{verdict}\n  {:<12} {:>14}  {:>12}\n",
        "", "predicted", "observed"
    )
    .replace("predicted", &style.dim("predicted"))
    .replace("observed", &style.dim("observed"));
    for (label, pred, obs) in rows {
        out.push_str(&format!("  {label:<12} {pred:>14}  {obs:>12}\n"));
    }
    if let Some(reason) = &o.reason {
        out.push_str(&format!("  {}\n", style.dim(reason)));
    }
    out
}

pub fn run(a: Args) -> anyhow::Result<ExitCode> {
    if a.hw.pretend.is_some() {
        bail!("--pretend is for planning; a report records this machine's real hardware");
    }
    if let Some(name) = a.from_log.clone() {
        return from_log(&a, &name);
    }
    let Some(arg) = a.model.as_deref() else {
        bail!("pass a model (a .gguf file or a name from `loadmaster scan`), or --from-log NAME");
    };
    let model = super::open_model(arg).map_err(|e| {
        if arg.starts_with("hf://") {
            e.context("a report runs the model, so it must be on disk: download it first")
        } else {
            e
        }
    })?;
    if model.format != loadmaster_model::Format::Gguf {
        bail!("report runs llama.cpp; pass a GGUF model");
    }
    let replayed = a.hw.hw_file.is_some();
    let snap = super::snapshot(&a.hw)?;
    let gpu = super::pick_gpu(&snap, "auto")?;
    if gpu.name.contains("(pretend)") {
        bail!("this hardware snapshot is a --pretend one; a report records real hardware");
    }
    let req = PlanRequest {
        ctx: a.ctx,
        ..PlanRequest::default()
    };
    let planned = loadmaster_plan::plan(&model, &snap, &gpu, &req);
    let ctx = match &planned {
        Ok(p) => p.ctx,
        Err(PlanError::CannotFit(n)) => n.ctx,
        Err(_) => a.ctx.unwrap_or(4096),
    };
    let mut argv: Vec<String> = match &planned {
        Ok(p) => {
            if let Some(why) = unified_refusal(p)
                && !a.unsafe_unified
            {
                bail!(why);
            }
            println!(
                "plan: fits on {}, {} predicted",
                loadmaster_plan::explain::hw_label(&[&gpu]),
                loadmaster_plan::explain::gib1(p.footprint.gpu_bytes)
            );
            p.argv.clone()
        }
        Err(PlanError::CannotFit(n)) => {
            if gpu.memory == MemoryKind::Unified {
                bail!(
                    "the plan does not fit ({} short); on unified memory a failed load can freeze or restart the machine, so report only runs plans that fit here",
                    loadmaster_plan::gib(n.short_by)
                );
            }
            if !a.try_anyway {
                bail!(
                    "the plan does not fit ({} short). To check that prediction on this dedicated GPU, rerun with --try-anyway: it loads every layer on the GPU and records the result",
                    loadmaster_plan::gib(n.short_by)
                );
            }
            println!("plan: does not fit; trying every layer on the GPU to check");
            let mut v = vec![
                "llama-server".to_owned(),
                "-m".into(),
                model.path.display().to_string(),
            ];
            if let Some(m) = &model.mmproj {
                v.extend(["--mmproj".into(), m.path.display().to_string()]);
            }
            v.extend(
                [
                    "-c",
                    &ctx.to_string(),
                    "-ngl",
                    "999",
                    "-fa",
                    "on",
                    "-np",
                    "1",
                ]
                .map(String::from),
            );
            v
        }
        Err(e) => bail!("{e}"),
    };
    argv[0] = a.runtime_bin.clone();
    argv.extend(["-lv".into(), "4".into()]);
    // The report names the runtime by its file name, never where it is installed.
    let mut public = argv.clone();
    public[0] = Path::new(&argv[0])
        .file_name()
        .map_or("llama-server".into(), |n| n.to_string_lossy().into_owned());
    let shown = redact(&loadmaster_plan::emit::shell_join(&public), &model);
    if crate::ui::style().wrap {
        println!(
            "will run:\n  {}",
            loadmaster_plan::emit::shell_wrap(&argv, 88).join("\n  ")
        );
    } else {
        println!("will run: {}", loadmaster_plan::emit::shell_join(&argv));
    }
    println!("(stops after one {GEN_TOKENS}-token request; the log is kept for you)");
    if !a.yes && !confirm("continue?")? {
        println!("not started");
        return Ok(ExitCode::from(3));
    }
    // On unified memory, stop the run if free RAM falls below the planner's margin.
    let watch = (!replayed && gpu.memory == MemoryKind::Unified)
        .then(|| snap.host.available_bytes / 100 * UNIFIED_RAM_MARGIN_PCT);
    let dir = super::state_dir().join("report");
    std::fs::create_dir_all(&dir)?;
    let stamp = loadmaster_probe::time::now().replace([':', '-'], "");
    let log_path = dir.join(format!("{stamp}-{}.log", slug(&model.quant.to_string())));
    let run = run_once(&argv, &log_path, Duration::from_secs(a.timeout_s), watch)?;
    eprintln!("runtime log: {}", log_path.display());
    let build = run.log.build.clone();
    let mut observed = observed_part(run);
    observed.reason = observed.reason.map(|r| redact(&r, &model));
    let prediction = match &planned {
        Ok(p) => prediction_part(Ok(p), ctx),
        Err(PlanError::CannotFit(n)) => prediction_part(Err(n), ctx),
        Err(_) => unreachable!("other errors returned above"),
    };
    let title = format!(
        "{} {} on {} at ctx {}: predicted {}, observed {}",
        model.name.as_deref().unwrap_or("model"),
        model.quant,
        gpu.name,
        ctx,
        prediction.result,
        observed.result
    );
    let report = Report {
        report: Meta {
            loadmaster: env!("CARGO_PKG_VERSION"),
            date: loadmaster_probe::time::now()[..10].to_owned(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            runtime: "llama.cpp",
            runtime_build: build,
            command: shown,
            mode: "run",
        },
        model: model_part(&model)?,
        hardware: hardware_part(&gpu, &snap.host),
        prediction,
        observed,
    };
    finish(&report, &a.out, &title, a.show)
}

/// `--from-log NAME`: the last start of NAME in `loadmaster up`'s log, against its
/// models.lock entry. No new run, so no speed.
fn from_log(a: &Args, name: &str) -> anyhow::Result<ExitCode> {
    let lock = loadmaster_lock::Lockfile::load(&a.lock)?;
    let entry = lock
        .model(name)
        .with_context(|| format!("no model `{name}` in {}", a.lock.display()))?;
    let path = super::state_dir().join("logs").join(format!("{name}.log"));
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
    let last = text
        .rsplit("--- loadmaster: starting ")
        .next()
        .unwrap_or(&text);
    let log = llama_log::parse(last);
    if log.buffers.is_empty() {
        bail!(
            "{} has no buffer sizes: llama-server prints them only at -lv 4. Add \"-lv\", \"4\" to `flags` for `{name}` in models.lock, restart `loadmaster up`, load the model once, then rerun",
            path.display()
        );
    }
    let model = super::open_model(&entry.source)
        .with_context(|| format!("opening `{name}`'s source {}", entry.source))?;
    let snap = super::snapshot(&a.hw)?;
    let gpu = super::pick_gpu(&snap, "auto")?;
    let req = PlanRequest {
        ctx: Some(entry.ctx),
        ..PlanRequest::default()
    };
    let planned = loadmaster_plan::plan(&model, &snap, &gpu, &req);
    let (observed, reason) = match log.outcome() {
        Outcome::Ready => ("fits", None),
        Outcome::Oom(r) => ("oom", Some(r)),
        Outcome::Failed(r) => ("failed", Some(r)),
        Outcome::Unknown => ("unknown", None),
    };
    let command = last.lines().next().unwrap_or_default().to_owned();
    let run = Run {
        log,
        observed,
        reason: reason.map(|r| redact(&r, &model)),
        load_seconds: None,
        speed: None,
    };
    let prediction = match &planned {
        Ok(p) => prediction_part(Ok(p), entry.ctx),
        Err(PlanError::CannotFit(n)) => prediction_part(Err(n), entry.ctx),
        Err(e) => bail!("{e}"),
    };
    let title = format!(
        "{} {} on {} at ctx {}: predicted {}, observed {}",
        model.name.as_deref().unwrap_or(name),
        model.quant,
        gpu.name,
        entry.ctx,
        prediction.result,
        observed
    );
    let build = run.log.build.clone();
    let report = Report {
        report: Meta {
            loadmaster: env!("CARGO_PKG_VERSION"),
            date: loadmaster_probe::time::now()[..10].to_owned(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            runtime: "llama.cpp",
            runtime_build: build,
            command: redact(&command, &model),
            mode: "log",
        },
        model: model_part(&model)?,
        hardware: hardware_part(&gpu, &snap.host),
        prediction,
        observed: observed_part(run),
    };
    finish(&report, &a.out, &title, a.show)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urls_are_encoded_and_bounded() {
        assert_eq!(percent_encode("a b/c=d&é"), "a%20b%2Fc%3Dd%26%C3%A9");
        let (u, prefilled) = issue_url("t", "x = 1");
        assert!(prefilled && u.ends_with("&report=x%20%3D%201"), "{u}");
        let (u, prefilled) = issue_url("t", &"x".repeat(MAX_URL));
        assert!(!prefilled && !u.contains("report="));
    }

    #[test]
    fn slugs() {
        assert_eq!(
            slug("Qwen3 8B Q4_K_M on NVIDIA GeForce RTX 3090 at ctx 32768"),
            "qwen3-8b-q4-k-m-on-nvidia-geforce-rtx-3090-at-ctx-32768"
        );
    }
}
