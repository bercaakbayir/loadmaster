use std::path::{Path, PathBuf};
use std::process::ExitCode;

use loadmaster_hw::{Gpu, HwSnapshot};
use loadmaster_model::{Format, ModelInfo, Quant, Source};
use loadmaster_plan::{Error, PlanRequest, Runtime, gib};
use serde::Serialize;

const CTXS: [u32; 3] = [8192, 32768, 131072];

/// Whether a model fits at a context: entirely on the GPU, only with offload, or not.
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Fit {
    Gpu,
    Offload,
    No,
    /// Not applicable: beyond the training context, no GPU, or no runtime for this format.
    Na,
}

impl Fit {
    fn cell(self) -> &'static str {
        match self {
            Fit::Gpu => "gpu",
            Fit::Offload => "offload",
            Fit::No => "no",
            Fit::Na => "-",
        }
    }
}

fn runtime_for(m: &ModelInfo) -> Runtime {
    match (&m.format, &m.quant) {
        (Format::Gguf, _) => Runtime::LlamaCpp,
        (_, Quant::Exl3 { .. }) => Runtime::Exl3,
        _ => Runtime::Vllm,
    }
}

fn fit(m: &ModelInfo, hw: &HwSnapshot, gpu: Option<&Gpu>, ctx: u32) -> Fit {
    let Some(gpu) = gpu else { return Fit::Na };
    if ctx > m.n_ctx_train {
        return Fit::Na;
    }
    let req = PlanRequest {
        ctx: Some(ctx),
        runtime: runtime_for(m),
        ..PlanRequest::default()
    };
    match loadmaster_plan::plan(m, hw, gpu, &req) {
        Ok(p) if p.placement.is_full(m.n_layer) => Fit::Gpu,
        Ok(_) => Fit::Offload,
        Err(Error::CannotFit(_)) => Fit::No,
        Err(_) => Fit::Na,
    }
}

fn source_kind(s: &Source) -> &'static str {
    match s {
        Source::HfCache { .. } => "hf",
        Source::Ollama { .. } => "ollama",
        Source::LmStudio { .. } => "lmstudio",
        Source::Loose => "path",
    }
}

#[derive(Serialize)]
struct Row<'a> {
    name: &'a str,
    path: &'a Path,
    source: &'a Source,
    size_bytes: u64,
    arch: Option<&'a str>,
    quant: Option<String>,
    n_ctx_train: Option<u32>,
    mmproj: Option<&'a Path>,
    chat_template_blake3: Option<&'a str>,
    runtime: Option<&'static str>,
    /// Fit at 8k, 32k and 128k context.
    fits: Option<[Fit; 3]>,
    error: Option<&'a str>,
}

#[derive(Serialize)]
struct JsonOut<'a> {
    gpu: Option<&'a str>,
    models: Vec<Row<'a>>,
    issues: &'a [loadmaster_store::Issue],
}

pub fn run(path: Vec<PathBuf>, json: bool, hw_file: Option<&Path>) -> anyhow::Result<ExitCode> {
    let scan = loadmaster_store::scan(&super::roots(path));
    let hw = super::snapshot(hw_file)?;
    let gpu = hw.best_gpu();
    let rows: Vec<Row> = scan
        .entries
        .iter()
        .map(|e| {
            let info = e.info.as_deref().ok();
            Row {
                name: &e.name,
                path: &e.path,
                source: &e.source,
                size_bytes: e.size_bytes,
                arch: info.map(|m| m.arch.as_str()),
                quant: info.map(|m| m.quant.to_string()),
                n_ctx_train: info.map(|m| m.n_ctx_train),
                mmproj: info
                    .and_then(|m| m.mmproj.as_ref())
                    .map(|m| m.path.as_path()),
                chat_template_blake3: info
                    .and_then(|m| m.chat_template.as_ref())
                    .map(|t| t.blake3.as_str()),
                runtime: info.map(|m| runtime_for(m).name()),
                fits: info.map(|m| CTXS.map(|c| fit(m, &hw, gpu, c))),
                error: e.info.as_ref().err().map(String::as_str),
            }
        })
        .collect();

    if json {
        let out = JsonOut {
            gpu: gpu.map(|g| g.name.as_str()),
            models: rows,
            issues: &scan.issues,
        };
        println!("{}", serde_json::to_string_pretty(&out)?);
        return Ok(ExitCode::SUCCESS);
    }

    if rows.is_empty() {
        println!("no models found (looked in the HF cache, Ollama, LM Studio and --path dirs)");
    } else {
        let name_w = rows
            .iter()
            .map(|r| r.name.len())
            .max()
            .unwrap_or(4)
            .clamp(4, 60);
        println!(
            "{:<name_w$}  {:<8} {:>10}  {:<10} {:<12} {:>7}  {:<6} {:<8}  {:<7} {:<7} {:<7}",
            "NAME",
            "SOURCE",
            "SIZE",
            "ARCH",
            "QUANT",
            "CTX",
            "MMPROJ",
            "TEMPLATE",
            "8k",
            "32k",
            "128k"
        );
        for r in &rows {
            let fits = r.fits.unwrap_or([Fit::Na; 3]);
            println!(
                "{:<name_w$}  {:<8} {:>10}  {:<10} {:<12} {:>7}  {:<6} {:<8}  {:<7} {:<7} {:<7}",
                truncate(r.name, name_w),
                source_kind(r.source),
                gib(r.size_bytes),
                r.arch.unwrap_or("?"),
                r.quant.as_deref().unwrap_or("?"),
                r.n_ctx_train.map_or("?".into(), |c| c.to_string()),
                if r.mmproj.is_some() { "yes" } else { "-" },
                r.chat_template_blake3.map_or("-", |h| &h[..8]),
                fits[0].cell(),
                fits[1].cell(),
                fits[2].cell(),
            );
            if let Some(err) = r.error {
                println!("{:<name_w$}  error: {err}", "");
            }
        }
        match gpu {
            Some(g) => println!(
                "\nfits on {}: gpu = entirely on GPU, offload = with CPU offload",
                g.name
            ),
            None => println!("\nno GPU detected; fit columns are empty"),
        }
    }
    for i in &scan.issues {
        println!("warning: {}: {}", i.path.display(), i.problem);
    }
    Ok(ExitCode::SUCCESS)
}

fn truncate(s: &str, w: usize) -> String {
    if s.chars().count() <= w {
        s.to_owned()
    } else {
        let keep: String = s.chars().take(w - 1).collect();
        format!("{keep}…")
    }
}
