use std::path::{Path, PathBuf};
use std::process::ExitCode;

use loadmaster_hw::{Gpu, HwSnapshot};
use loadmaster_model::{Format, ModelInfo, Quant, Source};
use loadmaster_plan::{Error, KvType, PlanRequest, Runtime};
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

/// A fit that comes with a cost the table marks with `*`.
#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Caveat {
    /// Only with a q4_0 KV cache, which lowers output quality.
    Q4Kv,
    /// Leaves under the unified-memory RAM safety margin.
    TightRam,
}

fn fit(m: &ModelInfo, hw: &HwSnapshot, gpu: Option<&Gpu>, ctx: u32) -> (Fit, Option<Caveat>) {
    let Some(gpu) = gpu else {
        return (Fit::Na, None);
    };
    if ctx > m.n_ctx_train {
        return (Fit::Na, None);
    }
    let req = PlanRequest {
        ctx: Some(ctx),
        runtime: runtime_for(m),
        ..PlanRequest::default()
    };
    match loadmaster_plan::plan(m, hw, gpu, &req) {
        Ok(p) => {
            let caveat = if p
                .warnings
                .iter()
                .any(|w| w.is(loadmaster_plan::WarningKind::TightRam))
            {
                Some(Caveat::TightRam)
            } else if p.kv == KvType::Q4_0 {
                Some(Caveat::Q4Kv)
            } else {
                None
            };
            let fit = if p.placement.is_full(m.n_layer) {
                Fit::Gpu
            } else {
                Fit::Offload
            };
            (fit, caveat)
        }
        Err(Error::CannotFit(_)) => (Fit::No, None),
        Err(_) => (Fit::Na, None),
    }
}

fn source_kind(s: &Source) -> &'static str {
    match s {
        Source::HfCache { .. } => "hf",
        Source::Ollama { .. } => "ollama",
        Source::LmStudio { .. } => "lmstudio",
        Source::Loose => "path",
        Source::HfRemote { .. } => "hf-remote",
    }
}

#[derive(Serialize)]
struct Row<'a> {
    name: &'a str,
    path: &'a Path,
    source: &'a Source,
    size_bytes: u64,
    arch: Option<&'a str>,
    /// `chat`, `vision` (chat with a projector) or `embed`.
    kind: Option<&'static str>,
    quant: Option<String>,
    n_ctx_train: Option<u32>,
    mmproj: Option<&'a Path>,
    chat_template_blake3: Option<&'a str>,
    runtime: Option<&'static str>,
    /// Fit at 8k, 32k and 128k context.
    fits: Option<[Fit; 3]>,
    /// What each fit costs, if anything.
    caveats: Option<[Option<Caveat>; 3]>,
    error: Option<&'a str>,
}

#[derive(Serialize)]
struct JsonOut<'a> {
    gpu: Option<&'a str>,
    models: Vec<Row<'a>>,
    issues: &'a [loadmaster_store::Issue],
}

pub fn run(
    path: Vec<PathBuf>,
    json: bool,
    wide: bool,
    hw: &crate::HwArgs,
) -> anyhow::Result<ExitCode> {
    let scan = loadmaster_store::scan(&super::roots(path));
    let hw = super::snapshot(hw)?;
    let gpu = hw.best_gpu();
    let rows: Vec<Row> = scan
        .entries
        .iter()
        .map(|e| {
            let info = e.info.as_deref().ok();
            let fits = info.map(|m| CTXS.map(|c| fit(m, &hw, gpu, c)));
            Row {
                name: &e.name,
                path: &e.path,
                source: &e.source,
                size_bytes: e.size_bytes,
                arch: info.map(|m| m.arch.as_str()),
                kind: info.map(|m| {
                    if m.is_embedding() {
                        "embed"
                    } else if m.mmproj.is_some() {
                        "vision"
                    } else {
                        "chat"
                    }
                }),
                quant: info.map(|m| m.quant.to_string()),
                n_ctx_train: info.map(|m| m.n_ctx_train),
                mmproj: info
                    .and_then(|m| m.mmproj.as_ref())
                    .map(|m| m.path.as_path()),
                chat_template_blake3: info
                    .and_then(|m| m.chat_template.as_ref())
                    .map(|t| t.blake3.as_str()),
                runtime: info.map(|m| runtime_for(m).name()),
                fits: fits.map(|f| f.map(|(fit, _)| fit)),
                caveats: fits.map(|f| f.map(|(_, c)| c)),
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

    let style = crate::ui::style();
    let mut out = String::new();
    if rows.is_empty() {
        out.push_str(
            "no models found (looked in the HF cache, Ollama, LM Studio and --path dirs)\n",
        );
    } else {
        let name_w = rows
            .iter()
            .map(|r| r.name.chars().count())
            .max()
            .unwrap_or(4)
            .clamp(4, 60);
        let mut header = format!(
            "{:<name_w$}  {:<8}  {:<6}  {:>9}  {:<10}",
            "NAME", "SOURCE", "KIND", "SIZE", "QUANT"
        );
        if wide {
            header.push_str(&format!(
                "  {:<10}  {:>7}  {:<6}  {:<8}",
                "ARCH", "CTX", "MMPROJ", "TEMPLATE"
            ));
        }
        header.push_str(&format!("  {:<8} {:<8} {}", "8k", "32k", "128k"));
        out.push_str(style.bold(header.trim_end()).as_str());
        out.push('\n');
        for r in &rows {
            let fits = r.fits.unwrap_or([Fit::Na; 3]);
            let caveats = r.caveats.unwrap_or([None; 3]);
            let cell = |i: usize| {
                let c = fits[i].cell();
                let text = if caveats[i].is_some() {
                    format!("{c}*")
                } else {
                    c.to_owned()
                };
                // The last column is not padded, so lines end without spaces.
                let padded = if i < 2 { format!("{text:<8}") } else { text };
                match fits[i] {
                    Fit::Gpu if caveats[i].is_none() => style.green(&padded),
                    Fit::Gpu | Fit::Offload => style.yellow(&padded),
                    Fit::No => style.red(&padded),
                    Fit::Na => style.dim(&padded),
                }
            };
            let mut line = format!(
                "{:<name_w$}  {:<8}  {:<6}  {:>9}  {:<10}",
                truncate(r.name, name_w),
                source_kind(r.source),
                r.kind.unwrap_or("?"),
                loadmaster_plan::explain::gib1(r.size_bytes),
                r.quant.as_deref().unwrap_or("?"),
            );
            if wide {
                line.push_str(&format!(
                    "  {:<10}  {:>7}  {:<6}  {:<8}",
                    r.arch.unwrap_or("?"),
                    r.n_ctx_train.map_or("?".into(), |c| c.to_string()),
                    if r.mmproj.is_some() { "yes" } else { "-" },
                    r.chat_template_blake3.map_or("-", |h| &h[..8]),
                ));
            }
            line.push_str(&format!("  {} {} {}", cell(0), cell(1), cell(2)));
            out.push_str(line.trim_end());
            out.push('\n');
            if let Some(err) = r.error {
                out.push_str(&format!("{:<name_w$}  error: {err}\n", ""));
            }
        }
        out.push('\n');
        match gpu {
            Some(g) => {
                let tight = rows
                    .iter()
                    .any(|r| r.caveats.is_some_and(|c| c.iter().any(Option::is_some)));
                out.push_str(&style.dim(&format!(
                    "on {}: gpu = fits on GPU · offload = needs CPU",
                    loadmaster_plan::explain::hw_label(&[g])
                )));
                out.push('\n');
                if tight {
                    out.push_str(&style.dim(&format!(
                        "* = only with q4_0 KV or under the {}% free-RAM margin; `loadmaster plan <name> --ctx <n>` says which",
                        loadmaster_plan::UNIFIED_RAM_MARGIN_PCT
                    )));
                    out.push('\n');
                }
            }
            None => out.push_str("no GPU detected; fit columns are empty\n"),
        }
    }
    if wide {
        for i in &scan.issues {
            out.push_str(&format!(
                "{} {}: {}\n",
                style.yellow("!"),
                i.path.display(),
                i.problem
            ));
        }
    } else if !scan.issues.is_empty() {
        let n = scan.issues.len();
        out.push_str(&format!(
            "{} {n} {} skipped (--wide lists {})\n",
            style.yellow("!"),
            if n == 1 { "directory" } else { "directories" },
            if n == 1 { "it" } else { "them" }
        ));
    }
    crate::ui::print(&out);
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
