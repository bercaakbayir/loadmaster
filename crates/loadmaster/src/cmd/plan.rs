use std::process::ExitCode;

use loadmaster_plan::{Error, PlanRequest, explain};
use serde::Serialize;

use crate::PlanArgs;

/// `--json` output: one object whatever the outcome.
#[derive(Serialize)]
#[serde(tag = "result", rename_all = "snake_case")]
enum JsonOut<'a> {
    Fits { plan: &'a loadmaster_plan::Plan },
    CannotFit { no_fit: &'a loadmaster_plan::NoFit },
}

pub fn run(args: PlanArgs) -> anyhow::Result<ExitCode> {
    let gguf = args.runtime == crate::Runtime::LlamaCpp;
    let model = match super::open_target(&args.model, args.offline, gguf)? {
        super::Target::Model(m) => *m,
        super::Target::Repo { spec, files } => return repo_table(&args, &spec, &files),
    };
    let snap = super::snapshot(&args.hw)?;
    let (gpu, extra_gpus) = super::pick_gpus(&snap, &args.gpu)?;

    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args.runtime.into(),
        mmproj_offload: !args.no_mmproj_offload,
        extra_gpus,
        ..PlanRequest::default()
    };
    match loadmaster_plan::plan(&model, &snap, &gpu, &req) {
        Ok(p) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::Fits { plan: &p })?
                );
            } else if args.command {
                println!("{}", loadmaster_plan::emit::command(&p));
            } else if args.explain {
                crate::ui::print(&explain::explain(&p, crate::ui::style()));
            } else {
                crate::ui::print(&explain::summary(&p, crate::ui::style()));
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(Error::CannotFit(n)) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::CannotFit { no_fit: &n })?
                );
            } else if !args.command {
                crate::ui::print(&explain::no_fit(&n, args.explain, crate::ui::style()));
            }
            Ok(ExitCode::from(1))
        }
        Err(e) => Err(e.into()),
    }
}

/// One quant in a Hub repo: its file (first part if split), label and total size.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RepoQuant {
    pub(crate) file: String,
    pub(crate) label: String,
    pub(crate) size: u64,
}

/// The quant in a GGUF file name: `Qwen3-30B-A3B-UD-Q4_K_XL-00001-of-00002.gguf` is
/// `UD-Q4_K_XL`. Falls back to the file stem.
fn quant_label(file: &str) -> String {
    let name = file.rsplit('/').next().unwrap_or(file);
    let stem = name.strip_suffix(".gguf").unwrap_or(name);
    let tokens: Vec<&str> = stem.split(['-', '.']).collect();
    let is_quant = |t: &str| {
        let u = t.to_ascii_uppercase();
        let digit_after = |p: &str| {
            u.strip_prefix(p)
                .is_some_and(|r| r.starts_with(|c: char| c.is_ascii_digit()))
        };
        digit_after("Q")
            || digit_after("IQ")
            || digit_after("TQ")
            || ["BF16", "F16", "F32", "FP16", "FP32", "MXFP4"].contains(&u.as_str())
    };
    match tokens.iter().position(|t| is_quant(t)) {
        Some(i) => {
            let start = if i > 0 && tokens[i - 1].eq_ignore_ascii_case("ud") {
                i - 1
            } else {
                i
            };
            let end = tokens[i..]
                .iter()
                .position(|t| t.len() == 5 && t.bytes().all(|b| b.is_ascii_digit()))
                .map_or(tokens.len(), |j| i + j);
            tokens[start..end].join("-")
        }
        None => stem.to_owned(),
    }
}

/// GGUF files in model repos that are not the model: importance matrices
/// (`imatrix_unsloth.gguf`) and multi-token-prediction draft heads (`MTP/mtp-*.gguf`).
fn is_auxiliary(path: &str) -> bool {
    let lower = path.to_ascii_lowercase();
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    name.contains("imatrix") || name.starts_with("mtp-") || lower.starts_with("mtp/")
}

/// The GGUF quants in a repo listing, smallest first; split parts are summed into their
/// first part, projectors and non-GGUF files skipped.
pub(crate) fn repo_quants(files: &[loadmaster_store::remote::RemoteFile]) -> Vec<RepoQuant> {
    let mut quants: Vec<RepoQuant> = Vec::new();
    let projectors = loadmaster_model::mmproj_candidates(files.iter().map(|f| f.path.as_str()));
    for f in files {
        if !f.path.ends_with(".gguf")
            || projectors.contains(&f.path.as_str())
            || is_auxiliary(&f.path)
        {
            continue;
        }
        // `name-00002-of-00003.gguf` belongs to `name-00001-of-00003.gguf`.
        let first = match f
            .path
            .strip_suffix(".gguf")
            .and_then(|s| s.rsplit_once("-of-"))
        {
            Some((head, count))
                if count.len() == 5
                    && head.len() > 6
                    && head.as_bytes()[head.len() - 6] == b'-' =>
            {
                format!("{}00001-of-{count}.gguf", &head[..head.len() - 5])
            }
            _ => f.path.clone(),
        };
        match quants.iter_mut().find(|q| q.file == first) {
            Some(q) => q.size += f.size,
            None => quants.push(RepoQuant {
                label: quant_label(&first),
                file: first,
                size: f.size,
            }),
        }
    }
    quants.sort_by_key(|q| (q.size, q.file.clone()));
    quants
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RepoRow {
    pub(crate) quant: String,
    pub(crate) file: String,
    pub(crate) size_bytes: u64,
    /// `gpu`, `offload` or `no`.
    pub(crate) fit: &'static str,
    pub(crate) ctx: u32,
    pub(crate) kv: Option<&'static str>,
    pub(crate) gpu_bytes: u64,
    /// Bytes over budget when it does not fit.
    pub(crate) short_by: Option<u64>,
    /// False when sized from the file size, not the file's own header.
    pub(crate) exact: bool,
    /// Estimated generation tok/s x10 at the planned context.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) tok_s_x10: Option<(u32, u32)>,
}

/// Every GGUF quant of a Hub repo planned on this hardware: the base quant (Q4_K_M, else
/// the median) from its own header, the others from it scaled to their file size.
pub(crate) struct RepoPlan {
    pub(crate) model: loadmaster_model::ModelInfo,
    pub(crate) base: RepoQuant,
    pub(crate) rows: Vec<RepoRow>,
}

/// The largest quant that runs entirely on the GPU, else the smallest that runs at all.
pub(crate) fn best_fit(rows: &[RepoRow]) -> Option<usize> {
    rows.iter()
        .rposition(|r| r.fit == "gpu")
        .or_else(|| rows.iter().position(|r| r.fit != "no"))
}

/// Plan every quant in `files` (one header fetch). Shared by `plan hf://org/repo` and
/// `suggest`.
pub(crate) fn plan_repo_quants(
    hub: &loadmaster_store::remote::Hub,
    spec: &loadmaster_store::remote::Spec,
    files: &[loadmaster_store::remote::RemoteFile],
    snap: &loadmaster_hw::HwSnapshot,
    gpu: &loadmaster_hw::Gpu,
    req: &PlanRequest,
) -> anyhow::Result<RepoPlan> {
    let quants = repo_quants(files);
    if quants.is_empty() {
        anyhow::bail!(
            "hf://{}: no GGUF files; for a safetensors repo pass --runtime vllm",
            spec.repo
        );
    }
    let base = quants
        .iter()
        .find(|q| q.label.eq_ignore_ascii_case("Q4_K_M"))
        .unwrap_or(&quants[quants.len() / 2])
        .clone();
    let model = hub.open_gguf(&spec.repo, &spec.rev, &base.file, files)?;
    // Header bytes are in the file size but not in the weights.
    let header = files
        .iter()
        .find(|f| f.path == base.file)
        .map_or(0, |f| f.size.saturating_sub(model.weight_bytes));
    let base_weights = base.size.saturating_sub(header).max(1) as f64;
    let rows: Vec<RepoRow> = quants
        .iter()
        .map(|q| -> anyhow::Result<RepoRow> {
            let exact = q.file == base.file;
            let m = if exact {
                model.clone()
            } else {
                let ratio = q.size.saturating_sub(header) as f64 / base_weights;
                loadmaster_plan::alternatives::requantized(&model, &q.label, ratio)
            };
            let row = |fit, ctx, kv, gpu_bytes, short_by, tok_s_x10| RepoRow {
                quant: q.label.clone(),
                file: q.file.clone(),
                size_bytes: q.size,
                fit,
                ctx,
                kv,
                gpu_bytes,
                short_by,
                exact,
                tok_s_x10,
            };
            Ok(match loadmaster_plan::plan(&m, snap, gpu, req) {
                Ok(p) => row(
                    if p.placement.is_full(m.n_layer) {
                        "gpu"
                    } else {
                        "offload"
                    },
                    p.ctx,
                    Some(p.kv.flag()),
                    p.footprint.gpu_bytes,
                    None,
                    p.speed
                        .as_ref()
                        .and_then(|s| s.at.last())
                        .map(|a| a.tok_s_x10),
                ),
                Err(Error::CannotFit(n)) => row(
                    "no",
                    n.ctx,
                    None,
                    n.closest.gpu_bytes,
                    Some(n.short_by),
                    None,
                ),
                Err(e) => return Err(e.into()),
            })
        })
        .collect::<anyhow::Result<_>>()?;
    Ok(RepoPlan { model, base, rows })
}

/// `plan hf://org/repo`: every GGUF quant in the repo against this GPU. One header is
/// fetched and planned exactly; the other quants are that model scaled to their file size.
fn repo_table(
    args: &PlanArgs,
    spec: &loadmaster_store::remote::Spec,
    files: &[loadmaster_store::remote::RemoteFile],
) -> anyhow::Result<ExitCode> {
    if args.runtime != crate::Runtime::LlamaCpp {
        anyhow::bail!(
            "hf://{}: a GGUF repo; plan it with --runtime llama.cpp, or pass a safetensors repo for {}",
            spec.repo,
            loadmaster_plan::Runtime::from(args.runtime).name()
        );
    }
    let snap = super::snapshot(&args.hw)?;
    let (gpu, extra_gpus) = super::pick_gpus(&snap, &args.gpu)?;
    let hw = explain::hw_label(&std::iter::once(&gpu).chain(&extra_gpus).collect::<Vec<_>>());
    let usable: u64 = gpu.usable_bytes + extra_gpus.iter().map(|g| g.usable_bytes).sum::<u64>();
    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args.runtime.into(),
        mmproj_offload: !args.no_mmproj_offload,
        extra_gpus,
        ..PlanRequest::default()
    };
    let RepoPlan { model, base, rows } =
        plan_repo_quants(&super::hub(args.offline), spec, files, &snap, &gpu, &req)?;
    let any_fit = rows.iter().any(|r| r.fit != "no");
    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "repo": spec.repo,
                "revision": spec.rev,
                "gpu": gpu.name,
                "header_from": base.file,
                "quants": rows,
            }))?
        );
    } else {
        let style = crate::ui::style();
        let pick = best_fit(&rows);
        let fit_cell = |r: &RepoRow| {
            let est = if r.exact { "" } else { "~" };
            match r.short_by {
                Some(s) => format!("✗ {est}{} short", explain::gib1(s)),
                None => format!("{est}{}", r.fit),
            }
        };
        let ctx_cell = |r: &RepoRow| {
            if r.fit == "no" {
                "-".to_owned()
            } else {
                r.ctx.to_string()
            }
        };
        let speed = rows.iter().any(|r| r.tok_s_x10.is_some());
        let tok_s = |r: &RepoRow| match r.tok_s_x10 {
            Some(t) => format!(
                "~{}",
                loadmaster_plan::speed::band_text(t).trim_end_matches(" tok/s")
            ),
            None => "-".to_owned(),
        };
        let w = rows.iter().map(|r| r.quant.len()).max().unwrap_or(5).max(5);
        let fw = rows
            .iter()
            .map(|r| fit_cell(r).chars().count())
            .max()
            .unwrap_or(3)
            .max(3);
        let tw = if speed {
            rows.iter()
                .map(|r| tok_s(r).len())
                .max()
                .unwrap_or(5)
                .max(5)
        } else {
            0
        };
        let mut out = String::new();
        let line = |out: &mut String, cells: [&str; 6]| {
            let mut l = format!(
                "{:<w$}  {:>9}  {:<fw$}  {:>6}  {:<4}",
                cells[0], cells[1], cells[2], cells[3], cells[4]
            );
            if speed {
                l.push_str(&format!("  {:>tw$}", cells[5]));
            }
            out.push_str(l.trim_end());
        };
        out.push_str(&format!(
            "{}  ·  {} quants  ·  {hw}, {} usable\n\n",
            style.bold(model.name.as_deref().unwrap_or(&spec.repo)),
            rows.len(),
            explain::gib1(usable)
        ));
        line(&mut out, ["QUANT", "SIZE", "FIT", "CTX", "KV", "TOK/S"]);
        out.push('\n');
        for (i, r) in rows.iter().enumerate() {
            let size = explain::gib1(r.size_bytes);
            let fit = fit_cell(r);
            let ctx = ctx_cell(r);
            let speed = tok_s(r);
            let mut row = String::new();
            line(
                &mut row,
                [&r.quant, &size, &fit, &ctx, r.kv.unwrap_or("-"), &speed],
            );
            if Some(i) == pick {
                out.push_str(&style.bold(&format!("{row}  ← best fit")));
            } else if r.fit == "no" {
                out.push_str(&style.dim(&row));
            } else {
                out.push_str(&row);
            }
            out.push('\n');
        }
        out.push('\n');
        if rows.iter().any(|r| !r.exact) {
            out.push_str(&style.dim(&format!(
                "~ = planned from the {} header, scaled to that file's size\n",
                base.label
            )));
        }
        let r = &rows[pick.unwrap_or(0)];
        out.push_str(&format!(
            "exact plan{}: loadmaster plan hf://{}/{}\n",
            if pick.is_some() {
                " for the best fit"
            } else {
                " for the smallest"
            },
            spec.repo,
            r.file
        ));
        crate::ui::print(&out);
    }
    Ok(ExitCode::from(if any_fit { 0 } else { 1 }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use loadmaster_store::remote::RemoteFile;

    #[test]
    fn quant_labels_from_file_names() {
        for (file, label) in [
            ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Q4_K_M"),
            ("Qwen3-30B-A3B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
            (
                "Q4_K_M/Qwen3-235B-A22B-Q4_K_M-00001-of-00003.gguf",
                "Q4_K_M",
            ),
            ("gpt-oss-20b-mxfp4.gguf", "mxfp4"),
            ("BF16/Qwen3-30B-A3B-BF16-00001-of-00002.gguf", "BF16"),
            ("Llama-3.2-1B-Instruct-IQ4_XS.gguf", "IQ4_XS"),
            // Qwen's own naming: lower case, `fp16`.
            ("qwen2.5-coder-7b-instruct-q4_k_m.gguf", "q4_k_m"),
            ("qwen2.5-coder-7b-instruct-fp16-00001-of-00004.gguf", "fp16"),
            ("model.gguf", "model"),
        ] {
            assert_eq!(quant_label(file), label, "{file}");
        }
    }

    #[test]
    fn split_parts_are_summed_and_projectors_skipped() {
        let f = |path: &str, size| RemoteFile {
            path: path.into(),
            size,
        };
        let q = repo_quants(&[
            f("BF16/M-BF16-00001-of-00002.gguf", 30),
            f("BF16/M-BF16-00002-of-00002.gguf", 20),
            f("M-Q4_K_M.gguf", 10),
            f("mmproj-F16.gguf", 1),
            f("README.md", 1),
            // Not models: an importance matrix and multi-token-prediction heads.
            f("imatrix_unsloth.gguf", 1),
            f("MTP/mtp-M-Q4_K_M.gguf", 2),
            f("mtp-M-BF16.gguf", 3),
        ]);
        assert_eq!(
            q,
            [
                RepoQuant {
                    file: "M-Q4_K_M.gguf".into(),
                    label: "Q4_K_M".into(),
                    size: 10
                },
                RepoQuant {
                    file: "BF16/M-BF16-00001-of-00002.gguf".into(),
                    label: "BF16".into(),
                    size: 50
                },
            ]
        );
    }
}
