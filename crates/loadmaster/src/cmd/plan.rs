use std::process::ExitCode;

use anyhow::Context;
use loadmaster_plan::{Error, PlanRequest, explain, gib};
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
    let gpu =
        match args.gpu.as_str() {
            "auto" => snap.best_gpu(),
            q => Some(snap.find_gpu(q).with_context(|| {
                format!("no GPU matches `{q}`; run `loadmaster gpu` to list them")
            })?),
        }
        .ok_or(Error::NoGpu)?
        .clone();

    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args.runtime.into(),
        ..PlanRequest::default()
    };
    match loadmaster_plan::plan(&model, &snap, &gpu, &req) {
        Ok(p) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::Fits { plan: &p })?
                );
            } else if args.explain {
                print!("{}", explain::explain(&p));
            } else {
                print!("{}", explain::summary(&p));
            }
            Ok(ExitCode::SUCCESS)
        }
        Err(Error::CannotFit(n)) => {
            if args.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&JsonOut::CannotFit { no_fit: &n })?
                );
            } else {
                print!("{}", explain::no_fit(&n, args.explain));
            }
            Ok(ExitCode::from(1))
        }
        Err(e) => Err(e.into()),
    }
}

/// One quant in a Hub repo: its file (first part if split), label and total size.
#[derive(Debug, Clone, PartialEq, Eq)]
struct RepoQuant {
    file: String,
    label: String,
    size: u64,
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
            || ["BF16", "F16", "F32", "MXFP4"].contains(&u.as_str())
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

/// The GGUF quants in a repo listing, smallest first; split parts are summed into their
/// first part, projectors and non-GGUF files skipped.
fn repo_quants(files: &[loadmaster_store::remote::RemoteFile]) -> Vec<RepoQuant> {
    let mut quants: Vec<RepoQuant> = Vec::new();
    let projectors = loadmaster_model::mmproj_candidates(files.iter().map(|f| f.path.as_str()));
    for f in files {
        if !f.path.ends_with(".gguf") || projectors.contains(&f.path.as_str()) {
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

#[derive(Serialize)]
struct RepoRow {
    quant: String,
    file: String,
    size_bytes: u64,
    /// `gpu`, `offload` or `no`.
    fit: &'static str,
    ctx: u32,
    kv: Option<&'static str>,
    gpu_bytes: u64,
    /// Bytes over budget when it does not fit.
    short_by: Option<u64>,
    /// False when sized from the file size, not the file's own header.
    exact: bool,
    /// Estimated generation tok/s x10 at the first fill (4k or the context).
    #[serde(skip_serializing_if = "Option::is_none")]
    tok_s_x10: Option<(u32, u32)>,
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
        .unwrap_or(&quants[quants.len() / 2]);
    let hub = super::hub(args.offline);
    let model = hub.open_gguf(&spec.repo, &spec.rev, &base.file, files)?;
    let snap = super::snapshot(&args.hw)?;
    let gpu = super::pick_gpu(&snap, &args.gpu)?;
    let req = PlanRequest {
        ctx: args.ctx,
        runtime: args.runtime.into(),
        ..PlanRequest::default()
    };
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
            Ok(match loadmaster_plan::plan(&m, &snap, &gpu, &req) {
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
                        .and_then(|s| s.at.first())
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
        let fit_cell = |r: &RepoRow| match r.short_by {
            Some(s) => format!("no, {} short", gib(s)),
            None => r.fit.to_owned(),
        };
        let w = rows.iter().map(|r| r.quant.len()).max().unwrap_or(5).max(5);
        let fw = rows
            .iter()
            .map(|r| fit_cell(r).len())
            .max()
            .unwrap_or(3)
            .max(7);
        println!(
            "{} · {} quants · {} ({} usable)",
            model.name.as_deref().unwrap_or(&spec.repo),
            rows.len(),
            gpu.name,
            gib(gpu.usable_bytes)
        );
        let speed = rows.iter().any(|r| r.tok_s_x10.is_some());
        let tok_s = |r: &RepoRow| match (speed, r.tok_s_x10) {
            (false, _) => String::new(),
            (true, Some(t)) => format!("  {:>15}", loadmaster_plan::speed::band_text(t)),
            (true, None) => format!("  {:>15}", "-"),
        };
        println!(
            "{:<w$}  {:>10}  {:<fw$}  {:>7}  {:<5} {:>10}{}",
            "QUANT",
            "SIZE",
            "FIT",
            "CTX",
            "KV",
            "GPU",
            if speed {
                format!("  {:>15}", "TOK/S (4k)")
            } else {
                String::new()
            }
        );
        for r in &rows {
            println!(
                "{:<w$}  {:>10}  {:<fw$}  {:>7}  {:<5} {:>10}{}{}",
                r.quant,
                gib(r.size_bytes),
                fit_cell(r),
                if r.fit == "no" {
                    "-".into()
                } else {
                    r.ctx.to_string()
                },
                r.kv.unwrap_or("-"),
                gib(r.gpu_bytes),
                tok_s(r),
                if r.exact { "" } else { "  *" }
            );
        }
        println!(
            "\n* estimated: the {} header scaled to that file's size",
            base.label
        );
        // The largest quant that runs entirely on the GPU, else the smallest that runs.
        let (why, pick) = match rows.iter().rev().find(|r| r.fit == "gpu") {
            Some(r) => ("largest quant entirely on the GPU", r),
            None => match rows.iter().find(|r| r.fit != "no") {
                Some(r) => ("smallest quant, with CPU offload", r),
                None => ("smallest quant", &rows[0]),
            },
        };
        println!(
            "exact plan for the {why}:\n  loadmaster plan hf://{}/{}",
            spec.repo, pick.file
        );
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
