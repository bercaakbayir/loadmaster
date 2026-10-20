//! `suggest`: models on the Hugging Face Hub that run on this machine. A live search rather
//! than a shipped catalog, so new models appear the day they are published (ADR 0014): the
//! Hub's listing narrows the field, file sizes rule out what can never fit, and each
//! remaining repo is planned from its header exactly as `plan hf://org/repo` does.

use std::process::ExitCode;
use std::sync::{Condvar, Mutex};

use loadmaster_hw::{Gpu, HwSnapshot, MemoryKind};
use loadmaster_model::ModelInfo;
use loadmaster_plan::{Error as PlanError, PlanRequest, explain};
use loadmaster_store::remote::{Error as HubError, Hub, HubModel, Search, Spec};
use serde::Serialize;

use super::plan::{best_fit, plan_repo_quants, repo_quants};
use crate::{SuggestArgs, SuggestKind, SuggestSort};

/// Known publishers, the only hand-kept data in `suggest` (it changes rarely; add an
/// organisation with a one-line PR). Quantizers that publish GGUFs of many models:
const QUANTIZERS: [&str; 4] = ["ggml-org", "unsloth", "bartowski", "lmstudio-community"];
/// Organisations that publish their own models (and often their own GGUF or AWQ builds).
const MAKERS: [&str; 36] = [
    "Qwen",
    "google",
    "meta-llama",
    "mistralai",
    "microsoft",
    "nvidia",
    "deepseek-ai",
    "openai",
    "LiquidAI",
    "openbmb",
    "ibm-granite",
    "JetBrains",
    "allenai",
    "HuggingFaceTB",
    "zai-org",
    "THUDM",
    "moonshotai",
    "MiniMaxAI",
    "tencent",
    "baidu",
    "XiaomiMiMo",
    "CohereLabs",
    "ByteDance-Seed",
    "inclusionAI",
    "stepfun-ai",
    "internlm",
    "OpenGVLab",
    "apple",
    "nomic-ai",
    "BAAI",
    "sentence-transformers",
    "intfloat",
    "mixedbread-ai",
    "Snowflake",
    "jinaai",
    "Alibaba-NLP",
];

/// Below this a GGUF cannot hold the whole model (the smallest llama.cpp quants are
/// about 1.6 bits per weight).
const MIN_BITS_PER_WEIGHT: f64 = 1.2;

/// Repos planned at once: each is a tree listing and one header fetch.
const WORKERS: usize = 6;

#[derive(Debug, Clone, Serialize)]
struct Suggestion {
    repo: String,
    params: Option<u64>,
    /// `chat`, `vision` or `embed`.
    kind: &'static str,
    quant: String,
    /// The GGUF file, for llama.cpp.
    #[serde(skip_serializing_if = "Option::is_none")]
    file: Option<String>,
    size_bytes: u64,
    /// `gpu`, `offload` or `no`.
    fit: &'static str,
    ctx: u32,
    gpu_bytes: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    short_by: Option<u64>,
    /// False when this quant was sized from another quant's header.
    exact: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tok_s_x10: Option<(u32, u32)>,
    /// The command that plans it exactly.
    plan: String,
    /// Position in the Hub's order (0 = most trending or downloaded).
    #[serde(skip)]
    rank: usize,
}

#[derive(Debug, Default, Serialize)]
struct Skipped {
    too_large: usize,
    duplicates: usize,
    /// Not by a known quantizer or the model's maker (shown with `--any-publisher`).
    other_publishers: usize,
    gated: usize,
    wrong_kind: usize,
    failed: usize,
    /// Smaller than `limit` models already found to fit on the GPU, so never planned.
    not_needed: usize,
}

/// Which repo a worker plans next, and when to stop.
#[derive(Default)]
struct Sched {
    next: usize,
    in_flight: usize,
    on_gpu: usize,
    done: usize,
    stop: bool,
}

/// What happened to one repo.
enum Outcome {
    Planned(Box<Suggestion>),
    TooLarge,
    Gated,
    WrongKind,
    Failed,
    RateLimited,
}

pub fn run(a: SuggestArgs) -> anyhow::Result<ExitCode> {
    let runtime: loadmaster_plan::Runtime = a.runtime.into();
    if runtime == loadmaster_plan::Runtime::Exl3 {
        anyhow::bail!(
            "suggest searches GGUF (llama.cpp) and safetensors (vLLM) repos; exl3 is not supported yet"
        );
    }
    let gguf = runtime == loadmaster_plan::Runtime::LlamaCpp;
    let snap = super::snapshot(&a.hw)?;
    let (gpu, extra_gpus) = super::pick_gpus(&snap, &a.gpu)?;
    let gpus: Vec<&Gpu> = std::iter::once(&gpu).chain(&extra_gpus).collect();
    let hw = explain::hw_label(&gpus);
    let usable: u64 = gpus.iter().map(|g| g.usable_bytes).sum();
    // What a GGUF can use at most: the GPUs plus host RAM for offloaded layers; on unified
    // memory a plan must fit in the RAM free now (ADR 0007).
    let reach = match gpu.memory {
        MemoryKind::Unified => usable.min(snap.host.available_bytes),
        MemoryKind::Dedicated if gguf => usable + snap.host.available_bytes,
        MemoryKind::Dedicated => usable,
    };
    let req = PlanRequest {
        ctx: a.ctx,
        runtime,
        extra_gpus: extra_gpus.clone(),
        ..PlanRequest::default()
    };

    let hub = super::hub(a.offline);
    let found = search(&hub, &a, gguf)?;
    let mut skipped = Skipped::default();
    let (candidates, searched) = select(
        found,
        a.any_publisher,
        hub.has_token(),
        a.candidates as usize,
        &mut skipped,
    );

    // Plan the biggest models first and stop once `limit` fit entirely on the GPU: nothing
    // smaller can outrank them, and each plan costs a header download (often 5-15 MB).
    let mut order: Vec<usize> = (0..candidates.len()).collect();
    order.sort_by_key(|&i| (std::cmp::Reverse(candidates[i].params.unwrap_or(0)), i));
    // At most `limit` repos confirmed on the GPU or in flight, so no header is fetched for
    // a model that cannot make the table.
    let sched = Mutex::new(Sched::default());
    let ready = Condvar::new();
    let outcomes: Mutex<Vec<Outcome>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..WORKERS.min(candidates.len()) {
            s.spawn(|| {
                loop {
                    let i = {
                        let mut g = sched.lock().unwrap();
                        loop {
                            if g.stop || g.next >= order.len() || g.on_gpu >= a.limit {
                                ready.notify_all();
                                break None;
                            }
                            if g.on_gpu + g.in_flight < a.limit {
                                g.next += 1;
                                g.in_flight += 1;
                                break Some(order[g.next - 1]);
                            }
                            g = ready.wait(g).unwrap();
                        }
                    };
                    let Some(i) = i else { break };
                    let job = Job {
                        hub: &hub,
                        snap: &snap,
                        gpu: &gpu,
                        req: &req,
                        reach,
                        usable,
                        kind: a.kind,
                    };
                    let o = job.run(&candidates[i], i, gguf);
                    let mut g = sched.lock().unwrap();
                    g.in_flight -= 1;
                    g.done += 1;
                    match &o {
                        Outcome::RateLimited => g.stop = true,
                        Outcome::Planned(s) if s.fit == "gpu" => g.on_gpu += 1,
                        _ => {}
                    }
                    outcomes.lock().unwrap().push(o);
                    ready.notify_all();
                }
            });
        }
    });

    skipped.not_needed = candidates.len() - sched.into_inner().unwrap().done;
    let mut results = Vec::new();
    let mut rate_limited = false;
    for o in outcomes.into_inner().unwrap() {
        match o {
            Outcome::Planned(s) if s.fit == "no" && !a.all => skipped.too_large += 1,
            Outcome::Planned(s) => results.push(*s),
            Outcome::TooLarge => skipped.too_large += 1,
            Outcome::Gated => skipped.gated += 1,
            Outcome::WrongKind => skipped.wrong_kind += 1,
            Outcome::Failed => skipped.failed += 1,
            Outcome::RateLimited => rate_limited = true,
        }
    }
    rank(&mut results);
    results.truncate(a.limit);
    let any_fit = results.iter().any(|r| r.fit != "no");

    if a.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "hardware": hw,
                "runtime": runtime.name(),
                "ctx": a.ctx,
                "query": a.query,
                "searched": searched,
                "rate_limited": rate_limited,
                "results": results,
                "skipped": skipped,
            }))?
        );
    } else {
        crate::ui::print(&render(
            &a,
            runtime,
            &hw,
            usable,
            searched,
            &results,
            &skipped,
            rate_limited,
        ));
    }
    Ok(ExitCode::from(if any_fit { 0 } else { 1 }))
}

/// The Hub searches for these arguments, merged. `chat` adds no pipeline filter for GGUF
/// (many GGUF repos state none; the header decides), `text-generation` for safetensors.
fn search(hub: &Hub, a: &SuggestArgs, gguf: bool) -> anyhow::Result<Vec<HubModel>> {
    let format = if gguf { "gguf" } else { "safetensors" };
    let pipelines: Vec<Option<&str>> = match a.kind {
        SuggestKind::Chat if gguf => vec![None],
        SuggestKind::Chat => vec![Some("text-generation")],
        SuggestKind::Vision => vec![Some("image-text-to-text")],
        SuggestKind::Embed => vec![Some("feature-extraction"), Some("sentence-similarity")],
    };
    let sort = match a.sort {
        SuggestSort::Trending => "trendingScore",
        SuggestSort::Downloads => "downloads",
        SuggestSort::New => "createdAt",
    };
    let mut out: Vec<HubModel> = Vec::new();
    for p in pipelines {
        let q = Search {
            filters: std::iter::once(format.to_owned())
                .chain(p.map(str::to_owned))
                .collect(),
            text: a.query.clone(),
            sort,
            // The publisher filter drops many; ask for more than will be planned.
            limit: if a.any_publisher {
                a.candidates
            } else {
                (a.candidates * 3).min(300)
            },
        };
        for m in hub.search(&q)? {
            if !out.iter().any(|o| o.id == m.id) {
                out.push(m);
            }
        }
    }
    Ok(out)
}

fn author(repo: &str) -> &str {
    repo.split('/').next().unwrap_or(repo)
}

/// A known publisher: a quantizer or a model maker from the lists above. (Being the base
/// model's own organisation is not enough: fine-tuners publish GGUFs of their own
/// fine-tunes.)
fn preferred(m: &HubModel) -> bool {
    let by = author(&m.id);
    QUANTIZERS.contains(&by) || MAKERS.iter().any(|o| o.eq_ignore_ascii_case(by))
}

/// The repos to plan, in the Hub's order, until `limit` are kept: gated ones need a token,
/// only known publishers unless `any_publisher`, and one repo per base model (a later repo
/// by a preferred publisher replaces an earlier one that is not). Returns them and how many
/// listings were examined; each skip is counted in `skipped`.
fn select(
    found: Vec<HubModel>,
    any_publisher: bool,
    has_token: bool,
    limit: usize,
    skipped: &mut Skipped,
) -> (Vec<HubModel>, usize) {
    let key = |m: &HubModel| m.base.clone().unwrap_or_else(|| m.id.clone());
    let mut kept: Vec<HubModel> = Vec::new();
    let mut examined = 0;
    for m in found {
        if kept.len() >= limit {
            break;
        }
        examined += 1;
        if m.gated && !has_token {
            skipped.gated += 1;
        } else if !any_publisher && !preferred(&m) {
            skipped.other_publishers += 1;
        } else if let Some(k) = kept.iter_mut().find(|k| key(k) == key(&m)) {
            skipped.duplicates += 1;
            if preferred(&m) && !preferred(k) {
                *k = m;
            }
        } else {
            kept.push(m);
        }
    }
    (kept, examined)
}

/// Bytes of safetensors weights from the Hub's per-dtype parameter counts.
fn safetensors_bytes(m: &HubModel) -> Option<u64> {
    if m.dtypes.is_empty() {
        return m.params.map(|p| p * 2);
    }
    Some(
        m.dtypes
            .iter()
            .map(|(dtype, n)| {
                n * match dtype.as_str() {
                    "F64" | "I64" | "U64" => 8,
                    "F32" | "I32" | "U32" => 4,
                    "F16" | "BF16" | "I16" | "U16" => 2,
                    _ => 1,
                }
            })
            .sum(),
    )
}

/// Fits entirely on the GPU, then with offload, then not at all; within each, the largest
/// model first (a rough proxy for quality), then the Hub's order.
fn rank(results: &mut [Suggestion]) {
    let class = |fit: &str| match fit {
        "gpu" => 0,
        "offload" => 1,
        _ => 2,
    };
    results.sort_by(|a, b| {
        class(a.fit)
            .cmp(&class(b.fit))
            .then(b.params.unwrap_or(0).cmp(&a.params.unwrap_or(0)))
            .then(a.rank.cmp(&b.rank))
    });
}

fn kind_of(m: &ModelInfo) -> &'static str {
    if m.is_embedding() {
        "embed"
    } else if m.mmproj.is_some() {
        "vision"
    } else {
        "chat"
    }
}

fn kind_matches(want: SuggestKind, got: &str) -> bool {
    match want {
        SuggestKind::Chat => got != "embed",
        SuggestKind::Vision => got == "vision",
        SuggestKind::Embed => got == "embed",
    }
}

struct Job<'a> {
    hub: &'a Hub,
    snap: &'a HwSnapshot,
    gpu: &'a Gpu,
    req: &'a PlanRequest,
    reach: u64,
    usable: u64,
    kind: SuggestKind,
}

impl Job<'_> {
    fn run(&self, m: &HubModel, rank: usize, gguf: bool) -> Outcome {
        let r = if gguf {
            self.gguf(m, rank)
        } else {
            self.safetensors(m, rank)
        };
        r.unwrap_or_else(|e| {
            tracing::debug!("suggest: {}: {e:#}", m.id);
            match e.downcast_ref::<HubError>() {
                Some(HubError::RateLimited(_)) => Outcome::RateLimited,
                Some(HubError::Gated(_)) => Outcome::Gated,
                _ => Outcome::Failed,
            }
        })
    }

    fn gguf(&self, m: &HubModel, rank: usize) -> anyhow::Result<Outcome> {
        let spec = Spec {
            repo: m.id.clone(),
            rev: "main".into(),
            file: None,
        };
        let files = self.hub.list(&spec.repo, &spec.rev)?;
        let quants = repo_quants(&files);
        let Some(smallest) = quants.first() else {
            return Ok(Outcome::Failed);
        };
        // Weights alone past everything the machine has: no header needed to rule it out.
        if smallest.size > self.reach {
            return Ok(Outcome::TooLarge);
        }
        let mut rp = plan_repo_quants(self.hub, &spec, &files, self.snap, self.gpu, self.req)?;
        // A "quant" far below any real one is a stray file, not the model.
        if let Some(params) = m.params.filter(|&p| p > 0) {
            rp.rows
                .retain(|r| r.size_bytes as f64 * 8.0 / params as f64 >= MIN_BITS_PER_WEIGHT);
            if rp.rows.is_empty() {
                return Ok(Outcome::Failed);
            }
        }
        let kind = kind_of(&rp.model);
        if !kind_matches(self.kind, kind) {
            return Ok(Outcome::WrongKind);
        }
        let i = best_fit(&rp.rows).unwrap_or(0);
        let row = &rp.rows[i];
        Ok(Outcome::Planned(Box::new(Suggestion {
            repo: m.id.clone(),
            params: m.params,
            kind,
            quant: row.quant.clone(),
            file: Some(row.file.clone()),
            size_bytes: row.size_bytes,
            fit: row.fit,
            ctx: row.ctx,
            gpu_bytes: row.gpu_bytes,
            short_by: row.short_by,
            exact: row.exact,
            tok_s_x10: row.tok_s_x10,
            plan: format!("loadmaster plan hf://{}/{}", m.id, row.file),
            rank,
        })))
    }

    fn safetensors(&self, m: &HubModel, rank: usize) -> anyhow::Result<Outcome> {
        // vLLM keeps every weight on the GPUs.
        if safetensors_bytes(m).is_some_and(|b| b > self.usable) {
            return Ok(Outcome::TooLarge);
        }
        let files = self.hub.list(&m.id, "main")?;
        if !files.iter().any(|f| f.path == "config.json") {
            return Ok(Outcome::Failed);
        }
        let model = self.hub.open_safetensors(&m.id, "main", &files)?;
        let kind = kind_of(&model);
        if !kind_matches(self.kind, kind) {
            return Ok(Outcome::WrongKind);
        }
        let s = |fit, ctx, gpu_bytes, short_by| Suggestion {
            repo: m.id.clone(),
            params: m.params,
            kind,
            quant: model.quant.to_string(),
            file: None,
            size_bytes: model.weight_bytes,
            fit,
            ctx,
            gpu_bytes,
            short_by,
            exact: true,
            tok_s_x10: None,
            plan: format!("loadmaster plan hf://{} --runtime vllm", m.id),
            rank,
        };
        Ok(Outcome::Planned(Box::new(
            match loadmaster_plan::plan(&model, self.snap, self.gpu, self.req) {
                Ok(p) => s("gpu", p.ctx, p.footprint.gpu_bytes, None),
                Err(PlanError::CannotFit(n)) => {
                    s("no", n.ctx, n.closest.gpu_bytes, Some(n.short_by))
                }
                Err(e) => return Err(e.into()),
            },
        )))
    }
}

/// `30.5B`, `596M`.
fn params_text(p: Option<u64>) -> String {
    match p {
        Some(p) if p >= 1_000_000_000 => format!("{:.1}B", p as f64 / 1e9),
        Some(p) => format!("{}M", p / 1_000_000),
        None => "-".into(),
    }
}

#[allow(clippy::too_many_arguments)]
fn render(
    a: &SuggestArgs,
    runtime: loadmaster_plan::Runtime,
    hw: &str,
    usable: u64,
    searched: usize,
    results: &[Suggestion],
    skipped: &Skipped,
    rate_limited: bool,
) -> String {
    let style = crate::ui::style();
    let mut out = String::new();
    let source = match a.sort {
        SuggestSort::Trending => "trending",
        SuggestSort::Downloads => "most downloaded",
        SuggestSort::New => "newest",
    };
    let format = if runtime == loadmaster_plan::Runtime::LlamaCpp {
        "GGUF"
    } else {
        "safetensors"
    };
    out.push_str(&format!(
        "{}  ·  {}  ·  {}  ·  {source} {format} on Hugging Face{}\n\n",
        style.bold(&format!(
            "Models for {hw}, {} usable",
            explain::gib1(usable)
        )),
        runtime.name(),
        a.ctx
            .map_or("largest context that fits".to_owned(), |c| format!(
                "ctx {c}"
            )),
        a.query
            .as_deref()
            .map_or(String::new(), |q| format!(" matching \"{q}\""))
    ));
    if results.is_empty() {
        out.push_str(&format!(
            "nothing that runs here among {searched} repos; try a search term, --sort downloads, a smaller --ctx, or --all\n"
        ));
    } else {
        let fit_cell = |r: &Suggestion| {
            let est = if r.exact { "" } else { "~" };
            match r.short_by {
                Some(s) => format!("✗ {} short", explain::gib1(s)),
                None => format!("{est}{}", r.fit),
            }
        };
        let tok_s = |r: &Suggestion| {
            r.tok_s_x10.map_or("-".to_owned(), |t| {
                format!(
                    "~{}",
                    loadmaster_plan::speed::band_text(t).trim_end_matches(" tok/s")
                )
            })
        };
        let name_w = results
            .iter()
            .map(|r| r.repo.chars().count())
            .max()
            .unwrap_or(5)
            .clamp(5, 48);
        let qw = results
            .iter()
            .map(|r| r.quant.len())
            .max()
            .unwrap_or(5)
            .max(5);
        let fw = results
            .iter()
            .map(|r| fit_cell(r).chars().count())
            .max()
            .unwrap_or(3)
            .max(3);
        let speed = results.iter().any(|r| r.tok_s_x10.is_some());
        let line = |cells: [&str; 7]| {
            let mut l = format!(
                "{:<name_w$}  {:>6}  {:<qw$}  {:>9}  {:<fw$}  {:>6}",
                cells[0], cells[1], cells[2], cells[3], cells[4], cells[5]
            );
            if speed {
                l.push_str(&format!("  {:>9}", cells[6]));
            }
            l.trim_end().to_owned()
        };
        out.push_str(&style.bold(&line([
            "MODEL", "PARAMS", "QUANT", "SIZE", "FIT", "CTX", "TOK/S",
        ])));
        out.push('\n');
        for r in results {
            let name = if r.repo.chars().count() > name_w {
                let keep: String = r.repo.chars().take(name_w - 1).collect();
                format!("{keep}…")
            } else {
                r.repo.clone()
            };
            let row = line([
                &name,
                &params_text(r.params),
                &r.quant,
                &explain::gib1(r.size_bytes),
                &fit_cell(r),
                &if r.fit == "no" {
                    "-".to_owned()
                } else {
                    r.ctx.to_string()
                },
                &tok_s(r),
            ]);
            out.push_str(&match r.fit {
                "no" => style.dim(&row),
                _ => row,
            });
            out.push('\n');
        }
    }
    out.push('\n');
    let mut counts = vec![format!("searched {searched} repos")];
    for (n, what) in [
        (skipped.too_large, "too large"),
        (skipped.duplicates, "duplicates"),
        (
            skipped.other_publishers,
            "by other publishers (--any-publisher)",
        ),
        (skipped.wrong_kind, "other kinds"),
        (skipped.failed, "unreadable"),
        (skipped.not_needed, "smaller, not needed"),
    ] {
        if n > 0 {
            counts.push(format!("{n} {what}"));
        }
    }
    if skipped.gated > 0 {
        counts.push(format!("{} gated (set HF_TOKEN)", skipped.gated));
    }
    out.push_str(&style.dim(&counts.join(" · ")));
    out.push('\n');
    if rate_limited {
        out.push_str(&format!(
            "{} stopped early: Hugging Face rate limit; set HF_TOKEN for a higher limit, or retry in a few minutes\n",
            style.yellow("!")
        ));
    }
    let fitting = results.iter().filter(|r| r.fit != "no").count();
    if fitting > 0 && fitting < a.limit && !rate_limited {
        out.push_str(&style.dim(&format!(
            "only {fitting} of the {} repos considered run here; a search term (`loadmaster suggest 3b`) or --candidates 100 looks further",
            a.candidates
        )));
        out.push('\n');
    }
    if let Some(top) = results.iter().find(|r| r.fit != "no") {
        out.push_str(&format!("exact plan: {}\n", top.plan));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hub_model(id: &str, base: Option<&str>) -> HubModel {
        HubModel {
            id: id.into(),
            downloads: 0,
            trending: 0.0,
            created: None,
            gated: false,
            pipeline: None,
            tags: Vec::new(),
            base: base.map(str::to_owned),
            params: None,
            arch: None,
            dtypes: Vec::new(),
        }
    }

    #[test]
    fn select_keeps_known_publishers_once_per_base_model() {
        let found = || {
            vec![
                hub_model("someone/Qwen3-8B-GGUF", Some("Qwen/Qwen3-8B")),
                hub_model("bartowski/Qwen3-8B-GGUF", Some("Qwen/Qwen3-8B")),
                hub_model("unsloth/Qwen3-8B-GGUF", Some("Qwen/Qwen3-8B")),
                hub_model(
                    "google/gemma-3-4b-it-qat-q4_0-gguf",
                    Some("google/gemma-3-4b-it"),
                ),
                hub_model("x/standalone-GGUF", None),
            ]
        };
        let ids = |v: &[HubModel]| v.iter().map(|m| m.id.clone()).collect::<Vec<_>>();
        let mut sk = Skipped::default();
        let (kept, examined) = select(found(), false, false, 10, &mut sk);
        assert_eq!(
            ids(&kept),
            [
                "bartowski/Qwen3-8B-GGUF",
                "google/gemma-3-4b-it-qat-q4_0-gguf"
            ],
            "a model maker counts as a known publisher; an unknown original upload does not"
        );
        assert_eq!((examined, sk.other_publishers, sk.duplicates), (5, 2, 1));

        // --any-publisher: the first per base, replaced by a known quantizer's.
        let mut sk = Skipped::default();
        let (kept, _) = select(found(), true, false, 10, &mut sk);
        assert_eq!(
            ids(&kept),
            [
                "bartowski/Qwen3-8B-GGUF",
                "google/gemma-3-4b-it-qat-q4_0-gguf",
                "x/standalone-GGUF"
            ]
        );
        assert_eq!(sk.duplicates, 2);

        // Stops once `limit` are kept.
        let mut sk = Skipped::default();
        let (kept, examined) = select(found(), true, false, 1, &mut sk);
        assert_eq!((kept.len(), examined), (1, 1));
    }

    fn suggestion(repo: &str, fit: &'static str, params: u64, rank: usize) -> Suggestion {
        Suggestion {
            repo: repo.into(),
            params: Some(params),
            kind: "chat",
            quant: "Q4_K_M".into(),
            file: None,
            size_bytes: 0,
            fit,
            ctx: 4096,
            gpu_bytes: 0,
            short_by: None,
            exact: true,
            tok_s_x10: None,
            plan: String::new(),
            rank,
        }
    }

    #[test]
    fn ranking_is_gpu_then_offload_then_largest_then_popular() {
        let mut r = vec![
            suggestion("small-gpu", "gpu", 4_000_000_000, 0),
            suggestion("huge-offload", "offload", 70_000_000_000, 1),
            suggestion("big-gpu", "gpu", 14_000_000_000, 2),
            suggestion("big-gpu-less-popular", "gpu", 14_000_000_000, 3),
            suggestion("no", "no", 1_000_000_000, 4),
        ];
        rank(&mut r);
        let order: Vec<&str> = r.iter().map(|s| s.repo.as_str()).collect();
        assert_eq!(
            order,
            [
                "big-gpu",
                "big-gpu-less-popular",
                "small-gpu",
                "huge-offload",
                "no"
            ]
        );
    }

    #[test]
    fn safetensors_size_comes_from_dtype_counts() {
        let mut m = hub_model("Qwen/Qwen3-8B-AWQ", None);
        m.dtypes = vec![
            ("I32".into(), 1_000),
            ("F16".into(), 500),
            ("F8_E4M3".into(), 10),
        ];
        assert_eq!(safetensors_bytes(&m), Some(4_000 + 1_000 + 10));
        m.dtypes.clear();
        m.params = Some(8);
        assert_eq!(safetensors_bytes(&m), Some(16), "BF16 assumed");
        assert_eq!(params_text(Some(30_532_000_000)), "30.5B");
        assert_eq!(params_text(Some(596_000_000)), "596M");
    }
}
