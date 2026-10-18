//! Footprint arithmetic, KV budget, layer-split search and runtime flag emitters.
//!
//! [`plan`] is a pure function of a [`ModelInfo`], a [`HwSnapshot`] and a [`PlanRequest`]:
//! no clock, randomness or environment, so the same inputs give byte-identical output.

pub mod emit;
pub mod explain;
pub mod footprint;

use loadmaster_hw::{Gpu, Host, HwSnapshot, MIB, Vendor};
use loadmaster_model::ModelInfo;
use serde::Serialize;

pub use footprint::{Config, Footprint, Headroom, KvType, Placement, Term, gib};

/// Default headroom on NVIDIA: driver reserve and CUDA context.
pub const HEADROOM_NVIDIA: u64 = 800 * MIB;
/// Default headroom on Apple Silicon: OS and display share the same memory.
pub const HEADROOM_APPLE: u64 = 1536 * MIB;
pub const DEFAULT_UBATCH: u32 = 512;
/// The smallest context the default-context search will pick.
pub const MIN_DEFAULT_CTX: u32 = 4096;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Runtime {
    LlamaCpp,
    Vllm,
    Exl3,
}

impl Runtime {
    pub fn name(self) -> &'static str {
        match self {
            Runtime::LlamaCpp => "llama.cpp",
            Runtime::Vllm => "vllm",
            Runtime::Exl3 => "exl3",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanRequest {
    /// Requested context; `None` picks the largest that fits fully on the GPU.
    pub ctx: Option<u32>,
    pub runtime: Runtime,
    /// Force a KV cache type instead of choosing one.
    pub kv: Option<KvType>,
    pub headroom: Option<u64>,
    pub n_ubatch: u32,
}

impl Default for PlanRequest {
    fn default() -> Self {
        PlanRequest {
            ctx: None,
            runtime: Runtime::LlamaCpp,
            kv: None,
            headroom: None,
            n_ubatch: DEFAULT_UBATCH,
        }
    }
}

/// One configuration the search evaluated, in order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Trial {
    pub label: String,
    pub gpu_bytes: u64,
    pub host_bytes: u64,
    pub fits: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ModelSummary {
    pub path: String,
    pub name: Option<String>,
    pub arch: String,
    pub quant: String,
    pub n_layer: u32,
    pub n_ctx_train: u32,
    pub weight_bytes: u64,
    pub mmproj: Option<String>,
    pub moe: bool,
    pub chat_template_blake3: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Plan {
    pub model: ModelSummary,
    pub gpu: Gpu,
    pub runtime: Runtime,
    pub ctx: u32,
    /// Why this context was used.
    pub ctx_reason: String,
    pub placement: Placement,
    pub kv: KvType,
    pub kv_reason: String,
    pub flash_attn: bool,
    pub n_ubatch: u32,
    pub footprint: Footprint,
    /// GPU bytes the footprint must fit in (`Gpu::usable_bytes`).
    pub budget: u64,
    /// Host bytes CPU-side tensors must fit in.
    pub host_budget: u64,
    pub trials: Vec<Trial>,
    pub warnings: Vec<String>,
    /// The launch command as argv.
    pub argv: Vec<String>,
}

/// A suggested change that would make the model fit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Fix {
    pub summary: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct NoFit {
    pub model: ModelSummary,
    pub gpu: Gpu,
    pub ctx: u32,
    /// `gpu` or `host`: which budget is exceeded by the closest configuration.
    pub limit: &'static str,
    pub short_by: u64,
    /// The closest configuration tried.
    pub closest: Footprint,
    pub trials: Vec<Trial>,
    pub fixes: Vec<Fix>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{} does not fit: {} short on {}", .0.model.path, gib(.0.short_by), .0.limit)]
    CannotFit(Box<NoFit>),
    #[error("planning for {} is not implemented yet; use --runtime llama.cpp", .0.name())]
    RuntimeNotSupported(Runtime),
    #[error(
        "{kv} KV cache does not work for {model}: KV row width is not a multiple of 32; use f16"
    )]
    KvTypeInvalid { kv: &'static str, model: String },
    #[error(
        "no GPU found; CPU-only planning is not supported yet. Run `loadmaster gpu` to see what was detected"
    )]
    NoGpu,
    #[error("context must be at least 1 token")]
    ZeroContext,
}

struct Search<'a> {
    model: &'a ModelInfo,
    gpu: &'a Gpu,
    host_budget: u64,
    host_total: u64,
    headroom: Headroom,
    n_ubatch: u32,
}

/// A configuration that was evaluated, with its footprint.
struct Candidate {
    placement: Placement,
    cfg: Config,
    fp: Footprint,
}

impl Search<'_> {
    fn config(&self, ctx: u32, kv: KvType) -> Config {
        Config {
            ctx,
            n_seq: 1,
            n_ubatch: self.n_ubatch,
            kv,
            // Always on: required for a quantised V cache, and the compute buffer shrinks.
            flash_attn: true,
            headroom: self.headroom.clone(),
        }
    }

    fn eval(&self, placement: Placement, ctx: u32, kv: KvType) -> Option<Candidate> {
        let cfg = self.config(ctx, kv);
        let fp = footprint::footprint(self.model, placement, &cfg)?;
        Some(Candidate { placement, cfg, fp })
    }

    fn fits(&self, c: &Candidate) -> bool {
        c.fp.gpu_bytes <= self.gpu.usable_bytes && c.fp.host_bytes <= self.host_budget
    }

    fn trial(&self, label: String, c: &Candidate) -> Trial {
        Trial {
            label,
            gpu_bytes: c.fp.gpu_bytes,
            host_bytes: c.fp.host_bytes,
            fits: self.fits(c),
        }
    }

    fn full_fits(&self, ctx: u32, kv: KvType) -> bool {
        self.eval(Placement::full(self.model.n_layer), ctx, kv)
            .is_some_and(|c| self.fits(&c))
    }

    /// Largest context (a multiple of 256) at which the model fits fully on the GPU with
    /// q8_0 or f16 KV, if any.
    fn max_full_ctx(&self, upper: u32) -> Option<u32> {
        let kv = [KvType::Q8_0, KvType::F16]
            .into_iter()
            .find(|&kv| footprint::kv_type_valid(self.model, kv))?;
        let step = footprint::KV_PAD;
        if !self.full_fits(step, kv) {
            return None;
        }
        let (mut lo, mut hi) = (1u32, upper.max(step) / step);
        // Footprint grows with ctx, so binary search on multiples of 256.
        while lo < hi {
            let mid = lo + (hi - lo).div_ceil(2);
            if self.full_fits(mid * step, kv) {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        Some(lo * step)
    }
}

pub fn headroom_for(gpu: &Gpu, overridden: Option<u64>) -> Headroom {
    match overridden {
        Some(bytes) => Headroom {
            bytes,
            basis: "user override".into(),
        },
        None => match gpu.vendor {
            Vendor::Nvidia => Headroom {
                bytes: HEADROOM_NVIDIA,
                basis: "NVIDIA driver and CUDA context (default 800 MiB)".into(),
            },
            Vendor::Apple => Headroom {
                bytes: HEADROOM_APPLE,
                basis: "macOS and display on unified memory (default 1536 MiB)".into(),
            },
        },
    }
}

fn host_budget(host: &Host) -> u64 {
    if host.available_bytes > 0 {
        host.available_bytes
    } else {
        host.total_bytes
    }
}

fn summary(model: &ModelInfo) -> ModelSummary {
    ModelSummary {
        path: model.path.display().to_string(),
        name: model.name.clone(),
        arch: model.arch.clone(),
        quant: model.quant.to_string(),
        n_layer: model.n_layer,
        n_ctx_train: model.n_ctx_train,
        weight_bytes: model.weight_bytes,
        mmproj: model.mmproj.as_ref().map(|m| m.path.display().to_string()),
        moe: model.is_moe(),
        chat_template_blake3: model.chat_template.as_ref().map(|t| t.blake3.clone()),
    }
}

/// Plan `model` on `gpu` (one of `hw.gpus`).
pub fn plan(
    model: &ModelInfo,
    hw: &HwSnapshot,
    gpu: &Gpu,
    req: &PlanRequest,
) -> Result<Plan, Error> {
    if req.runtime != Runtime::LlamaCpp {
        return Err(Error::RuntimeNotSupported(req.runtime));
    }
    if req.ctx == Some(0) {
        return Err(Error::ZeroContext);
    }
    if let Some(kv) = req.kv
        && !footprint::kv_type_valid(model, kv)
    {
        return Err(Error::KvTypeInvalid {
            kv: kv.flag(),
            model: model.path.display().to_string(),
        });
    }
    let s = Search {
        model,
        gpu,
        host_budget: host_budget(&hw.host),
        host_total: hw.host.total_bytes,
        headroom: headroom_for(gpu, req.headroom),
        n_ubatch: req.n_ubatch,
    };
    let mut warnings = Vec::new();
    if !model.arch_known {
        warnings.push(format!(
            "architecture `{}` is not in loadmaster's table; assumed dense full attention",
            model.arch
        ));
    }
    if !model.complete {
        warnings.push("model file is shorter than its tensor table (incomplete download?)".into());
    }

    let kv_order: Vec<KvType> = match req.kv {
        Some(kv) => vec![kv],
        None => KvType::ALL
            .into_iter()
            .filter(|&kv| footprint::kv_type_valid(model, kv))
            .collect(),
    };

    let (ctx, ctx_reason) = match req.ctx {
        Some(ctx) => {
            if ctx > model.n_ctx_train {
                warnings.push(format!(
                    "ctx {ctx} exceeds the training context {}; quality degrades without RoPE scaling",
                    model.n_ctx_train
                ));
            }
            (ctx, "requested with --ctx".to_owned())
        }
        None => default_ctx(&s, &kv_order),
    };

    let n_layer = model.n_layer;
    let mut st = State::default();

    // 1. Everything on the GPU, best KV type first.
    for &kv in &kv_order {
        if let Some(c) = s.eval(Placement::full(n_layer), ctx, kv)
            && st.consider(&s, format!("all layers on GPU, {} KV", kv.flag()), c)
        {
            break;
        }
    }

    // 2. Offload with q8_0 KV (or the forced type): routed experts first for MoE, since
    //    they dominate bytes but not latency, then whole layers.
    if st.chosen.is_none() {
        let kv = match req.kv {
            Some(kv) => kv,
            None if kv_order.contains(&KvType::Q8_0) => KvType::Q8_0,
            None => kv_order[0],
        };
        let full = Placement::full(n_layer);
        if model.is_moe() {
            let moe = (1..=n_layer).find_map(|k| {
                s.eval(
                    Placement {
                        n_cpu_moe: k,
                        ..full
                    },
                    ctx,
                    kv,
                )
                .filter(|c| s.fits(c))
            });
            let c = moe.or_else(|| {
                s.eval(
                    Placement {
                        n_cpu_moe: n_layer,
                        ..full
                    },
                    ctx,
                    kv,
                )
            });
            if let Some(c) = c {
                let k = c.placement.n_cpu_moe;
                st.consider(
                    &s,
                    format!(
                        "experts of first {k}/{n_layer} layers on CPU, {} KV",
                        kv.flag()
                    ),
                    c,
                );
            }
        }
        if st.chosen.is_none() {
            let n_cpu_moe = if model.is_moe() { n_layer } else { 0 };
            let at = |ngl| {
                s.eval(
                    Placement {
                        n_gpu_layers: ngl,
                        n_cpu_moe,
                    },
                    ctx,
                    kv,
                )
            };
            let c = (0..=n_layer)
                .rev()
                .find_map(|ngl| at(ngl).filter(|c| s.fits(c)))
                .or_else(|| at(0));
            if let Some(c) = c {
                let ngl = c.placement.n_gpu_layers;
                st.consider(
                    &s,
                    format!("{ngl}/{n_layer} layers on GPU, {} KV", kv.flag()),
                    c,
                );
            }
        }
    }

    let State {
        trials,
        chosen,
        closest,
    } = st;
    let Some(c) = chosen else {
        let closest = closest.expect("at least one configuration is evaluated");
        return Err(no_fit(&s, model, ctx, closest, trials));
    };

    let kv_reason = kv_reason(req.kv, &c, n_layer, &kv_order);
    if c.cfg.kv == KvType::Q4_0 && req.kv.is_none() {
        warnings.push(
            "q4_0 KV cache measurably lowers output quality; consider a smaller --ctx or a smaller quant".into(),
        );
    }
    let mut plan = Plan {
        model: summary(model),
        gpu: gpu.clone(),
        runtime: req.runtime,
        ctx,
        ctx_reason,
        placement: c.placement,
        kv: c.cfg.kv,
        kv_reason,
        flash_attn: c.cfg.flash_attn,
        n_ubatch: c.cfg.n_ubatch,
        footprint: c.fp,
        budget: gpu.usable_bytes,
        host_budget: s.host_budget,
        trials,
        warnings,
        argv: Vec::new(),
    };
    plan.argv = emit::llama_cpp::argv(&plan);
    Ok(plan)
}

#[derive(Default)]
struct State {
    trials: Vec<Trial>,
    chosen: Option<Candidate>,
    closest: Option<Candidate>,
}

impl State {
    /// Record a trial; keep it as the choice if it fits, else as the closest miss.
    fn consider(&mut self, s: &Search<'_>, label: String, c: Candidate) -> bool {
        let t = s.trial(label, &c);
        let fits = t.fits;
        self.trials.push(t);
        if fits {
            self.chosen = Some(c);
        } else if self
            .closest
            .as_ref()
            .is_none_or(|b| overshoot(s, &c) < overshoot(s, b))
        {
            self.closest = Some(c);
        }
        fits
    }
}

fn overshoot(s: &Search<'_>, c: &Candidate) -> u64 {
    c.fp.gpu_bytes.saturating_sub(s.gpu.usable_bytes)
        + c.fp.host_bytes.saturating_sub(s.host_budget)
}

fn default_ctx(s: &Search<'_>, kv_order: &[KvType]) -> (u32, String) {
    let train = s.model.n_ctx_train.max(1);
    let mut candidates = vec![train];
    let mut p = (train - 1).checked_ilog2().map_or(1, |e| 1u32 << e);
    while p >= MIN_DEFAULT_CTX {
        candidates.push(p);
        p /= 2;
    }
    let first_two: Vec<KvType> = kv_order
        .iter()
        .copied()
        .filter(|kv| *kv != KvType::Q4_0)
        .collect();
    for &ctx in &candidates {
        if let Some(kv) = first_two.iter().find(|&&kv| s.full_fits(ctx, kv)) {
            let reason = if ctx == train {
                format!(
                    "not given; the training context {train} fits on GPU with {} KV",
                    kv.flag()
                )
            } else {
                format!(
                    "not given; largest power of two below the training context {train} that fits on GPU with {} KV",
                    kv.flag()
                )
            };
            return (ctx, reason);
        }
    }
    let ctx = train.min(MIN_DEFAULT_CTX);
    (
        ctx,
        format!(
            "not given; nothing from {train} down to {MIN_DEFAULT_CTX} fits fully on GPU, so {ctx} with offload"
        ),
    )
}

fn kv_reason(forced: Option<KvType>, c: &Candidate, n_layer: u32, kv_order: &[KvType]) -> String {
    if forced.is_some() {
        return "requested with --kv".into();
    }
    let kv = c.cfg.kv;
    if !c.placement.is_full(n_layer) {
        return format!(
            "nothing fits with every layer on GPU; offloading keeps {} because quantising the cache further costs quality, offloading only speed",
            kv.flag()
        );
    }
    let rejected: Vec<&str> = kv_order
        .iter()
        .take_while(|&&k| k != kv)
        .map(|k| k.flag())
        .collect();
    if rejected.is_empty() {
        format!("{} fits with every layer on GPU", kv.flag())
    } else {
        format!(
            "{} does not fit with every layer on GPU; {} does",
            rejected.join(" or "),
            kv.flag()
        )
    }
}

fn no_fit(
    s: &Search<'_>,
    model: &ModelInfo,
    ctx: u32,
    closest: Candidate,
    trials: Vec<Trial>,
) -> Error {
    let gpu_over = closest.fp.gpu_bytes.saturating_sub(s.gpu.usable_bytes);
    let host_over = closest.fp.host_bytes.saturating_sub(s.host_budget);
    let (limit, short_by) = if gpu_over >= host_over {
        ("gpu", gpu_over)
    } else {
        ("host", host_over)
    };
    let mut fixes = Vec::new();
    if let Some(max) = s.max_full_ctx(ctx.saturating_sub(1)) {
        fixes.push(Fix {
            summary: format!("use --ctx {max} (largest context that fits fully on GPU)"),
        });
    }
    let held = if limit == "gpu" {
        s.gpu.in_use_bytes()
    } else {
        s.host_total.saturating_sub(s.host_budget)
    };
    if held >= short_by {
        fixes.push(Fix {
            summary: format!(
                "free {} of {} memory held by other processes ({} in use)",
                gib(short_by),
                if limit == "gpu" { "GPU" } else { "host" },
                gib(held)
            ),
        });
    }
    fixes.push(Fix {
        summary: format!(
            "use a quant at least {} smaller than {} ({})",
            gib(short_by),
            model.quant,
            gib(model.weight_bytes)
        ),
    });
    fixes.truncate(2);
    Error::CannotFit(Box::new(NoFit {
        model: summary(model),
        gpu: s.gpu.clone(),
        ctx,
        limit,
        short_by,
        closest: closest.fp,
        trials,
        fixes,
    }))
}
