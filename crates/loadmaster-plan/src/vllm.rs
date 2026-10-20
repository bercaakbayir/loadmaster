//! vLLM planning (spec §7): vLLM claims `gpu_memory_utilization x total` up front and fills
//! what weights and activations leave with KV blocks. The plan picks the largest
//! utilisation that fits in free memory, then the largest `max_model_len` whose KV fits.

use loadmaster_hw::{Gpu, HwSnapshot, MIB, Vendor};
use loadmaster_model::{Format, ModelInfo, Quant};

use crate::footprint::{Footprint, KvType, Placement, Term};
use crate::{
    Error, Fix, NoFit, Plan, PlanRequest, Trial, emit, gib, headroom_for, host_budget, summary,
};

/// vLLM allocates the KV cache in blocks of this many tokens.
pub const BLOCK: u32 = 16;
pub const MAX_UTILIZATION_X100: u64 = 95;

/// Activation memory vLLM's profiler reserves: 0.5 GiB + 20 MiB per billion parameters.
pub fn activation_bytes(model: &ModelInfo) -> u64 {
    512 * MIB + (model.params_b() * 20.0).ceil() as u64 * MIB
}

/// KV bytes for one sequence of `ctx` tokens. Sliding-window layers keep at most the
/// window (vLLM's hybrid allocator).
pub fn kv_bytes(model: &ModelInfo, ctx: u32, kv: KvType) -> u64 {
    let ctx = u64::from(ctx.next_multiple_of(BLOCK));
    let elem = match kv {
        KvType::Fp8 => 1,
        _ => 2,
    };
    model
        .layers
        .iter()
        .map(|l| {
            let tokens = match (l.swa, model.swa_window) {
                (true, Some(w)) => ctx.min(u64::from(w)),
                _ => ctx,
            };
            (u64::from(l.k_dim) + u64::from(l.v_dim)) * elem * tokens
        })
        .sum()
}

/// Largest context (a multiple of the block size, at most `upper`) whose KV fits `budget`.
fn max_ctx(model: &ModelInfo, kv: KvType, budget: u64, upper: u32) -> Option<u32> {
    if kv_bytes(model, BLOCK, kv) > budget {
        return None;
    }
    let (mut lo, mut hi) = (1u32, (upper / BLOCK).max(1));
    while lo < hi {
        let mid = lo + (hi - lo).div_ceil(2);
        if kv_bytes(model, mid * BLOCK, kv) <= budget {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    Some(lo * BLOCK)
}

fn kv_name(kv: KvType) -> &'static str {
    match kv {
        KvType::Fp8 => "fp8",
        _ => "auto (model dtype)",
    }
}

pub(crate) fn plan(
    model: &ModelInfo,
    hw: &HwSnapshot,
    gpu: &Gpu,
    req: &PlanRequest,
    mut warnings: Vec<crate::Warning>,
) -> Result<Plan, Error> {
    let unavailable = |reason: String| Error::RuntimeUnavailable {
        runtime: "vllm",
        reason,
    };
    if gpu.vendor != Vendor::Nvidia {
        return Err(unavailable(
            "vLLM needs an NVIDIA GPU (CUDA); use --runtime llama.cpp".into(),
        ));
    }
    if model.format != Format::Safetensors {
        return Err(unavailable(
            "vLLM serves Hugging Face model directories; this is a GGUF file, use --runtime llama.cpp".into(),
        ));
    }
    if matches!(model.quant, Quant::Exl3 { .. }) {
        return Err(unavailable(
            "EXL3 quants run on ExLlamaV3; use --runtime exl3".into(),
        ));
    }

    let headroom = headroom_for(gpu, req.headroom);
    // Tensor parallel over several GPUs: the largest size that divides the attention heads
    // (vLLM's only divisibility check; KV heads are replicated when fewer than the size).
    let all: Vec<&Gpu> = std::iter::once(gpu).chain(&req.extra_gpus).collect();
    let tp = [8usize, 4, 2, 1]
        .into_iter()
        .find(|&t| t <= all.len() && model.n_head as usize % t == 0)
        .unwrap_or(1);
    let used = &all[..tp];
    if tp < all.len() {
        warnings.push(crate::Warning::new(
            format!("vLLM: using {tp} of {} GPUs", all.len()),
            format!(
                "the tensor-parallel size must divide the {} attention heads; set CUDA_VISIBLE_DEVICES to the ones you want",
                model.n_head
            ),
        ));
    }
    if used.iter().any(|g| g.total_bytes != used[0].total_bytes) {
        warnings.push(crate::Warning::new(
            "vLLM sizes every GPU like the smallest",
            "one fraction of memory applies to every GPU in a tensor-parallel group",
        ));
    }
    // --gpu-memory-utilization is one fraction of each GPU's total; the tightest GPU sets it.
    let util_x100 = used
        .iter()
        .map(|g| g.usable_bytes.saturating_sub(headroom.bytes) * 100 / g.total_bytes.max(1))
        .min()
        .unwrap_or(0)
        .min(MAX_UTILIZATION_X100);
    let total = used.iter().map(|g| g.total_bytes).min().unwrap_or(0);
    let pool = total / 100 * util_x100;
    let vision = model.mmproj.as_ref().map_or(0, |m| m.weight_bytes);
    let weights = model.weight_bytes - vision;
    let activation = activation_bytes(model);
    let mm = crate::footprint::mmproj(model);
    // Per GPU: a 1/tp shard of the weights, plus the activation reserve and the vision
    // tower, which every rank holds.
    let fixed = weights / tp as u64 + activation + mm.as_ref().map_or(0, |t| t.bytes);
    // KV pool across the group, divided by how many times KV heads are replicated.
    let n_kv = u64::from(model.n_head_kv.max(1));
    let kv_per_rank = (n_kv / tp as u64).max(1);
    let kv_budget = pool.saturating_sub(fixed) * tp as u64 * n_kv / (kv_per_rank * tp as u64);

    let kv_order: Vec<KvType> = match req.kv {
        Some(kv) => vec![kv],
        None => vec![KvType::F16, KvType::Fp8],
    };
    let upper = req.ctx.unwrap_or(model.n_ctx_train);
    let mut trials = Vec::new();
    let mut chosen = None;
    for &kv in &kv_order {
        let max = max_ctx(model, kv, kv_budget, upper.max(model.n_ctx_train));
        let need = match req.ctx {
            Some(ctx) => kv_bytes(model, ctx, kv),
            None => max.map_or(0, |m| kv_bytes(model, m.min(model.n_ctx_train), kv)),
        };
        let ok = match req.ctx {
            Some(_) => need <= kv_budget,
            // Without --ctx, accept the model dtype if it reaches the training context or
            // 8192 tokens; otherwise try fp8 for more.
            None => max.is_some_and(|m| m >= model.n_ctx_train.min(8192)),
        };
        trials.push(Trial {
            label: format!(
                "KV {}: pool holds {} tokens",
                kv_name(kv),
                max.map_or("0".into(), |m| m.to_string())
            ),
            gpu_bytes: (fixed + headroom.bytes) * tp as u64 + need,
            host_bytes: 0,
            fits: ok,
        });
        if ok {
            chosen = Some((kv, max.unwrap_or(0)));
            break;
        }
    }
    // Without --ctx, fall back to whatever fits rather than failing on the 8192 floor.
    if chosen.is_none() && req.ctx.is_none() {
        chosen = kv_order
            .iter()
            .rev()
            .find_map(|&kv| max_ctx(model, kv, kv_budget, model.n_ctx_train).map(|m| (kv, m)));
    }

    let ranks = tp as u64;
    let per_gpu = |s: String| {
        if tp > 1 {
            format!("{s}; x {tp} GPUs")
        } else {
            s
        }
    };
    let headroom_term = Term {
        key: "headroom",
        bytes: headroom.bytes * ranks,
        detail: per_gpu(format!(
            "{}; outside --gpu-memory-utilization",
            headroom.basis
        )),
    };
    let mut terms = vec![
        Term {
            key: "weights",
            bytes: weights,
            detail: if tp > 1 {
                format!("every tensor on GPU, sharded {tp} ways (tensor parallel)")
            } else {
                "every tensor on GPU (vLLM does not offload by default)".into()
            },
        },
        Term {
            key: "activation",
            bytes: activation * ranks,
            detail: per_gpu(format!(
                "vLLM profiling reserve: 512 MiB + 20 MiB x {:.1}B params",
                model.params_b()
            )),
        },
    ];
    terms.extend(mm.map(|t| Term {
        bytes: t.bytes * ranks,
        detail: per_gpu(t.detail),
        ..t
    }));

    let Some((kv, max)) = chosen else {
        // Bytes missing for the smallest acceptable KV: the requested context, or one block.
        let last = *kv_order.last().unwrap_or(&KvType::Fp8);
        let want = kv_bytes(model, req.ctx.unwrap_or(BLOCK), last);
        let short_by = ((fixed + want / ranks).saturating_sub(pool) * ranks).max(1);
        terms.push(headroom_term);
        let gpu_bytes = terms.iter().map(|t| t.bytes).sum();
        return Err(Error::CannotFit(Box::new(NoFit {
            model: summary(model),
            gpu: gpu.clone(),
            gpus: if used.len() > 1 {
                used.iter().map(|g| (*g).clone()).collect()
            } else {
                Vec::new()
            },
            ctx: upper,
            limit: "gpu",
            short_by,
            closest: Footprint {
                gpu: terms,
                gpu_bytes,
                host: vec![],
                host_bytes: 0,
                per_device: Vec::new(),
            },
            trials,
            fixes: vec![
                Fix {
                    summary: match max_ctx(model, KvType::Fp8, kv_budget, upper) {
                        Some(m) => format!("use --ctx {m} (largest that fits with fp8 KV)"),
                        None => "weights and activations alone exceed the GPU; use a smaller quant (AWQ/GPTQ 4-bit)".into(),
                    },
                },
                Fix {
                    summary: "use --runtime llama.cpp with a GGUF quant, which can offload layers to the CPU".into(),
                },
            ],
        })));
    };

    let ctx = req.ctx.unwrap_or(max.min(model.n_ctx_train));
    let ctx_reason = match req.ctx {
        Some(_) => "requested with --ctx".to_owned(),
        None if ctx == model.n_ctx_train => format!(
            "not given; the training context {} fits in the KV pool",
            model.n_ctx_train
        ),
        None => format!(
            "not given; the largest max_model_len whose KV fits the pool (training context {})",
            model.n_ctx_train
        ),
    };
    if kv == KvType::Fp8 && req.kv.is_none() {
        warnings.push(crate::Warning::new(
            "fp8 KV cache: small quality cost",
            "needs a GPU with fp8 support for best speed",
        ));
    }
    terms.push(Term {
        key: "kv",
        bytes: kv_budget,
        detail: format!(
            "KV pool {}: what --gpu-memory-utilization 0.{util_x100:02} leaves; {} tokens of {} KV fit, one {ctx}-token sequence needs {}",
            gib(kv_budget),
            max,
            kv_name(kv),
            gib(kv_bytes(model, ctx, kv))
        ),
    });
    terms.push(headroom_term);
    let gpu_bytes = terms.iter().map(|t| t.bytes).sum();
    let kv_reason = match (req.kv, kv) {
        (Some(_), _) => "requested with --kv".to_owned(),
        (None, KvType::Fp8) => {
            "the model dtype holds too few tokens; fp8 doubles the pool".to_owned()
        }
        _ => "the model dtype fits".to_owned(),
    };
    let mut plan = Plan {
        model: summary(model),
        gpu: gpu.clone(),
        runtime: req.runtime,
        ctx,
        ctx_reason,
        placement: Placement::full(model.n_layer),
        kv,
        kv_reason,
        flash_attn: true,
        n_ubatch: req.n_ubatch,
        footprint: Footprint {
            gpu: terms,
            gpu_bytes,
            host: vec![],
            host_bytes: 0,
            per_device: if tp > 1 {
                used.iter()
                    .map(|_| crate::footprint::DeviceSide {
                        layers: model.n_layer,
                        bytes: gpu_bytes / ranks,
                    })
                    .collect()
            } else {
                Vec::new()
            },
        },
        budget: gpu.usable_bytes,
        host_budget: host_budget(&hw.host),
        host_need: 0,
        host_basis: hw.host.available_basis.clone(),
        trials,
        warnings,
        argv: Vec::new(),
        config_file: None,
        alternatives: Vec::new(),
        mmproj_on_cpu: false,
        gpus: if tp > 1 {
            used.iter().map(|g| (*g).clone()).collect()
        } else {
            Vec::new()
        },
        devices: Vec::new(),
        speed: None,
    };
    plan.argv = emit::vllm::argv(&plan, util_x100);
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use loadmaster_model::Layer;

    fn layer(swa: bool) -> Layer {
        Layer {
            bytes: 0,
            expert_bytes: 0,
            k_dim: 1024,
            v_dim: 1024,
            swa,
        }
    }

    #[test]
    fn kv_bytes_counts_blocks_and_windows() {
        let mut m = loadmaster_model_stub();
        m.layers = vec![layer(false), layer(true)];
        m.swa_window = Some(1024);
        // Full layer: 8192 x 2048 x 2 B; SWA layer: 1024 x 2048 x 2 B.
        assert_eq!(kv_bytes(&m, 8192, KvType::F16), 8192 * 4096 + 1024 * 4096);
        assert_eq!(
            kv_bytes(&m, 8192, KvType::Fp8),
            (8192 * 4096 + 1024 * 4096) / 2
        );
        assert_eq!(kv_bytes(&m, 1, KvType::F16), 16 * 4096 * 2);
    }

    fn loadmaster_model_stub() -> ModelInfo {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/hf/Meta-Llama-3.1-8B-Instruct");
        ModelInfo::open(dir).unwrap()
    }
}
