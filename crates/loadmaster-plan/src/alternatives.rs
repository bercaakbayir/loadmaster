//! Nearby configurations, rendered under the command (spec task 1.9): the next context up,
//! one quant step up and down, and the fastest no-offload variant when the plan offloads.

use loadmaster_hw::{Gpu, HwSnapshot, MemoryKind};
use loadmaster_model::{Format, ModelInfo, Quant};
use serde::Serialize;

use crate::footprint::Placement;
use crate::{Error, Plan, PlanRequest, Runtime, plan_one};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Alternative {
    pub label: String,
    pub ctx: u32,
    pub quant: String,
    pub fits: bool,
    /// GPU bytes of the configuration, or of the closest miss.
    pub gpu_bytes: u64,
    /// Difference from the chosen plan, in bytes (negative = less).
    pub delta_bytes: i64,
    pub placement: String,
    /// Estimated slowdown versus everything on the GPU, as `(low, high)` x 100.
    pub slowdown_x100: Option<(u32, u32)>,
    /// Estimated generation speed (tok/s x10, low and high) when bandwidth is known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tok_s_x10: Option<(u32, u32)>,
    /// Caveats, e.g. that a quant size is estimated.
    pub note: Option<String>,
}

/// Bits per weight of llama.cpp quant mixes (as reported by llama-quantize for 7-8B
/// models), the ladder for "quant up/down".
const GGUF_LADDER: &[(&str, f64)] = &[
    ("Q2_K", 3.35),
    ("Q3_K_M", 3.91),
    ("Q4_K_M", 4.89),
    ("Q5_K_M", 5.70),
    ("Q6_K", 6.56),
    ("Q8_0", 8.50),
    ("F16", 16.0),
];

fn bpw(name: &str) -> Option<f64> {
    let known: &[(&str, f64)] = &[
        ("Q3_K_S", 3.50),
        ("Q3_K_L", 4.27),
        ("Q4_0", 4.55),
        ("Q4_K_S", 4.58),
        ("IQ4_XS", 4.25),
        ("IQ4_NL", 4.50),
        ("Q5_0", 5.54),
        ("Q5_K_S", 5.54),
        ("BF16", 16.0),
    ];
    GGUF_LADDER
        .iter()
        .chain(known)
        .find(|(n, _)| *n == name)
        .map(|(_, b)| *b)
}

/// `model` with every tensor scaled by `ratio` and relabelled as quant `to`: an estimate
/// of another quant of the same model (from bits per weight, or from file sizes).
pub fn requantized(model: &ModelInfo, to: &str, ratio: f64) -> ModelInfo {
    let scale = |b: u64| (b as f64 * ratio).round() as u64;
    let mut m = model.clone();
    for l in &mut m.layers {
        l.bytes = scale(l.bytes);
        l.expert_bytes = scale(l.expert_bytes);
    }
    m.embd_bytes = scale(m.embd_bytes);
    m.output_bytes = scale(m.output_bytes);
    m.other_bytes = scale(m.other_bytes);
    m.weight_bytes = scale(m.weight_bytes);
    m.quant = Quant::Gguf(to.to_owned());
    m
}

/// Share of per-token weight reads that happen on the CPU. For MoE only the routed
/// experts in use are read, so CPU-side experts count at `n_used / n_expert`.
fn cpu_read_fraction(model: &ModelInfo, p: Placement) -> f64 {
    let (gpu, cpu) = crate::speed::weight_reads(model, p);
    if gpu + cpu == 0 {
        0.0
    } else {
        cpu as f64 / (gpu + cpu) as f64
    }
}

/// Token generation is memory-bound: time scales with bytes read over bandwidth. A
/// dedicated GPU reads 6-16x faster than dual-channel host RAM; on unified memory the GPU
/// is 1.5-3x faster than the CPU cores on the same RAM. Always a band, always an estimate.
pub fn slowdown_x100(model: &ModelInfo, p: Placement, gpu: &Gpu) -> Option<(u32, u32)> {
    let f = cpu_read_fraction(model, p);
    if f == 0.0 {
        return None;
    }
    let (lo, hi) = match gpu.memory {
        MemoryKind::Dedicated => (6.0, 16.0),
        MemoryKind::Unified => (1.5, 3.0),
    };
    let band = |r: f64| (((1.0 - f) + f * r) * 100.0).round() as u32;
    Some((band(lo), band(hi)))
}

fn placement_text(p: &Plan) -> String {
    let n = p.model.n_layer;
    let pl = p.placement;
    if pl.is_full(n) {
        format!("all on GPU, {} KV", p.kv.flag())
    } else if pl.gpu_layer_count(n) == n {
        format!(
            "experts of {} layers on CPU, {} KV",
            pl.n_cpu_moe,
            p.kv.flag()
        )
    } else {
        format!(
            "{}/{n} layers on GPU, {} KV",
            pl.gpu_layer_count(n),
            p.kv.flag()
        )
    }
}

fn row(
    label: String,
    model: &ModelInfo,
    hw: &HwSnapshot,
    gpu: &Gpu,
    req: &PlanRequest,
    base: &Plan,
    note: Option<String>,
) -> Option<Alternative> {
    let delta = |b: u64| b as i64 - base.footprint.gpu_bytes as i64;
    match plan_one(model, hw, gpu, req) {
        Ok(p) => Some(Alternative {
            label,
            ctx: p.ctx,
            quant: p.model.quant.clone(),
            fits: true,
            gpu_bytes: p.footprint.gpu_bytes,
            delta_bytes: delta(p.footprint.gpu_bytes),
            placement: placement_text(&p),
            slowdown_x100: slowdown_x100(model, p.placement, gpu),
            tok_s_x10: p
                .speed
                .as_ref()
                .and_then(|s| s.at.first())
                .map(|a| a.tok_s_x10),
            note,
        }),
        Err(Error::CannotFit(n)) => Some(Alternative {
            label,
            ctx: n.ctx,
            quant: n.model.quant.clone(),
            fits: false,
            gpu_bytes: n.closest.gpu_bytes,
            delta_bytes: delta(n.closest.gpu_bytes),
            placement: format!("does not fit ({} short)", crate::gib(n.short_by)),
            slowdown_x100: None,
            tok_s_x10: None,
            note,
        }),
        Err(_) => None,
    }
}

pub fn compute(
    model: &ModelInfo,
    hw: &HwSnapshot,
    gpu: &Gpu,
    req: &PlanRequest,
    base: &Plan,
) -> Vec<Alternative> {
    let mut out = Vec::new();
    let at_ctx = |ctx: u32| PlanRequest {
        ctx: Some(ctx),
        ..req.clone()
    };

    // More context: the next power of two, up to the training context.
    let up = (base.ctx + 1).next_power_of_two().min(model.n_ctx_train);
    if up > base.ctx {
        out.extend(row(
            format!("more context ({up})"),
            model,
            hw,
            gpu,
            &at_ctx(up),
            base,
            None,
        ));
    }

    // Quant up and down, GGUF only: the ladder neighbours, sizes scaled by bits per weight.
    if model.format == Format::Gguf
        && let Quant::Gguf(name) = &model.quant
        && let Some(cur) = bpw(name)
    {
        let lower = GGUF_LADDER.iter().rev().find(|(_, b)| *b < cur - 0.05);
        let higher = GGUF_LADDER.iter().find(|(_, b)| *b > cur + 0.05);
        for (dir, step) in [("up", higher), ("down", lower)] {
            if let Some((to, b)) = step {
                let m = requantized(model, to, b / cur);
                out.extend(row(
                    format!("quant {dir} ({to})"),
                    &m,
                    hw,
                    gpu,
                    &at_ctx(base.ctx),
                    base,
                    Some(format!(
                        "size estimated from {b:.2} bpw; plan the real file to confirm"
                    )),
                ));
            }
        }
    }

    // The plan offloads: the largest context that runs entirely on the GPU instead.
    if req.runtime == Runtime::LlamaCpp && !base.placement.is_full(model.n_layer) {
        let mut ctx = base.ctx;
        while ctx > 512 {
            ctx /= 2;
            if let Ok(p) = plan_one(model, hw, gpu, &at_ctx(ctx))
                && p.placement.is_full(model.n_layer)
            {
                out.extend(row(
                    format!("no offload ({ctx})"),
                    model,
                    hw,
                    gpu,
                    &at_ctx(ctx),
                    base,
                    None,
                ));
                break;
            }
        }
    }

    // A vision projector that crowds the GPU: keep it on the CPU instead (images then
    // encode on the CPU, slower, but the language model keeps the room).
    if req.runtime == Runtime::LlamaCpp
        && req.mmproj_offload
        && let Some(m) = &model.mmproj
        && (!base.placement.is_full(model.n_layer)
            || base.kv == crate::KvType::Q4_0
            || base.warnings.iter().any(|w| w.starts_with("tight on RAM")))
    {
        let r = PlanRequest {
            mmproj_offload: false,
            ..at_ctx(base.ctx)
        };
        out.extend(row(
            "projector on CPU".into(),
            model,
            hw,
            gpu,
            &r,
            base,
            Some(format!(
                "--no-mmproj-offload: frees {} of projector weights and encoder; images encode on the CPU",
                crate::gib(m.weight_bytes + m.compute_bytes)
            )),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ladder_is_sorted() {
        assert!(GGUF_LADDER.windows(2).all(|w| w[0].1 < w[1].1));
    }
}
