//! Generation speed from memory bandwidth (ADR 0009). Producing a token reads every weight
//! that token uses (for MoE, only the routed experts in use) and the KV cache so far, so
//! time per token is bytes over bandwidth, plus a fixed overhead per layer (kernel
//! launches and synchronisation). Prompt processing is compute-bound and not estimated.
//!
//! `t = gpu_bytes / (BW_gpu x e) + cpu_bytes / BW_host + n_layer x c`, reported as a band
//! from the `e` and `c` ranges of the backend. The ranges were fitted so every published
//! llama.cpp tg128 result in `validation/speed.toml` falls inside them.

use loadmaster_hw::{Gpu, MemoryKind, Vendor};
use loadmaster_model::ModelInfo;
use serde::Serialize;

use crate::footprint::{self, Config, Placement};

/// Fitted ranges for one backend: bandwidth efficiency `e` and per-layer overhead `c`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Band {
    pub name: &'static str,
    pub eff: (f64, f64),
    /// Microseconds per layer per token.
    pub per_layer_us: (f64, f64),
}

/// Fitted to 41 Metal tg128 results (M1 to M4 Max, Q4_0/Q8_0/F16 7B): e 0.86-0.97,
/// c 140-180 us, widened by a few points for models outside that set.
pub const METAL: Band = Band {
    name: "Metal",
    eff: (0.84, 0.97),
    per_layer_us: (130.0, 190.0),
};

/// Fitted to 8 CUDA tg128 results (RTX 3080 to A100, Q4_0 7B): e 0.98-0.99, c 45-105 us,
/// widened for models outside that set; few points, so the widest band.
pub const CUDA: Band = Band {
    name: "CUDA",
    eff: (0.95, 1.0),
    per_layer_us: (40.0, 110.0),
};

/// Host RAM bandwidth for CPU-offloaded weights next to a dedicated GPU: dual-channel
/// DDR4-3200 (51 GB/s) to DDR5-6000 (96 GB/s), as a range, since it cannot be detected.
const HOST_DEDICATED_GBS: (f64, f64) = (40.0, 90.0);

/// The estimate at one context fill.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SpeedAt {
    /// Tokens already in the context (KV read per new token).
    pub fill: u32,
    /// Bytes read per token from GPU memory and from host RAM.
    pub gpu_bytes: u64,
    pub cpu_bytes: u64,
    /// Tokens per second, x10 (the band's low and high ends).
    pub tok_s_x10: (u32, u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Speed {
    pub at: Vec<SpeedAt>,
    pub backend: &'static str,
    pub bandwidth_bytes_s: u64,
    pub bandwidth_basis: String,
    /// How CPU-side reads were costed, when any weights are on the CPU.
    pub host_basis: Option<String>,
}

impl Speed {
    /// `95-150 tok/s` at the first fill.
    pub fn headline(&self) -> String {
        self.at
            .first()
            .map_or(String::new(), |a| band_text(a.tok_s_x10))
    }
}

pub fn band_text((lo, hi): (u32, u32)) -> String {
    let f = |x: u32| {
        if x >= 100 {
            format!("{}", (x + 5) / 10)
        } else {
            format!("{:.1}", f64::from(x) / 10.0)
        }
    };
    format!("{}-{} tok/s", f(lo), f(hi))
}

/// Bytes of weights read per token on the GPU and on the CPU. MoE routed experts count at
/// `n_expert_used / n_expert`; token embeddings are a row lookup and are not counted.
pub fn weight_reads(model: &ModelInfo, p: Placement) -> (u64, u64) {
    let n = model.n_layer;
    let active = match (model.n_expert, model.n_expert_used) {
        (Some(e), Some(u)) if e > 0 => f64::from(u) / f64::from(e),
        _ => 1.0,
    };
    let (mut gpu, mut cpu) = (0.0, 0.0);
    for (il, l) in (0u32..).zip(&model.layers) {
        let dense = l.bytes.saturating_sub(l.expert_bytes) as f64;
        let experts = l.expert_bytes as f64 * active;
        if p.layer_on_gpu(il, n) {
            gpu += dense;
            if il < p.n_cpu_moe {
                cpu += experts;
            } else {
                gpu += experts;
            }
        } else {
            cpu += dense + experts;
        }
    }
    let out = model.output_bytes as f64;
    if p.output_on_gpu(n) {
        gpu += out;
    } else {
        cpu += out;
    }
    (gpu.round() as u64, cpu.round() as u64)
}

/// KV bytes read per token with `fill` tokens in the context, on the GPU and on the CPU
/// (a layer's cache lives with the layer). Sliding-window layers read at most the window.
fn kv_reads(model: &ModelInfo, p: Placement, cfg: &Config, fill: u32) -> (u64, u64) {
    let (mut gpu, mut cpu) = (0, 0);
    for (il, l) in (0u32..).zip(&model.layers) {
        let cells = match (l.swa, model.swa_window) {
            (true, Some(w)) => fill.min(w),
            _ => fill,
        };
        let b = footprint::kv_bytes_per_cell(cfg.kv, l).unwrap_or(0) * u64::from(cells);
        if p.layer_on_gpu(il, model.n_layer) {
            gpu += b;
        } else {
            cpu += b;
        }
    }
    (gpu, cpu)
}

pub fn backend(gpu: &Gpu) -> Band {
    match gpu.vendor {
        Vendor::Apple => METAL,
        Vendor::Nvidia => CUDA,
    }
}

/// Host bandwidth range in bytes/s for CPU-side reads, and how it was chosen.
fn host_bandwidth(gpu: &Gpu, bw: u64) -> ((f64, f64), String) {
    match gpu.memory {
        // The same RAM, but the CPU cores reach a fraction of what the GPU does.
        MemoryKind::Unified => (
            (bw as f64 / 3.0, bw as f64 / 1.5),
            "CPU cores on the same unified memory: 1/3 to 2/3 of its bandwidth (unvalidated)"
                .into(),
        ),
        MemoryKind::Dedicated => (
            (HOST_DEDICATED_GBS.0 * 1e9, HOST_DEDICATED_GBS.1 * 1e9),
            format!(
                "host RAM assumed {:.0}-{:.0} GB/s (dual-channel DDR4 to DDR5; not detected)",
                HOST_DEDICATED_GBS.0, HOST_DEDICATED_GBS.1
            ),
        ),
    }
}

/// Tokens per second x10, low and high, for bytes read per token.
pub fn tok_s_x10(
    band: Band,
    gpu_bytes: u64,
    cpu_bytes: u64,
    n_layer: u32,
    bw: u64,
    host_bw: (f64, f64),
) -> (u32, u32) {
    let t = |eff: f64, host: f64, c_us: f64| {
        gpu_bytes as f64 / (bw as f64 * eff)
            + if cpu_bytes > 0 {
                cpu_bytes as f64 / host
            } else {
                0.0
            }
            + f64::from(n_layer) * c_us * 1e-6
    };
    let slow = t(band.eff.0, host_bw.0, band.per_layer_us.1);
    let fast = t(band.eff.1, host_bw.1, band.per_layer_us.0);
    let x10 = |t: f64| (10.0 / t).round() as u32;
    (x10(slow), x10(fast))
}

/// The estimate for a placement, at 4k context (when the plan is longer) and at the
/// planned context; `None` when the GPU's bandwidth is unknown.
pub fn estimate(model: &ModelInfo, p: Placement, cfg: &Config, gpu: &Gpu) -> Option<Speed> {
    let bw = gpu.bandwidth_bytes_s?;
    let band = backend(gpu);
    let (host_bw, host_text) = host_bandwidth(gpu, bw);
    let (wg, wc) = weight_reads(model, p);
    let fills: Vec<u32> = if cfg.ctx > 4096 {
        vec![4096, cfg.ctx]
    } else {
        vec![cfg.ctx]
    };
    let at = fills
        .into_iter()
        .map(|fill| {
            let (kg, kc) = kv_reads(model, p, cfg, fill);
            let (g, c) = (wg + kg, wc + kc);
            SpeedAt {
                fill,
                gpu_bytes: g,
                cpu_bytes: c,
                tok_s_x10: tok_s_x10(band, g, c, model.n_layer, bw, host_bw),
            }
        })
        .collect();
    Some(Speed {
        at,
        backend: band.name,
        bandwidth_bytes_s: bw,
        bandwidth_basis: gpu.bandwidth_basis.clone(),
        host_basis: (wc > 0).then_some(host_text),
    })
}

/// Several GPUs splitting the layers (`split[d]` layer slots each, the output head the last
/// slot of the last GPU): they work one after another on each token, so the time is the
/// sum of each GPU's bytes over its own bandwidth. `None` unless every GPU's bandwidth is
/// known.
pub fn estimate_split(
    model: &ModelInfo,
    p: Placement,
    cfg: &Config,
    gpus: &[Gpu],
    split: &[u32],
) -> Option<Speed> {
    let bws: Vec<u64> = gpus
        .iter()
        .map(|g| g.bandwidth_bytes_s)
        .collect::<Option<_>>()?;
    let band = backend(&gpus[0]);
    let (host_bw, host_text) = host_bandwidth(&gpus[0], bws[0]);
    let n = model.n_layer;
    let active = match (model.n_expert, model.n_expert_used) {
        (Some(e), Some(u)) if e > 0 => f64::from(u) / f64::from(e),
        _ => 1.0,
    };
    // The GPU of each GPU-resident layer, in order.
    let mut device_of = Vec::new();
    for (d, &k) in split.iter().enumerate() {
        device_of.extend(std::iter::repeat_n(d, k as usize));
    }
    let fills: Vec<u32> = if cfg.ctx > 4096 {
        vec![4096, cfg.ctx]
    } else {
        vec![cfg.ctx]
    };
    let mut any_cpu = false;
    let at = fills
        .into_iter()
        .map(|fill| {
            let mut dev = vec![0f64; gpus.len()];
            let mut cpu = 0f64;
            let mut slot = 0;
            for (il, l) in (0u32..).zip(&model.layers) {
                let dense = l.bytes.saturating_sub(l.expert_bytes) as f64;
                let experts = l.expert_bytes as f64 * active;
                let cells = match (l.swa, model.swa_window) {
                    (true, Some(w)) => fill.min(w),
                    _ => fill,
                };
                let kv = (footprint::kv_bytes_per_cell(cfg.kv, l).unwrap_or(0) * u64::from(cells))
                    as f64
                    * if model.has_kv_cache() { 1.0 } else { 0.0 };
                if p.layer_on_gpu(il, n) {
                    let d = device_of.get(slot).copied().unwrap_or(gpus.len() - 1);
                    slot += 1;
                    dev[d] += dense + kv;
                    if il < p.n_cpu_moe {
                        cpu += experts;
                    } else {
                        dev[d] += experts;
                    }
                } else {
                    cpu += dense + experts + kv;
                }
            }
            if p.output_on_gpu(n) {
                dev[gpus.len() - 1] += model.output_bytes as f64;
            } else {
                cpu += model.output_bytes as f64;
            }
            any_cpu |= cpu > 0.0;
            let t = |eff: f64, host: f64, c_us: f64| {
                dev.iter()
                    .zip(&bws)
                    .map(|(b, bw)| b / (*bw as f64 * eff))
                    .sum::<f64>()
                    + if cpu > 0.0 { cpu / host } else { 0.0 }
                    + f64::from(n) * c_us * 1e-6
            };
            let slow = t(band.eff.0, host_bw.0, band.per_layer_us.1);
            let fast = t(band.eff.1, host_bw.1, band.per_layer_us.0);
            SpeedAt {
                fill,
                gpu_bytes: dev.iter().sum::<f64>().round() as u64,
                cpu_bytes: cpu.round() as u64,
                tok_s_x10: ((10.0 / slow).round() as u32, (10.0 / fast).round() as u32),
            }
        })
        .collect();
    Some(Speed {
        at,
        backend: band.name,
        bandwidth_bytes_s: bws.iter().copied().min().unwrap_or(0),
        bandwidth_basis: format!(
            "{} GPUs in sequence: {}",
            gpus.len(),
            gpus.iter()
                .map(|g| g.bandwidth_basis.as_str())
                .collect::<Vec<_>>()
                .join(" | ")
        ),
        host_basis: any_cpu.then_some(host_text),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: f64 = (1u64 << 30) as f64;

    #[test]
    fn bands_contain_the_published_benchmarks() {
        #[derive(serde::Deserialize)]
        struct File {
            point: Vec<Point>,
        }
        #[derive(serde::Deserialize)]
        struct Point {
            device: String,
            backend: String,
            bandwidth_gbs: f64,
            n_layer: u32,
            model_gib: f64,
            tg_tok_s: f64,
        }
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../validation/speed.toml");
        let f: File = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(f.point.len() >= 40);
        for p in f.point {
            let band = if p.backend == "metal" { METAL } else { CUDA };
            let (lo, hi) = tok_s_x10(
                band,
                (p.model_gib * GIB) as u64,
                0,
                p.n_layer,
                (p.bandwidth_gbs * 1e9) as u64,
                (1.0, 1.0),
            );
            let obs = (p.tg_tok_s * 10.0).round() as u32;
            assert!(
                lo <= obs && obs <= hi,
                "{} {} GiB: observed {} outside {}",
                p.device,
                p.model_gib,
                p.tg_tok_s,
                band_text((lo, hi))
            );
        }
    }

    #[test]
    fn offload_is_slower_and_bands_are_ordered() {
        let on_gpu = tok_s_x10(CUDA, 4_000_000_000, 0, 32, 1_008_000_000_000, (40e9, 90e9));
        let half = tok_s_x10(
            CUDA,
            2_000_000_000,
            2_000_000_000,
            32,
            1_008_000_000_000,
            (40e9, 90e9),
        );
        assert!(on_gpu.0 < on_gpu.1 && half.0 < half.1);
        assert!(half.1 < on_gpu.0, "{half:?} vs {on_gpu:?}");
    }

    #[test]
    fn band_text_rounds_sensibly() {
        assert_eq!(band_text((953, 1504)), "95-150 tok/s");
        assert_eq!(band_text((42, 97)), "4.2-9.7 tok/s");
    }
}
