//! Footprint arithmetic (spec §7, with the llama.cpp placement rules in ADR 0004).
//!
//! Every function here is pure over `ModelInfo` and a [`Config`]; every term carries the
//! inputs it was computed from so `--explain` can print it.

use loadmaster_hw::MIB;
use loadmaster_model::gguf::GgmlType;
use loadmaster_model::{Layer, ModelInfo};
use serde::Serialize;

/// llama.cpp pads the KV cache size to a multiple of this.
pub const KV_PAD: u32 = 256;
/// Fixed parts of the llama.cpp compute buffer (ADR 0005).
pub const COMPUTE_BASE: u64 = 16 * MIB;
pub const COMPUTE_BASE_NO_FA: u64 = 32 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    /// 16-bit (f16 on llama.cpp, the model dtype on vLLM, FP16 on exl3).
    F16,
    Q8_0,
    Q4_0,
    /// vLLM `--kv-cache-dtype fp8`: one byte per element.
    Fp8,
}

impl KvType {
    /// The llama.cpp cache types, in preference order.
    pub const ALL: [KvType; 3] = [KvType::F16, KvType::Q8_0, KvType::Q4_0];

    /// Bytes for a row of `n` elements, or `None` when `n` is not a whole number of blocks.
    pub fn row_bytes(self, n: u64) -> Option<u64> {
        match self {
            KvType::F16 => GgmlType::F16.row_bytes(n),
            KvType::Q8_0 => GgmlType::Q8_0.row_bytes(n),
            KvType::Q4_0 => GgmlType::Q4_0.row_bytes(n),
            KvType::Fp8 => Some(n),
        }
    }

    /// The llama.cpp `-ctk`/`-ctv` value (and the name used in explanations).
    pub fn flag(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
            KvType::Q4_0 => "q4_0",
            KvType::Fp8 => "fp8",
        }
    }

    pub fn is_quantized(self) -> bool {
        self != KvType::F16
    }
}

/// Where the weights go, in llama.cpp terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Placement {
    /// The `-ngl` value. `n_layer + 1` or more offloads every layer and the output head.
    pub n_gpu_layers: u32,
    /// `--n-cpu-moe`: routed experts of the first `k` layers stay on the CPU.
    pub n_cpu_moe: u32,
}

impl Placement {
    pub fn full(n_layer: u32) -> Self {
        Placement {
            n_gpu_layers: n_layer + 1,
            n_cpu_moe: 0,
        }
    }

    /// llama.cpp counts the output head as one of the `-ngl` layers and offloads it first,
    /// then the *last* `ngl - 1` repeating layers (observed: ADR 0004).
    pub fn layer_on_gpu(&self, il: u32, n_layer: u32) -> bool {
        il >= (n_layer + 1).saturating_sub(self.n_gpu_layers)
    }

    pub fn output_on_gpu(&self, _n_layer: u32) -> bool {
        self.n_gpu_layers >= 1
    }

    pub fn gpu_layer_count(&self, n_layer: u32) -> u32 {
        self.n_gpu_layers.saturating_sub(1).min(n_layer)
    }

    pub fn is_full(&self, n_layer: u32) -> bool {
        self.gpu_layer_count(n_layer) == n_layer && self.n_cpu_moe == 0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Headroom {
    pub bytes: u64,
    pub basis: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Config {
    pub ctx: u32,
    pub n_seq: u32,
    pub n_ubatch: u32,
    pub kv: KvType,
    pub flash_attn: bool,
    pub headroom: Headroom,
    /// The vision projector's weights and encoder on the GPU (llama.cpp's default;
    /// `--no-mmproj-offload` keeps them on the CPU).
    pub mmproj_on_gpu: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Term {
    /// Stable identifier: `weights`, `kv`, `compute`, `mmproj`, `headroom`.
    pub key: &'static str,
    pub bytes: u64,
    /// How the number was computed, with its inputs.
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Footprint {
    /// Device-side terms; they sum to `gpu_bytes`.
    pub gpu: Vec<Term>,
    pub gpu_bytes: u64,
    /// Host-side terms (weights and KV left on the CPU); they sum to `host_bytes`.
    pub host: Vec<Term>,
    pub host_bytes: u64,
}

impl Footprint {
    pub fn term(&self, key: &str) -> Option<&Term> {
        self.gpu.iter().find(|t| t.key == key)
    }
}

/// KV cells a layer allocates. Sliding-window layers hold the window plus one ubatch
/// (llama.cpp `size_swa`), never more than the full context.
pub fn kv_cells(cfg: &Config, layer: &Layer, window: Option<u32>) -> u64 {
    let full = u64::from(cfg.ctx.next_multiple_of(KV_PAD));
    match (layer.swa, window) {
        (true, Some(w)) => {
            let swa = (u64::from(w) * u64::from(cfg.n_seq) + u64::from(cfg.n_ubatch))
                .next_multiple_of(u64::from(KV_PAD));
            swa.min(full)
        }
        _ => full,
    }
}

/// Bytes per KV cell for one layer, or `None` when the row width is not a whole number
/// of blocks for a quantised cache type.
pub fn kv_bytes_per_cell(kv: KvType, layer: &Layer) -> Option<u64> {
    Some(kv.row_bytes(layer.k_dim.into())? + kv.row_bytes(layer.v_dim.into())?)
}

/// Whether every layer's KV rows are valid for `kv`.
pub fn kv_type_valid(model: &ModelInfo, kv: KvType) -> bool {
    model
        .layers
        .iter()
        .all(|l| kv_bytes_per_cell(kv, l).is_some())
}

/// llama.cpp GPU compute buffer: the peak of live intermediate tensors for one ubatch.
///
/// Fitted to buffer sizes llama.cpp reported (`validation/observed.toml`, ADR 0005), rounded
/// up so every observation is covered:
/// - flash attention: `16 MiB + mask + max(FFN, attention)`, where
///   mask = `2 B x n_ubatch x ctx`, FFN = `16 B x n_ubatch x ffn_width`, and
///   attention = `12 B x n_ubatch x (n_embd + q + k + v) + 2 B x ctx x (k + v)`;
/// - without: `32 MiB + n_ubatch x ctx x (4 B x n_head + 4 B)` (f32 scores and mask).
pub fn compute_buffer(model: &ModelInfo, cfg: &Config) -> Term {
    let ub = u64::from(cfg.n_ubatch);
    let ctx = u64::from(cfg.ctx.next_multiple_of(KV_PAD));
    let (k, v) = kv_width(model);
    let q = u64::from(model.n_head) * u64::from(model.head_dim);
    let embd = u64::from(model.n_embd);
    if cfg.flash_attn {
        let mask = 2 * ub * ctx;
        let ffn = 16 * ub * u64::from(model.ffn_width());
        let attn = 12 * ub * (embd + q + k + v) + 2 * ctx * (k + v);
        let (peak, which) = if ffn >= attn {
            (
                ffn,
                format!("FFN 16 B x n_ubatch {ub} x ffn {}", model.ffn_width()),
            )
        } else {
            (
                attn,
                format!(
                    "attention 12 B x n_ubatch {ub} x (n_embd {embd} + q {q} + k {k} + v {v}) + 2 B x ctx {ctx} x (k + v)"
                ),
            )
        };
        Term {
            key: "compute",
            bytes: COMPUTE_BASE + mask + peak,
            detail: format!(
                "16 MiB + mask 2 B x n_ubatch {ub} x ctx {ctx} + peak of {which}; flash attention"
            ),
        }
    } else {
        let scores = ub * ctx * (4 * u64::from(model.n_head) + 4);
        Term {
            key: "compute",
            bytes: COMPUTE_BASE_NO_FA + scores,
            detail: format!(
                "32 MiB + f32 scores and mask n_ubatch {ub} x ctx {ctx} x (4 B x n_head {} + 4 B)",
                model.n_head
            ),
        }
    }
}

/// Host compute buffer. With every layer on the GPU llama.cpp keeps only the input side on
/// the CPU (observed exactly `2 B x n_ubatch x ctx + 16 B x n_ubatch x n_embd`); with layers
/// on the CPU the CPU runs a full graph, bounded by the GPU formula.
pub fn host_compute_buffer(model: &ModelInfo, cfg: &Config, all_layers_on_gpu: bool) -> Term {
    if all_layers_on_gpu {
        let ub = u64::from(cfg.n_ubatch);
        let ctx = u64::from(cfg.ctx.next_multiple_of(KV_PAD));
        Term {
            key: "compute_cpu",
            bytes: 2 * ub * ctx + 16 * ub * u64::from(model.n_embd),
            detail: format!(
                "mask 2 B x n_ubatch {ub} x ctx {ctx} + 16 B x n_ubatch x n_embd {}",
                model.n_embd
            ),
        }
    } else {
        let t = compute_buffer(model, cfg);
        Term {
            key: "compute_cpu",
            bytes: t.bytes,
            detail: "layers run on the CPU: same bound as the GPU compute buffer".into(),
        }
    }
}

/// Largest per-layer K and V widths.
fn kv_width(model: &ModelInfo) -> (u64, u64) {
    model.layers.iter().fold((0, 0), |(k, v), l| {
        (k.max(u64::from(l.k_dim)), v.max(u64::from(l.v_dim)))
    })
}

pub fn mmproj(model: &ModelInfo) -> Option<Term> {
    let m = model.mmproj.as_ref()?;
    let embeddings = u64::from(m.n_img_tokens) * u64::from(model.n_embd) * 2;
    Some(Term {
        key: "mmproj",
        bytes: m.weight_bytes + embeddings,
        detail: format!(
            "{} weights + one image: {} tokens ({}) x n_embd {} x 2 B",
            gib(m.weight_bytes),
            m.n_img_tokens,
            m.img_tokens_basis,
            model.n_embd
        ),
    })
}

/// The vision encoder's compute buffer (an upper-bound estimate; ADR 0011).
pub fn vision_compute(model: &ModelInfo) -> Option<Term> {
    let m = model.mmproj.as_ref()?;
    (m.compute_bytes > 0).then(|| Term {
        key: "vision_compute",
        bytes: m.compute_bytes,
        detail: m.compute_basis.clone(),
    })
}

/// The full footprint of `model` under `placement` and `cfg`. `None` when `cfg.kv` is not
/// valid for this model's KV row widths.
pub fn footprint(model: &ModelInfo, placement: Placement, cfg: &Config) -> Option<Footprint> {
    let n_layer = model.n_layer;
    let mut w_gpu = 0u64;
    let mut w_cpu = model.embd_bytes;
    let mut experts_cpu = 0u64;
    let mut kv_gpu = 0u64;
    let mut kv_cpu = 0u64;
    let mut gpu_layers = 0u32;
    for (il, layer) in (0u32..).zip(&model.layers) {
        let kv = kv_bytes_per_cell(cfg.kv, layer)? * kv_cells(cfg, layer, model.swa_window);
        if placement.layer_on_gpu(il, n_layer) {
            gpu_layers += 1;
            kv_gpu += kv;
            if il < placement.n_cpu_moe {
                w_gpu += layer.bytes - layer.expert_bytes;
                experts_cpu += layer.expert_bytes;
            } else {
                w_gpu += layer.bytes;
            }
        } else {
            kv_cpu += kv;
            w_cpu += layer.bytes;
        }
    }
    let output_gpu = placement.output_on_gpu(n_layer);
    if output_gpu {
        w_gpu += model.output_bytes;
    } else {
        w_cpu += model.output_bytes;
    }
    // Small non-layer tensors (rope_freqs etc.) follow the layers.
    if gpu_layers > 0 {
        w_gpu += model.other_bytes;
    } else {
        w_cpu += model.other_bytes;
    }
    w_cpu += experts_cpu;

    let mut weights_detail = if gpu_layers == 0 && !output_gpu {
        "nothing on GPU".to_owned()
    } else {
        format!("{gpu_layers}/{n_layer} layers")
    };
    if experts_cpu > 0 {
        weights_detail += &format!(
            " (experts of first {} layers on CPU, {})",
            placement.n_cpu_moe.min(n_layer),
            gib(experts_cpu)
        );
    }
    if output_gpu {
        weights_detail += &format!(
            " + output head {}{}",
            gib(model.output_bytes),
            if model.output_tied {
                " (tied: second copy of token embeddings)"
            } else {
                ""
            }
        );
    } else if gpu_layers > 0 {
        weights_detail += "; output head on CPU";
    }
    weights_detail += &format!("; token embeddings {} stay on CPU", gib(model.embd_bytes));

    let mut gpu = vec![
        Term {
            key: "weights",
            bytes: w_gpu,
            detail: weights_detail,
        },
        Term {
            key: "kv",
            bytes: kv_gpu,
            detail: kv_detail(model, cfg, gpu_layers),
        },
        if gpu_layers > 0 {
            compute_buffer(model, cfg)
        } else {
            Term {
                key: "compute",
                bytes: 0,
                detail: "no layers on GPU".into(),
            }
        },
    ];
    let vision: Vec<Term> = mmproj(model)
        .into_iter()
        .chain(vision_compute(model))
        .collect();
    let (vision_gpu, vision_cpu) = if cfg.mmproj_on_gpu {
        (vision, Vec::new())
    } else {
        (Vec::new(), vision)
    };
    gpu.extend(vision_gpu);
    gpu.push(Term {
        key: "headroom",
        bytes: cfg.headroom.bytes,
        detail: cfg.headroom.basis.clone(),
    });
    let gpu_bytes = gpu.iter().map(|t| t.bytes).sum();

    let mut host = vec![Term {
        key: "weights_cpu",
        bytes: w_cpu,
        detail: format!(
            "token embeddings{}{}{}",
            if gpu_layers < n_layer {
                format!(" + {} layers", n_layer - gpu_layers)
            } else {
                String::new()
            },
            if experts_cpu > 0 {
                " + routed experts"
            } else {
                ""
            },
            if output_gpu { "" } else { " + output head" },
        ),
    }];
    if kv_cpu > 0 {
        host.push(Term {
            key: "kv_cpu",
            bytes: kv_cpu,
            detail: format!("KV of {} CPU layers", n_layer - gpu_layers),
        });
    }
    host.push(host_compute_buffer(
        model,
        cfg,
        gpu_layers == n_layer && experts_cpu == 0,
    ));
    for t in vision_cpu {
        host.push(Term {
            key: if t.key == "mmproj" {
                "mmproj_cpu"
            } else {
                "vision_compute_cpu"
            },
            detail: format!("{} (on CPU: --no-mmproj-offload)", t.detail),
            ..t
        });
    }
    let host_bytes = host.iter().map(|t| t.bytes).sum();
    Some(Footprint {
        gpu,
        gpu_bytes,
        host,
        host_bytes,
    })
}

fn kv_detail(model: &ModelInfo, cfg: &Config, gpu_layers: u32) -> String {
    let ty = cfg.kv.flag();
    let gpu = &model.layers[(model.n_layer - gpu_layers) as usize..];
    let Some(sample) = gpu.first() else {
        return format!("{ty}, no layers on GPU");
    };
    let width = |l: &Layer| format!("(K {} + V {})", l.k_dim, l.v_dim);
    let elem = match cfg.kv {
        KvType::F16 => "2 B".to_owned(),
        KvType::Q8_0 => "34/32 B".to_owned(),
        KvType::Q4_0 => "18/32 B".to_owned(),
        KvType::Fp8 => "1 B".to_owned(),
    };
    let swa = gpu.iter().filter(|l| l.swa).count();
    if swa > 0 {
        let swa_layer = gpu.iter().find(|l| l.swa).unwrap_or(sample);
        let global_layer = gpu.iter().find(|l| !l.swa).unwrap_or(sample);
        format!(
            "{ty}: {} global layers x {} cells x {} + {swa} sliding-window layers x {} cells (window {}) x {}, x {elem}",
            gpu.len() - swa,
            kv_cells(cfg, global_layer, model.swa_window),
            width(global_layer),
            kv_cells(cfg, swa_layer, model.swa_window),
            model.swa_window.unwrap_or(0),
            width(swa_layer),
        )
    } else {
        format!(
            "{ty}: {} layers x {} cells x {}{} x {elem}",
            gpu.len(),
            kv_cells(cfg, sample, model.swa_window),
            width(sample),
            if model.kv_layout == loadmaster_model::KvLayout::Mla {
                " (MLA latent)"
            } else {
                ""
            },
        )
    }
}

/// Bytes as GiB with two decimals, the unit every human-facing number uses.
pub fn gib(bytes: u64) -> String {
    format!("{:.2} GiB", bytes as f64 / (1u64 << 30) as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn layer(k: u32, v: u32, swa: bool) -> Layer {
        Layer {
            bytes: 0,
            expert_bytes: 0,
            k_dim: k,
            v_dim: v,
            swa,
        }
    }

    fn cfg(ctx: u32, kv: KvType) -> Config {
        Config {
            ctx,
            n_seq: 1,
            n_ubatch: 512,
            kv,
            flash_attn: true,
            mmproj_on_gpu: true,
            headroom: Headroom {
                bytes: 0,
                basis: String::new(),
            },
        }
    }

    #[test]
    fn llama3_8b_kv_matches_llama_cpp() {
        // llama.cpp reports "KV self size = 1024.00 MiB" for Llama-3-8B at -c 8192, f16,
        // and 544.00 MiB at q8_0: 32 layers x 8 KV heads x 128 dims.
        let l = layer(1024, 1024, false);
        let per_layer =
            |kv| kv_bytes_per_cell(kv, &l).unwrap() * kv_cells(&cfg(8192, kv), &l, None);
        assert_eq!(32 * per_layer(KvType::F16), 1024 * MIB);
        assert_eq!(32 * per_layer(KvType::Q8_0), 544 * MIB);
        assert_eq!(32 * per_layer(KvType::Q4_0), 288 * MIB);
    }

    #[test]
    fn swa_layers_cap_at_window_plus_ubatch() {
        let l = layer(2048, 2048, true);
        assert_eq!(kv_cells(&cfg(32768, KvType::F16), &l, Some(1024)), 1536);
        assert_eq!(kv_cells(&cfg(1000, KvType::F16), &l, Some(1024)), 1024);
        assert_eq!(
            kv_cells(&cfg(32768, KvType::F16), &layer(1, 1, false), Some(1024)),
            32768
        );
    }

    #[test]
    fn ctx_is_padded() {
        assert_eq!(
            kv_cells(&cfg(1000, KvType::F16), &layer(1, 1, false), None),
            1024
        );
    }

    #[test]
    fn quantised_kv_needs_whole_blocks() {
        assert!(kv_bytes_per_cell(KvType::Q8_0, &layer(80, 80, false)).is_none());
        assert!(kv_bytes_per_cell(KvType::F16, &layer(80, 80, false)).is_some());
    }

    #[test]
    fn placement_offloads_last_layers() {
        // -ngl 3 on 4 layers: output head + the last 2 layers.
        let p = Placement {
            n_gpu_layers: 3,
            n_cpu_moe: 0,
        };
        let on: Vec<bool> = (0..4).map(|il| p.layer_on_gpu(il, 4)).collect();
        assert_eq!(on, [false, false, true, true]);
        assert!(p.output_on_gpu(4));
        assert_eq!(p.gpu_layer_count(4), 2);
        assert!(Placement::full(4).is_full(4));
        assert!(
            !Placement {
                n_gpu_layers: 0,
                n_cpu_moe: 0
            }
            .output_on_gpu(4)
        );
    }
}
