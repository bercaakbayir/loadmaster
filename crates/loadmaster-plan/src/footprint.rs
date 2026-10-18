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
/// Fixed part of the llama.cpp compute buffer.
pub const COMPUTE_BASE: u64 = 128 * MIB;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvType {
    F16,
    Q8_0,
    Q4_0,
}

impl KvType {
    pub const ALL: [KvType; 3] = [KvType::F16, KvType::Q8_0, KvType::Q4_0];

    pub fn ggml(self) -> GgmlType {
        match self {
            KvType::F16 => GgmlType::F16,
            KvType::Q8_0 => GgmlType::Q8_0,
            KvType::Q4_0 => GgmlType::Q4_0,
        }
    }

    /// The llama.cpp `-ctk`/`-ctv` value.
    pub fn flag(self) -> &'static str {
        match self {
            KvType::F16 => "f16",
            KvType::Q8_0 => "q8_0",
            KvType::Q4_0 => "q4_0",
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

    /// llama.cpp offloads the *last* `-ngl` repeating layers.
    pub fn layer_on_gpu(&self, il: u32, n_layer: u32) -> bool {
        il >= n_layer.saturating_sub(self.n_gpu_layers)
    }

    /// The output head is offloaded only when `-ngl` exceeds the layer count.
    pub fn output_on_gpu(&self, n_layer: u32) -> bool {
        self.n_gpu_layers > n_layer
    }

    pub fn gpu_layer_count(&self, n_layer: u32) -> u32 {
        self.n_gpu_layers.min(n_layer)
    }

    pub fn is_full(&self, n_layer: u32) -> bool {
        self.output_on_gpu(n_layer) && self.n_cpu_moe == 0
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
    let t = kv.ggml();
    Some(t.row_bytes(layer.k_dim.into())? + t.row_bytes(layer.v_dim.into())?)
}

/// Whether every layer's KV rows are valid for `kv`.
pub fn kv_type_valid(model: &ModelInfo, kv: KvType) -> bool {
    model
        .layers
        .iter()
        .all(|l| kv_bytes_per_cell(kv, l).is_some())
}

/// llama.cpp compute buffer. With flash attention: `n_ubatch * (n_embd + n_vocab) * 4`
/// plus a fixed 128 MiB. Without it, the f32 attention scores for one ubatch are added.
pub fn compute_buffer(model: &ModelInfo, cfg: &Config) -> Term {
    let ub = u64::from(cfg.n_ubatch);
    let base = ub * (u64::from(model.n_embd) + u64::from(model.n_vocab)) * 4 + COMPUTE_BASE;
    if cfg.flash_attn {
        Term {
            key: "compute",
            bytes: base,
            detail: format!(
                "n_ubatch {ub} x (n_embd {} + n_vocab {}) x 4 B + 128 MiB, flash attention",
                model.n_embd, model.n_vocab
            ),
        }
    } else {
        let ctx = u64::from(cfg.ctx.next_multiple_of(KV_PAD));
        let scores = ub * ctx * u64::from(model.n_head) * 4;
        Term {
            key: "compute",
            bytes: base + scores,
            detail: format!(
                "n_ubatch {ub} x (n_embd {} + n_vocab {}) x 4 B + 128 MiB, plus attention scores n_ubatch x ctx {ctx} x n_head {} x 4 B",
                model.n_embd, model.n_vocab, model.n_head
            ),
        }
    }
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

    let mut weights_detail = format!("{gpu_layers}/{n_layer} layers");
    if placement.n_cpu_moe > 0 {
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
    } else {
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
        compute_buffer(model, cfg),
    ];
    gpu.extend(mmproj(model));
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
        let p = Placement {
            n_gpu_layers: 2,
            n_cpu_moe: 0,
        };
        let on: Vec<bool> = (0..4).map(|il| p.layer_on_gpu(il, 4)).collect();
        assert_eq!(on, [false, false, true, true]);
        assert!(!p.output_on_gpu(4));
        assert!(Placement::full(4).output_on_gpu(4));
        assert!(
            !Placement {
                n_gpu_layers: 4,
                n_cpu_moe: 0
            }
            .output_on_gpu(4)
        );
    }
}
