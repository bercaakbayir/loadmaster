//! Per-architecture facts that GGUF metadata does not state directly: which layers use
//! sliding-window attention, which are recurrent, and how the KV cache is laid out. One
//! row per architecture.
//!
//! Patterns follow `llama_model::load_hparams` in llama.cpp.

use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KvLayout {
    /// K and V rows of `n_head_kv * head_dim` per layer (MHA, GQA, MQA).
    Gqa,
    /// Multi-head latent attention: one compressed latent of `kv_lora_rank` plus the
    /// RoPE dimensions per layer (DeepSeek-V2/V3, Kimi K2).
    Mla,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SwaPattern {
    None,
    /// llama.cpp `set_swa_pattern(n)`: layer `il` is sliding-window iff `il % n < n - 1`,
    /// i.e. every `n`th layer is global.
    Every(u32),
}

impl SwaPattern {
    pub fn is_swa(self, il: u32) -> bool {
        match self {
            SwaPattern::None => false,
            SwaPattern::Every(n) => n > 0 && il % n < n - 1,
        }
    }
}

/// Hybrid models whose other layers are recurrent (Gated DeltaNet linear attention):
/// they keep a fixed-size state per sequence instead of a KV cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RecurrentPattern {
    None,
    /// Layer `il` is full attention iff `(il + 1) % n == 0`; every other layer is
    /// recurrent. `n` is `{arch}.full_attention_interval` (`full_attention_interval` in
    /// config.json) when stated, this default otherwise. llama.cpp src/models/qwen35.cpp,
    /// qwen35moe.cpp and qwen3next.cpp `load_hparams` at 6c01da6.
    FullAttentionEvery(u32),
}

impl RecurrentPattern {
    /// The same pattern with the interval the file states, if it states one.
    pub fn with_interval(self, n: Option<u32>) -> Self {
        match (self, n) {
            (RecurrentPattern::FullAttentionEvery(_), Some(n)) if n > 0 => {
                RecurrentPattern::FullAttentionEvery(n)
            }
            (p, _) => p,
        }
    }

    pub fn is_recurrent(self, il: u32) -> bool {
        match self {
            RecurrentPattern::None => false,
            RecurrentPattern::FullAttentionEvery(n) => n > 0 && (il + 1) % n != 0,
        }
    }
}

/// Shape of a Gated DeltaNet layer's state, from the `{arch}.ssm.*` keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SsmDims {
    /// `ssm.conv_kernel`
    pub d_conv: u32,
    /// `ssm.inner_size`: value heads x value head dim.
    pub d_inner: u32,
    /// `ssm.state_size`: key (and value) head dim.
    pub d_state: u32,
    /// `ssm.group_count`: key heads.
    pub n_group: u32,
}

/// llama.cpp keeps recurrent state in f32 (`type_r`, `type_s` default).
pub const RECURRENT_STATE_ELEM_BYTES: u64 = 4;

impl SsmDims {
    /// Conv state elements per sequence (llama.cpp `n_embd_r`, src/llama-hparams.cpp at
    /// 6c01da6): the last `d_conv - 1` inputs of the Q, K and V channels.
    pub fn n_embd_r(self) -> u64 {
        u64::from(self.d_conv.saturating_sub(1))
            * (u64::from(self.d_inner) + 2 * u64::from(self.n_group) * u64::from(self.d_state))
    }

    /// Recurrent state elements per sequence (llama.cpp `n_embd_s`).
    pub fn n_embd_s(self) -> u64 {
        u64::from(self.d_state) * u64::from(self.d_inner)
    }

    /// Bytes of one layer's conv and recurrent state for one sequence (one row each of
    /// `cache_r_l` and `cache_s_l`, llama-memory-recurrent.cpp).
    pub fn state_bytes(self) -> u64 {
        (self.n_embd_r() + self.n_embd_s()) * RECURRENT_STATE_ELEM_BYTES
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchSpec {
    pub name: &'static str,
    pub kv: KvLayout,
    pub swa: SwaPattern,
    pub recurrent: RecurrentPattern,
}

const fn gqa(name: &'static str) -> ArchSpec {
    ArchSpec {
        name,
        kv: KvLayout::Gqa,
        swa: SwaPattern::None,
        recurrent: RecurrentPattern::None,
    }
}

const fn hybrid(name: &'static str) -> ArchSpec {
    ArchSpec {
        recurrent: RecurrentPattern::FullAttentionEvery(4),
        ..gqa(name)
    }
}

pub const ARCHS: &[ArchSpec] = &[
    gqa("llama"),
    gqa("llama4"),
    gqa("mistral3"),
    gqa("qwen2"),
    gqa("qwen2moe"),
    gqa("qwen2vl"),
    gqa("qwen3"),
    gqa("qwen3moe"),
    gqa("phi3"),
    gqa("granite"),
    gqa("glm4"),
    gqa("glm4moe"),
    // Qwen3.5 and Qwen3-Next: Gated DeltaNet layers with full attention every 4th.
    hybrid("qwen35"),
    hybrid("qwen35moe"),
    hybrid("qwen3next"),
    // Embedding encoders: no KV cache in llama.cpp, so no attention pattern matters.
    gqa("bert"),
    gqa("modern-bert"),
    gqa("nomic-bert"),
    gqa("nomic-bert-moe"),
    gqa("neo-bert"),
    gqa("jina-bert-v2"),
    gqa("jina-bert-v3"),
    gqa("eurobert"),
    gqa("t5encoder"),
    ArchSpec {
        name: "gemma2",
        kv: KvLayout::Gqa,
        swa: SwaPattern::Every(2),
        recurrent: RecurrentPattern::None,
    },
    ArchSpec {
        name: "gemma3",
        kv: KvLayout::Gqa,
        swa: SwaPattern::Every(6),
        recurrent: RecurrentPattern::None,
    },
    ArchSpec {
        name: "gpt-oss",
        kv: KvLayout::Gqa,
        swa: SwaPattern::Every(2),
        recurrent: RecurrentPattern::None,
    },
    ArchSpec {
        name: "deepseek2",
        kv: KvLayout::Mla,
        swa: SwaPattern::None,
        recurrent: RecurrentPattern::None,
    },
];

pub fn lookup(arch: &str) -> Option<&'static ArchSpec> {
    ARCHS.iter().find(|a| a.name == arch)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma3_has_one_global_layer_in_six() {
        let p = lookup("gemma3").unwrap().swa;
        let swa: Vec<bool> = (0..12).map(|il| p.is_swa(il)).collect();
        assert_eq!(
            swa,
            [
                true, true, true, true, true, false, true, true, true, true, true, false
            ]
        );
    }

    #[test]
    fn gpt_oss_alternates_starting_with_swa() {
        let p = lookup("gpt-oss").unwrap().swa;
        assert!(p.is_swa(0));
        assert!(!p.is_swa(1));
    }

    #[test]
    fn qwen35_is_full_attention_every_fourth_layer() {
        let p = lookup("qwen35").unwrap().recurrent;
        let recurrent: Vec<bool> = (0..8).map(|il| p.is_recurrent(il)).collect();
        assert_eq!(
            recurrent,
            [true, true, true, false, true, true, true, false]
        );
        let p = p.with_interval(Some(2));
        assert!(p.is_recurrent(0));
        assert!(!p.is_recurrent(1));
        assert!(
            !RecurrentPattern::None
                .with_interval(Some(4))
                .is_recurrent(0)
        );
    }

    #[test]
    fn qwen35_9b_state_matches_llama_cpp() {
        // Qwen3.5-9B: conv_kernel 4, inner_size 4096, state_size 128, group_count 16.
        let d = SsmDims {
            d_conv: 4,
            d_inner: 4096,
            d_state: 128,
            n_group: 16,
        };
        assert_eq!(d.n_embd_r(), 3 * 8192);
        assert_eq!(d.n_embd_s(), 128 * 4096);
        assert_eq!(d.state_bytes(), (24_576 + 524_288) * 4);
    }

    #[test]
    fn names_are_unique() {
        for (i, a) in ARCHS.iter().enumerate() {
            assert!(
                ARCHS[i + 1..].iter().all(|b| b.name != a.name),
                "{}",
                a.name
            );
        }
    }
}
