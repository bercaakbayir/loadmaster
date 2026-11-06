//! Per-architecture facts that GGUF metadata does not state directly: which layers use
//! sliding-window attention and how the KV cache is laid out. One row per architecture.
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArchSpec {
    pub name: &'static str,
    pub kv: KvLayout,
    pub swa: SwaPattern,
}

const fn gqa(name: &'static str) -> ArchSpec {
    ArchSpec {
        name,
        kv: KvLayout::Gqa,
        swa: SwaPattern::None,
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
    },
    ArchSpec {
        name: "gemma3",
        kv: KvLayout::Gqa,
        swa: SwaPattern::Every(6),
    },
    ArchSpec {
        name: "gpt-oss",
        kv: KvLayout::Gqa,
        swa: SwaPattern::Every(2),
    },
    ArchSpec {
        name: "deepseek2",
        kv: KvLayout::Mla,
        swa: SwaPattern::None,
    },
];

pub fn lookup(arch: &str) -> Option<&'static ArchSpec> {
    ARCHS.iter().find(|a| a.name == arch)
}

/// What a known non-LLM architecture is, for the "not supported" message: diffusion GGUFs
/// (city96's and unsloth's `flux`, `sd3`, `wan`, ...) and HF `model_type`s of
/// encoder-decoder and speech models. Prefixes cover versions (`flux2`, `wan2.2`).
pub fn non_llm_kind(arch: &str) -> Option<&'static str> {
    const KINDS: &[(&str, &str)] = &[
        ("flux", "image generation"),
        ("chroma", "image generation"),
        ("sd", "image generation"),
        ("aura", "image generation"),
        ("hidream", "image generation"),
        ("lumina", "image generation"),
        ("qwen_image", "image generation"),
        ("qwenimage", "image generation"),
        ("pixart", "image generation"),
        ("kolors", "image generation"),
        ("stablediffusion", "image generation"),
        ("wan", "video generation"),
        ("ltx", "video generation"),
        ("hyvid", "video generation"),
        ("hunyuanvideo", "video generation"),
        ("cosmos", "video generation"),
        ("mochi", "video generation"),
        ("cogvideo", "video generation"),
        ("whisper", "speech-to-text"),
        ("parakeet", "speech-to-text"),
        ("t5", "encoder-decoder"),
        ("mt5", "encoder-decoder"),
        ("umt5", "encoder-decoder"),
        ("bart", "encoder-decoder"),
        ("mbart", "encoder-decoder"),
        ("marian", "encoder-decoder"),
        ("pegasus", "encoder-decoder"),
    ];
    let a = arch.to_ascii_lowercase();
    // `t5encoder` is llama.cpp's embedding encoder, which loadmaster plans.
    if lookup(&a).is_some() {
        return None;
    }
    KINDS.iter().find_map(|&(prefix, kind)| {
        let rest = a.strip_prefix(prefix)?;
        // `sd` must not match `sdar`; versions and separators may follow.
        let ok = rest.is_empty()
            || rest.starts_with(|c: char| c.is_ascii_digit() || matches!(c, '.' | '_' | '-'))
            || (prefix != "sd" && prefix.len() > 3);
        ok.then_some(kind)
    })
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
    fn non_llm_kinds() {
        assert_eq!(non_llm_kind("flux"), Some("image generation"));
        assert_eq!(non_llm_kind("flux2"), Some("image generation"));
        assert_eq!(non_llm_kind("sd3"), Some("image generation"));
        assert_eq!(non_llm_kind("wan"), Some("video generation"));
        assert_eq!(non_llm_kind("t5"), Some("encoder-decoder"));
        assert_eq!(non_llm_kind("t5encoder"), None);
        assert_eq!(non_llm_kind("llama"), None);
        assert_eq!(non_llm_kind("qwen35"), None);
        assert_eq!(non_llm_kind("sdar"), None);
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
