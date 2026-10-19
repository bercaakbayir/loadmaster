//! Vision projectors: how many embeddings one image becomes and how big the vision encoder's
//! working set is, per `clip.projector_type`. Each rule mirrors llama.cpp's
//! `clip_n_output_tokens` and warmup sizing (tools/mtmd/clip.cpp at commit f174553; see
//! docs/decisions/0011-vision-and-embeddings.md).

use crate::gguf::Gguf;

/// What the planner needs from a projector.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projector {
    pub kind: String,
    /// False when `kind` is not in the table and the plain patch-grid rule was assumed.
    pub known: bool,
    /// Embeddings one image becomes (for dynamic-resolution projectors, the default
    /// maximum: a large image).
    pub n_img_tokens: u32,
    pub tokens_basis: String,
    /// Patches the encoder runs on for the image llama.cpp reserves its compute buffer
    /// with (`warmup_image_size`).
    pub warmup_patches: u32,
    pub n_embd: u32,
    pub n_head: u32,
    pub n_ff: u32,
}

/// Families of `clip_n_output_tokens` rules.
enum Rule {
    /// `(image_size / patch)^2`, unchanged.
    Grid,
    /// The grid divided by 4 (LDP, LDPv2, GLM-Edge adapter).
    Quarter,
    /// The grid divided by `scale_factor^2` (pixel shuffle / pooling), `scale_factor`
    /// from metadata or the given default.
    Shuffle(u32),
    /// A fixed query count (MiniCPM-V resampler).
    Queries,
    /// Dynamic resolution with an `n x n` spatial merge: up to `max_tokens` per image,
    /// reserved at `warmup_tokens` (Qwen-VL family, GLM-4V, Pixtral, Kimi-VL, ...).
    Dynamic {
        merge: u32,
        max_tokens: u32,
        warmup_tokens: u32,
    },
    /// Gemma 3n: `image_size / patch` tokens.
    Side,
}

fn rule(kind: &str) -> Option<Rule> {
    Some(match kind {
        "mlp" | "mlp_norm" | "janus_pro" | "phi4" => Rule::Grid,
        "ldp" | "ldpv2" | "adapter" => Rule::Quarter,
        "gemma3" => Rule::Shuffle(4),
        "idefics3" | "internvl" | "llama4" | "cohere2v" | "nemotron" => Rule::Shuffle(2),
        "gemma4v" => Rule::Dynamic {
            merge: 3,
            max_tokens: 1120,
            warmup_tokens: 256,
        },
        "resampler" => Rule::Queries,
        "qwen2vl_merger" | "qwen2.5vl_merger" | "qwen3vl_merger" | "glm4v" => Rule::Dynamic {
            merge: 2,
            max_tokens: 4096,
            warmup_tokens: 46 * 46,
        },
        "pixtral" | "lightonocr" => Rule::Dynamic {
            merge: 1,
            max_tokens: 1024,
            warmup_tokens: 256,
        },
        "kimivl" => Rule::Dynamic {
            merge: 2,
            max_tokens: 1024,
            warmup_tokens: 256,
        },
        "lfm2" => Rule::Dynamic {
            merge: 2,
            max_tokens: 256,
            warmup_tokens: 256,
        },
        "gemma3nv" => Rule::Side,
        _ => return None,
    })
}

impl Projector {
    pub fn from_gguf(g: &Gguf) -> Self {
        let kind = g
            .get_str("clip.projector_type")
            .or_else(|| g.get_str("clip.vision.projector_type"))
            .unwrap_or("unknown")
            .to_owned();
        let image = g.get_u32("clip.vision.image_size").unwrap_or(336);
        let patch = g.get_u32("clip.vision.patch_size").unwrap_or(14).max(1);
        let n_embd = g.get_u32("clip.vision.embedding_length").unwrap_or(1024);
        let n_ff = g
            .get_u32("clip.vision.feed_forward_length")
            .unwrap_or(n_embd * 4);
        // SigLIP and CLIP towers use 64- or 72-wide heads; 64 over-counts heads, so the
        // attention estimate errs high when the file does not say.
        let n_head = g
            .get_u32("clip.vision.attention.head_count")
            .unwrap_or((n_embd / 64).max(1));
        let scale = g.get_u32("clip.vision.projector.scale_factor");
        let merge_meta = g.get_u32("clip.vision.spatial_merge_size");
        let side = image / patch;
        let grid = side * side;
        let (known, n_img_tokens, tokens_basis, warmup_patches) = match rule(&kind) {
            Some(Rule::Grid) => (true, grid, format!("({image}/{patch})^2 patches"), grid),
            Some(Rule::Quarter) => (
                true,
                grid / 4,
                format!("({image}/{patch})^2 patches / 4"),
                grid,
            ),
            Some(Rule::Shuffle(default)) => {
                let s = scale.unwrap_or(default).max(1);
                (
                    true,
                    grid / (s * s),
                    format!("({image}/{patch})^2 patches / {s}^2"),
                    grid,
                )
            }
            Some(Rule::Queries) => {
                let q = g.get_u32("clip.minicpmv_query_num").unwrap_or(64);
                (true, q, format!("{q} resampler queries"), grid)
            }
            Some(Rule::Dynamic {
                merge,
                max_tokens,
                warmup_tokens,
            }) => {
                let m = merge_meta.unwrap_or(merge).max(1);
                (
                    true,
                    max_tokens,
                    format!(
                        "up to {max_tokens} per image (llama.cpp's default maximum; dynamic resolution)"
                    ),
                    warmup_tokens * m * m,
                )
            }
            Some(Rule::Side) => (true, side, format!("{image}/{patch} tokens"), grid),
            None => (
                false,
                grid,
                format!("({image}/{patch})^2 patches, assumed for unknown projector `{kind}`"),
                grid,
            ),
        };
        Projector {
            kind,
            known,
            n_img_tokens,
            tokens_basis,
            warmup_patches,
            n_embd,
            n_head,
            n_ff,
        }
    }

    /// Estimate of the vision encoder's compute buffer, reserved for the warmup image. clip
    /// runs flash attention whenever the backend supports it (`CLIP_FLASH_ATTN_TYPE_AUTO`,
    /// the default), so this is the language model's flash-attention compute shape
    /// (ADR 0005) with the image's patches as one batch: mask, plus the peak of the FFN and
    /// the attention projections. Unvalidated: no observed run yet.
    pub fn compute_bytes(&self) -> u64 {
        let p = u64::from(self.warmup_patches);
        let (e, f) = (u64::from(self.n_embd), u64::from(self.n_ff));
        let ffn = 16 * p * f;
        let attn = 12 * p * 4 * e + 2 * p * 2 * e;
        (16 << 20) + 2 * p * p + ffn.max(attn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf::Value;
    use crate::gguf::writer::GgufWriter;

    fn projector(kind: &str, kv: &[(&str, u32)]) -> Projector {
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("clip".into()));
        w.kv("clip.projector_type", Value::String(kind.into()));
        for (k, v) in kv {
            w.kv(k, Value::U32(*v));
        }
        Projector::from_gguf(&Gguf::parse_bytes(&w.finish_header_only()).unwrap())
    }

    const SIGLIP: [(&str, u32); 4] = [
        ("clip.vision.image_size", 896),
        ("clip.vision.patch_size", 14),
        ("clip.vision.embedding_length", 1152),
        ("clip.vision.attention.head_count", 16),
    ];

    #[test]
    fn gemma3_pools_to_256() {
        let p = projector("gemma3", &SIGLIP);
        assert!(p.known);
        assert_eq!(p.n_img_tokens, 256);
        assert_eq!(p.warmup_patches, 64 * 64);
    }

    #[test]
    fn llava_mlp_is_the_patch_grid() {
        let p = projector(
            "mlp",
            &[
                ("clip.vision.image_size", 336),
                ("clip.vision.patch_size", 14),
            ],
        );
        assert_eq!(p.n_img_tokens, 576);
    }

    #[test]
    fn qwen_vl_is_dynamic_and_reserved_at_the_warmup_size() {
        let p = projector(
            "qwen2.5vl_merger",
            &[
                ("clip.vision.patch_size", 14),
                ("clip.vision.embedding_length", 1280),
            ],
        );
        assert_eq!(p.n_img_tokens, 4096);
        assert_eq!(p.warmup_patches, 46 * 46 * 4);
        assert!(p.tokens_basis.contains("dynamic"));
    }

    #[test]
    fn scale_factor_and_queries_come_from_metadata() {
        let p = projector(
            "idefics3",
            &[
                ("clip.vision.image_size", 512),
                ("clip.vision.patch_size", 16),
                ("clip.vision.projector.scale_factor", 4),
            ],
        );
        assert_eq!(p.n_img_tokens, 64);
        let p = projector("resampler", &[("clip.minicpmv_query_num", 96)]);
        assert_eq!(p.n_img_tokens, 96);
    }

    #[test]
    fn unknown_projectors_fall_back_to_the_grid() {
        let p = projector(
            "brand-new",
            &[
                ("clip.vision.image_size", 448),
                ("clip.vision.patch_size", 14),
            ],
        );
        assert!(!p.known);
        assert_eq!(p.n_img_tokens, 1024);
        assert!(p.tokens_basis.contains("brand-new"));
    }

    #[test]
    fn compute_bound_grows_with_patches_squared() {
        let small = projector(
            "mlp",
            &[
                ("clip.vision.image_size", 224),
                ("clip.vision.patch_size", 14),
            ],
        );
        let big = projector(
            "mlp",
            &[
                ("clip.vision.image_size", 448),
                ("clip.vision.patch_size", 14),
            ],
        );
        // 4x the patches: the mask grows 16x, activations 4x; the sum lands between.
        let ratio = big.compute_bytes() as f64 / small.compute_bytes() as f64;
        assert!((2.0..16.0).contains(&ratio), "{ratio}");
        // Gemma 3's SigLIP at 896 px: ~0.3 GiB, not the ~1.2 GiB of full f32 attention scores.
        let g = projector("gemma3", &SIGLIP);
        let gib = g.compute_bytes() as f64 / f64::from(1u32 << 30);
        assert!((0.2..0.5).contains(&gib), "{gib}");
    }
}
