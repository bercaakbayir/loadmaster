//! Projectors (`mmproj`): how many embeddings one image or one 30 s audio clip becomes and
//! how big each encoder's working set is, per projector type. Each rule mirrors llama.cpp's
//! `clip_n_output_tokens` and warmup sizing (tools/mtmd/clip.cpp at commit f174553; see
//! docs/decisions/0011-vision-and-embeddings.md and 0015-audio-encoders.md).

use crate::gguf::Gguf;

/// What the planner needs from one encoder of a projector file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Projector {
    pub kind: String,
    /// False when `kind` is not in the table and a fallback rule was assumed.
    pub known: bool,
    /// Embeddings one image becomes (for dynamic-resolution projectors, the default
    /// maximum: a large image), or one 30 s audio chunk.
    pub n_tokens: u32,
    pub tokens_basis: String,
    /// Positions the encoder runs on for the input llama.cpp reserves its compute buffer
    /// with: image patches (`warmup_image_size`) or audio frames after the convolutions
    /// (`warmup_audio_size`).
    pub warmup_positions: u32,
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

/// The projector type `clip.cpp` resolves for one modality: the shared key first, then the
/// per-modality one. Qwen2.5-Omni's `qwen2.5o` stands for both of its encoders.
fn projector_type(g: &Gguf, modality: &str) -> String {
    let kind = g
        .get_str("clip.projector_type")
        .or_else(|| g.get_str(&format!("clip.{modality}.projector_type")))
        .unwrap_or("unknown");
    match (kind, modality) {
        ("qwen2.5o", "vision") => "qwen2.5vl_merger",
        ("qwen2.5o", _) => "qwen2a",
        (k, _) => k,
    }
    .to_owned()
}

/// Which encoders a projector file holds (`clip.has_vision_encoder`,
/// `clip.has_audio_encoder`). A file that states neither is an older vision projector.
pub fn modalities(g: &Gguf) -> (bool, bool) {
    let flag = |k: &str| matches!(g.get(k), Some(crate::gguf::Value::Bool(true)));
    let (v, a) = (
        flag("clip.has_vision_encoder"),
        flag("clip.has_audio_encoder"),
    );
    if v || a { (v, a) } else { (true, false) }
}

impl Projector {
    pub fn vision(g: &Gguf) -> Self {
        let kind = projector_type(g, "vision");
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
        let (known, n_tokens, tokens_basis, warmup_positions) = match rule(&kind) {
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
            n_tokens,
            tokens_basis,
            warmup_positions,
            n_embd,
            n_head,
            n_ff,
        }
    }

    /// The audio encoder. llama.cpp warms it up with `warmup_audio_size` mel frames (3000:
    /// one 30 s chunk at 100 frames/s, the size mtmd-audio.cpp splits longer audio into).
    pub fn audio(g: &Gguf) -> Self {
        let kind = projector_type(g, "audio");
        let n_embd = g.get_u32("clip.audio.embedding_length").unwrap_or(1280);
        let n_ff = g
            .get_u32("clip.audio.feed_forward_length")
            .unwrap_or(n_embd * 4);
        let n_head = g
            .get_u32("clip.audio.attention.head_count")
            .unwrap_or((n_embd / 64).max(1));
        let stack = g
            .get_u32("clip.audio.projector.stack_factor")
            .unwrap_or(1)
            .max(1);
        // Whisper-style encoders halve the frames in their first convolution.
        const F: u32 = 3000;
        let half = F / 2;
        let (known, n_tokens, tokens_basis, warmup_positions) = match kind.as_str() {
            "ultravox" | "meralion" => (
                true,
                F.div_ceil(stack) / 2,
                format!("{F} mel frames / stack {stack} / 2"),
                half,
            ),
            "voxtral" => (
                true,
                F.div_ceil(stack) / 4,
                format!("{F} mel frames / stack {stack} / 2 / 2 (avg pool)"),
                half,
            ),
            "qwen2a" | "musicflamingo" => (
                true,
                F / 4,
                format!("{F} mel frames / 2 / 2 (avg pool)"),
                half,
            ),
            "qwen3a" => {
                let t = F / 100 * 13;
                (
                    true,
                    t,
                    format!("{F} mel frames in 100-frame chunks x 13"),
                    t,
                )
            }
            "glma" => (
                true,
                F / 2 / stack + 2,
                format!("{F} mel frames / 2 / stack {stack} + 2"),
                half,
            ),
            "voxtral_rt" => {
                // Streaming: 2 context frames + 8 s of new frames per step.
                let f = 2 + 2 * 400;
                (
                    true,
                    (f - 2) / 2 / stack,
                    format!("{f} mel frames per streaming step / 2 / stack {stack}"),
                    (f - 2) / 2,
                )
            }
            "lfm2a" | "d1omni_a" => {
                let t = F.div_ceil(2).div_ceil(2).div_ceil(2);
                (
                    true,
                    t,
                    format!("{F} mel frames / 8 (three stride-2 convolutions)"),
                    t,
                )
            }
            "gemma4a" => {
                let t = ((F - 1) / 2) / 2 + 1;
                (
                    true,
                    t,
                    format!("{F} mel frames / 4 (two stride-2 convolutions)"),
                    t,
                )
            }
            _ => (
                false,
                half,
                format!("{F} mel frames / 2, assumed for unknown audio projector `{kind}`"),
                half,
            ),
        };
        Projector {
            kind,
            known,
            n_tokens,
            tokens_basis,
            warmup_positions,
            n_embd,
            n_head,
            n_ff,
        }
    }

    /// Estimate of the encoder's compute buffer, reserved for the warmup input. clip runs
    /// flash attention whenever the backend supports it (`CLIP_FLASH_ATTN_TYPE_AUTO`, the
    /// default), so this is the language model's flash-attention compute shape (ADR 0005)
    /// with the warmup positions as one batch: mask, plus the peak of the FFN and the
    /// attention projections. Unvalidated: no observed run yet.
    pub fn compute_bytes(&self) -> u64 {
        let p = u64::from(self.warmup_positions);
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
        Projector::vision(&Gguf::parse_bytes(&w.finish_header_only()).unwrap())
    }

    /// A projector header from string and integer keys.
    fn header(strs: &[(&str, &str)], ints: &[(&str, u32)], flags: &[&str]) -> Gguf {
        let mut w = GgufWriter::new();
        w.kv("general.architecture", Value::String("clip".into()));
        for (k, v) in strs {
            w.kv(k, Value::String((*v).into()));
        }
        for (k, v) in ints {
            w.kv(k, Value::U32(*v));
        }
        for k in flags {
            w.kv(k, Value::Bool(true));
        }
        Gguf::parse_bytes(&w.finish_header_only()).unwrap()
    }

    /// The Whisper-large encoder of ggml-org's Ultravox and Voxtral projectors (headers
    /// read 2026-10-20).
    const WHISPER: [(&str, u32); 3] = [
        ("clip.audio.embedding_length", 1280),
        ("clip.audio.feed_forward_length", 5120),
        ("clip.audio.attention.head_count", 20),
    ];

    fn audio(kind: &str, stack: Option<u32>) -> Projector {
        let mut ints = WHISPER.to_vec();
        ints.extend(stack.map(|s| ("clip.audio.projector.stack_factor", s)));
        let g = header(
            &[("clip.projector_type", kind)],
            &ints,
            &["clip.has_audio_encoder"],
        );
        assert_eq!(modalities(&g), (false, true));
        Projector::audio(&g)
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
        assert_eq!(p.n_tokens, 256);
        assert_eq!(p.warmup_positions, 64 * 64);
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
        assert_eq!(p.n_tokens, 576);
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
        assert_eq!(p.n_tokens, 4096);
        assert_eq!(p.warmup_positions, 46 * 46 * 4);
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
        assert_eq!(p.n_tokens, 64);
        let p = projector("resampler", &[("clip.minicpmv_query_num", 96)]);
        assert_eq!(p.n_tokens, 96);
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
        assert_eq!(p.n_tokens, 1024);
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

    #[test]
    fn whisper_style_audio_follows_clip_n_output_tokens() {
        let u = audio("ultravox", Some(8));
        assert!(u.known);
        assert_eq!(u.n_tokens, 187);
        assert_eq!(u.warmup_positions, 1500);
        assert_eq!((u.n_embd, u.n_ff, u.n_head), (1280, 5120, 20));
        let v = audio("voxtral", Some(4));
        assert_eq!(v.n_tokens, 187);
        assert!(v.tokens_basis.contains("avg pool"));
        assert_eq!(audio("qwen2a", None).n_tokens, 750);
        assert_eq!(audio("qwen3a", None).n_tokens, 390);
        assert_eq!(audio("glma", Some(4)).n_tokens, 377);
        assert_eq!(audio("lfm2a", None).n_tokens, 375);
        assert_eq!(audio("voxtral_rt", Some(4)).n_tokens, 100);
    }

    #[test]
    fn unknown_audio_projectors_assume_whisper_frames() {
        let p = audio("granite_speech", None);
        assert!(!p.known);
        assert_eq!(p.n_tokens, 1500);
        assert!(p.tokens_basis.contains("granite_speech"));
    }

    #[test]
    fn gemma4_holds_both_encoders_with_their_own_types() {
        // ggml-org/gemma-4-E2B-it-GGUF's projector: per-modality type keys.
        let g = header(
            &[
                ("clip.vision.projector_type", "gemma4v"),
                ("clip.audio.projector_type", "gemma4a"),
            ],
            &[
                ("clip.vision.image_size", 224),
                ("clip.vision.patch_size", 16),
                ("clip.audio.embedding_length", 1024),
            ],
            &["clip.has_vision_encoder", "clip.has_audio_encoder"],
        );
        assert_eq!(modalities(&g), (true, true));
        assert_eq!(Projector::vision(&g).kind, "gemma4v");
        let a = Projector::audio(&g);
        assert_eq!((a.kind.as_str(), a.n_tokens), ("gemma4a", 750));
    }

    #[test]
    fn qwen_omni_splits_into_its_vision_and_audio_types() {
        let g = header(
            &[("clip.projector_type", "qwen2.5o")],
            &[("clip.vision.patch_size", 14)],
            &["clip.has_vision_encoder", "clip.has_audio_encoder"],
        );
        let v = Projector::vision(&g);
        assert_eq!(v.kind, "qwen2.5vl_merger");
        assert!(v.known);
        let a = Projector::audio(&g);
        assert_eq!((a.kind.as_str(), a.n_tokens), ("qwen2a", 750));
    }

    #[test]
    fn files_without_modality_flags_are_vision() {
        let g = header(&[("clip.projector_type", "mlp")], &[], &[]);
        assert_eq!(modalities(&g), (true, false));
    }
}
