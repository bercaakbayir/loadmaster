//! Synthetic header-only GGUF fixtures (and, via `hf_fixtures`, HF model directories) shaped
//! like real models. The tensor tables match
//! the real architectures closely enough for footprint arithmetic; tensor types approximate
//! each quant mix. Regenerate with `cargo xtask gen-fixtures`; CI runs `--check`.

use std::path::Path;

use anyhow::{Context, bail};
use loadmaster_model::gguf::GgmlType::{self, *};
use loadmaster_model::gguf::Value;
use loadmaster_model::gguf::writer::GgufWriter;

const CHATML: &str = "{% for message in messages %}<|im_start|>{{ message.role }}\n{{ message.content }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";
const LLAMA3: &str = "{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}";
const GEMMA: &str = "{% for message in messages %}<start_of_turn>{{ 'model' if message.role == 'assistant' else message.role }}\n{{ message.content }}<end_of_turn>\n{% endfor %}{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}";

/// Tensor types for one quant mix: `main` for most matrices, `hi` where llama.cpp's
/// `_M` mixes keep more bits (attn_v, ffn_down), and the embedding/output types.
#[derive(Clone, Copy)]
struct Mix {
    ftype: u32,
    main: GgmlType,
    hi: GgmlType,
    experts: GgmlType,
    embd: GgmlType,
    output: GgmlType,
}

const Q4_K_M: Mix = Mix {
    ftype: 15,
    main: Q4_K,
    hi: Q6_K,
    experts: Q4_K,
    embd: Q4_K,
    output: Q6_K,
};

const MXFP4_MOE: Mix = Mix {
    ftype: 38,
    main: Q8_0,
    hi: Q8_0,
    experts: MXFP4,
    embd: Q8_0,
    output: Q8_0,
};

struct Moe {
    n_expert: u64,
    n_used: u32,
    n_ff_exp: u64,
    n_shared: u64,
    leading_dense: u32,
    biases: bool,
}

struct Mla {
    kv_lora_rank: u64,
    rope: u64,
    qk_nope: u64,
    v_head: u64,
}

struct Spec {
    dir: &'static str,
    file: &'static str,
    arch: &'static str,
    name: &'static str,
    n_layer: u32,
    n_embd: u64,
    n_head: u64,
    n_head_kv: u64,
    head_dim: u64,
    n_ff: u64,
    n_vocab: u64,
    n_ctx: u32,
    mix: Mix,
    tied: bool,
    qk_norm: bool,
    post_norms: bool,
    sinks: bool,
    swa_window: Option<u32>,
    moe: Option<Moe>,
    mla: Option<Mla>,
    template: &'static str,
}

fn specs() -> Vec<Spec> {
    let dense = |dir, file, arch, name| Spec {
        dir,
        file,
        arch,
        name,
        n_layer: 0,
        n_embd: 0,
        n_head: 0,
        n_head_kv: 0,
        head_dim: 0,
        n_ff: 0,
        n_vocab: 0,
        n_ctx: 0,
        mix: Q4_K_M,
        tied: false,
        qk_norm: false,
        post_norms: false,
        sinks: false,
        swa_window: None,
        moe: None,
        mla: None,
        template: CHATML,
    };
    vec![
        Spec {
            n_layer: 36,
            n_embd: 4096,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            n_ff: 12288,
            n_vocab: 151936,
            n_ctx: 40960,
            qk_norm: true,
            ..dense("qwen3-8b", "Qwen3-8B-Q4_K_M.gguf", "qwen3", "Qwen3 8B")
        },
        Spec {
            n_layer: 32,
            n_embd: 4096,
            n_head: 32,
            n_head_kv: 8,
            head_dim: 128,
            n_ff: 14336,
            n_vocab: 128256,
            n_ctx: 131072,
            template: LLAMA3,
            ..dense(
                "llama-3.1-8b",
                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "llama",
                "Meta Llama 3.1 8B Instruct",
            )
        },
        Spec {
            n_layer: 48,
            n_embd: 2048,
            n_head: 32,
            n_head_kv: 4,
            head_dim: 128,
            n_ff: 6144,
            n_vocab: 151936,
            n_ctx: 40960,
            qk_norm: true,
            moe: Some(Moe {
                n_expert: 128,
                n_used: 8,
                n_ff_exp: 768,
                n_shared: 0,
                leading_dense: 0,
                biases: false,
            }),
            ..dense(
                "qwen3-30b-a3b",
                "Qwen3-30B-A3B-Q4_K_M.gguf",
                "qwen3moe",
                "Qwen3 30B A3B",
            )
        },
        Spec {
            n_layer: 48,
            n_embd: 3840,
            n_head: 16,
            n_head_kv: 8,
            head_dim: 256,
            n_ff: 15360,
            n_vocab: 262144,
            n_ctx: 131072,
            tied: true,
            qk_norm: true,
            post_norms: true,
            swa_window: Some(1024),
            template: GEMMA,
            ..dense(
                "gemma-3-12b",
                "gemma-3-12b-it-Q4_K_M.gguf",
                "gemma3",
                "Gemma 3 12B It",
            )
        },
        Spec {
            n_layer: 27,
            n_embd: 2048,
            n_head: 16,
            n_head_kv: 16,
            head_dim: 192,
            n_ff: 10944,
            n_vocab: 102400,
            n_ctx: 163840,
            moe: Some(Moe {
                n_expert: 64,
                n_used: 6,
                n_ff_exp: 1408,
                n_shared: 2,
                leading_dense: 1,
                biases: false,
            }),
            mla: Some(Mla {
                kv_lora_rank: 512,
                rope: 64,
                qk_nope: 128,
                v_head: 128,
            }),
            ..dense(
                "deepseek-v2-lite",
                "DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
                "deepseek2",
                "DeepSeek V2 Lite Chat",
            )
        },
        Spec {
            n_layer: 24,
            n_embd: 2880,
            n_head: 64,
            n_head_kv: 8,
            head_dim: 64,
            n_ff: 2880,
            n_vocab: 201088,
            n_ctx: 131072,
            mix: MXFP4_MOE,
            sinks: true,
            swa_window: Some(128),
            moe: Some(Moe {
                n_expert: 32,
                n_used: 4,
                n_ff_exp: 2880,
                n_shared: 0,
                leading_dense: 0,
                biases: true,
            }),
            ..dense(
                "gpt-oss-20b",
                "gpt-oss-20b-mxfp4.gguf",
                "gpt-oss",
                "Gpt Oss 20b",
            )
        },
    ]
}

/// The type a quantizer would actually use for a row of `n`: fall back from k-quants to
/// Q8_0 and then F16 when the row is not a whole number of blocks.
fn fit(ty: GgmlType, n: u64) -> GgmlType {
    [ty, Q8_0, F16]
        .into_iter()
        .find(|t| t.row_bytes(n).is_some())
        .unwrap_or(F32)
}

fn model(s: &Spec) -> Vec<u8> {
    let mut w = GgufWriter::new();
    let a = s.arch;
    let k = |key: &str| format!("{a}.{key}");
    w.kv("general.architecture", Value::String(a.into()));
    w.kv("general.name", Value::String(s.name.into()));
    w.kv("general.file_type", Value::U32(s.mix.ftype));
    w.kv(&k("block_count"), Value::U32(s.n_layer));
    w.kv(&k("context_length"), Value::U32(s.n_ctx));
    w.kv(&k("embedding_length"), Value::U32(s.n_embd as u32));
    w.kv(&k("feed_forward_length"), Value::U32(s.n_ff as u32));
    w.kv(&k("attention.head_count"), Value::U32(s.n_head as u32));
    w.kv(
        &k("attention.head_count_kv"),
        Value::U32(s.n_head_kv as u32),
    );
    let (key_len, value_len) = match &s.mla {
        Some(m) => (m.qk_nope + m.rope, m.v_head),
        None => (s.head_dim, s.head_dim),
    };
    w.kv(&k("attention.key_length"), Value::U32(key_len as u32));
    w.kv(&k("attention.value_length"), Value::U32(value_len as u32));
    if let Some(win) = s.swa_window {
        w.kv(&k("attention.sliding_window"), Value::U32(win));
    }
    if let Some(m) = &s.moe {
        w.kv(&k("expert_count"), Value::U32(m.n_expert as u32));
        w.kv(&k("expert_used_count"), Value::U32(m.n_used));
        w.kv(
            &k("expert_feed_forward_length"),
            Value::U32(m.n_ff_exp as u32),
        );
        if m.n_shared > 0 {
            w.kv(&k("expert_shared_count"), Value::U32(m.n_shared as u32));
        }
        if m.leading_dense > 0 {
            w.kv(&k("leading_dense_block_count"), Value::U32(m.leading_dense));
        }
    }
    if let Some(m) = &s.mla {
        w.kv(
            &k("attention.kv_lora_rank"),
            Value::U32(m.kv_lora_rank as u32),
        );
        w.kv(&k("rope.dimension_count"), Value::U32(m.rope as u32));
    }
    w.kv("tokenizer.chat_template", Value::String(s.template.into()));

    let e = s.n_embd;
    let mix = s.mix;
    let mut t = |name: String, dims: &[u64], ty: GgmlType| {
        w.tensor(&name, dims, fit(ty, dims[0]));
    };
    t("token_embd.weight".into(), &[e, s.n_vocab], mix.embd);
    for il in 0..s.n_layer {
        let b = |n: &str| format!("blk.{il}.{n}");
        t(b("attn_norm.weight"), &[e], F32);
        match &s.mla {
            Some(m) => {
                let h = s.n_head;
                t(b("attn_q.weight"), &[e, h * (m.qk_nope + m.rope)], mix.main);
                t(
                    b("attn_kv_a_mqa.weight"),
                    &[e, m.kv_lora_rank + m.rope],
                    mix.main,
                );
                t(b("attn_kv_a_norm.weight"), &[m.kv_lora_rank], F32);
                t(
                    b("attn_kv_b.weight"),
                    &[m.kv_lora_rank, h * (m.qk_nope + m.v_head)],
                    mix.main,
                );
                t(b("attn_output.weight"), &[h * m.v_head, e], mix.main);
            }
            None => {
                let q = s.n_head * s.head_dim;
                let kv = s.n_head_kv * s.head_dim;
                t(b("attn_q.weight"), &[e, q], mix.main);
                t(b("attn_k.weight"), &[e, kv], mix.main);
                t(b("attn_v.weight"), &[e, kv], mix.hi);
                t(b("attn_output.weight"), &[q, e], mix.main);
                if s.qk_norm {
                    t(b("attn_q_norm.weight"), &[s.head_dim], F32);
                    t(b("attn_k_norm.weight"), &[s.head_dim], F32);
                }
                if s.sinks {
                    t(b("attn_sinks.weight"), &[s.n_head], F32);
                }
            }
        }
        if s.post_norms {
            t(b("post_attention_norm.weight"), &[e], F32);
            t(b("post_ffw_norm.weight"), &[e], F32);
        }
        t(b("ffn_norm.weight"), &[e], F32);
        match &s.moe {
            Some(m) if il >= m.leading_dense => {
                let (x, f) = (m.n_expert, m.n_ff_exp);
                t(b("ffn_gate_inp.weight"), &[e, x], F32);
                t(b("ffn_gate_exps.weight"), &[e, f, x], mix.experts);
                t(b("ffn_up_exps.weight"), &[e, f, x], mix.experts);
                t(b("ffn_down_exps.weight"), &[f, e, x], mix.experts);
                if m.biases {
                    t(b("ffn_gate_inp.bias"), &[x], F32);
                    t(b("ffn_gate_exps.bias"), &[f, x], F32);
                    t(b("ffn_up_exps.bias"), &[f, x], F32);
                    t(b("ffn_down_exps.bias"), &[e, x], F32);
                }
                if m.n_shared > 0 {
                    let sf = f * m.n_shared;
                    t(b("ffn_gate_shexp.weight"), &[e, sf], mix.main);
                    t(b("ffn_up_shexp.weight"), &[e, sf], mix.main);
                    t(b("ffn_down_shexp.weight"), &[sf, e], mix.hi);
                }
            }
            _ => {
                t(b("ffn_gate.weight"), &[e, s.n_ff], mix.main);
                t(b("ffn_up.weight"), &[e, s.n_ff], mix.main);
                t(b("ffn_down.weight"), &[s.n_ff, e], mix.hi);
            }
        }
    }
    t("output_norm.weight".into(), &[e], F32);
    if !s.tied {
        t("output.weight".into(), &[e, s.n_vocab], mix.output);
    }
    w.finish_header_only()
}

/// Gemma 3 SigLIP vision tower and projector, F16.
fn gemma3_mmproj() -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String("clip".into()));
    w.kv("clip.projector_type", Value::String("gemma3".into()));
    w.kv("clip.has_vision_encoder", Value::Bool(true));
    w.kv("clip.vision.image_size", Value::U32(896));
    w.kv("clip.vision.patch_size", Value::U32(14));
    w.kv("clip.vision.embedding_length", Value::U32(1152));
    w.kv("clip.vision.feed_forward_length", Value::U32(4304));
    w.kv("clip.vision.block_count", Value::U32(27));
    w.kv("clip.vision.projection_dim", Value::U32(3840));
    let (e, f) = (1152, 4304);
    w.tensor("v.patch_embd.weight", &[14, 14, 3, e], F16);
    w.tensor("v.position_embd.weight", &[e, 4096], F16);
    for il in 0..27 {
        let b = |n: &str| format!("v.blk.{il}.{n}");
        for n in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            w.tensor(&b(&format!("{n}.weight")), &[e, e], F16);
            w.tensor(&b(&format!("{n}.bias")), &[e], F32);
        }
        w.tensor(&b("ffn_up.weight"), &[e, f], F16);
        w.tensor(&b("ffn_down.weight"), &[f, e], F16);
        w.tensor(&b("ln1.weight"), &[e], F32);
        w.tensor(&b("ln2.weight"), &[e], F32);
    }
    w.tensor("v.post_ln.weight", &[e], F32);
    w.tensor("mm.input_projection.weight", &[e, 3840], F16);
    w.tensor("mm.soft_emb_norm.weight", &[e], F32);
    w.finish_header_only()
}

pub fn generate(root: &Path, check: bool) -> anyhow::Result<()> {
    let mut files: Vec<(String, Vec<u8>)> = specs()
        .iter()
        .map(|s| (format!("gguf/{}/{}", s.dir, s.file), model(s)))
        .collect();
    files.push((
        "gguf/gemma-3-12b/mmproj-gemma-3-12b-it-F16.gguf".into(),
        gemma3_mmproj(),
    ));
    for dir in crate::hf_fixtures::dirs() {
        for (name, bytes) in dir.files {
            files.push((format!("hf/{}/{name}", dir.name), bytes));
        }
    }

    let mut stale = Vec::new();
    for (rel, bytes) in &files {
        let binary = rel.ends_with(".gguf") || rel.ends_with(".safetensors");
        if binary && bytes.len() > 64 * 1024 {
            bail!("{rel}: fixture is {} bytes, limit is 64 KiB", bytes.len());
        }
        let path = root.join(rel);
        if check {
            if std::fs::read(&path).ok().as_deref() != Some(bytes.as_slice()) {
                stale.push(rel.clone());
            }
        } else {
            std::fs::create_dir_all(path.parent().unwrap_or(root))?;
            std::fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
            println!("{rel} ({} bytes)", bytes.len());
        }
    }
    if !stale.is_empty() {
        bail!(
            "fixtures out of date, run `cargo xtask gen-fixtures`:\n  {}",
            stale.join("\n  ")
        );
    }
    if check {
        println!("fixtures up to date");
    }
    Ok(())
}
