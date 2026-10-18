//! Synthetic Hugging Face model directories: real `config.json` shapes and header-only
//! safetensors files whose tensor tables match the real checkpoints.

use std::collections::BTreeMap;

use serde_json::{Value, json};

/// A tensor: name, dtype, shape. Bytes follow from dtype and shape.
type T = (String, &'static str, Vec<u64>);

fn dtype_bytes(dtype: &str) -> u64 {
    match dtype {
        "F32" | "I32" => 4,
        "BF16" | "F16" | "I16" => 2,
        "F8_E4M3" | "U8" | "I8" => 1,
        d => panic!("unknown dtype {d}"),
    }
}

/// Header-only safetensors bytes for `tensors`, packed in order.
fn safetensors(tensors: &[T]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0u64;
    for (name, dtype, shape) in tensors {
        let n = shape.iter().product::<u64>() * dtype_bytes(dtype);
        header.insert(
            name.clone(),
            json!({ "dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n] }),
        );
        offset += n;
    }
    header.insert("__metadata__".into(), json!({ "format": "pt" }));
    let mut text = serde_json::to_vec(&Value::Object(header)).unwrap();
    while text.len() % 8 != 0 {
        text.push(b' ');
    }
    let mut out = (text.len() as u64).to_le_bytes().to_vec();
    out.extend(text);
    out
}

fn dense_layer(prefix: &str, h: u64, q: u64, kv: u64, ff: u64, dt: &'static str) -> Vec<T> {
    let p = |n: &str| format!("{prefix}.{n}");
    vec![
        (p("input_layernorm.weight"), dt, vec![h]),
        (p("self_attn.q_proj.weight"), dt, vec![q, h]),
        (p("self_attn.k_proj.weight"), dt, vec![kv, h]),
        (p("self_attn.v_proj.weight"), dt, vec![kv, h]),
        (p("self_attn.o_proj.weight"), dt, vec![h, q]),
        (p("post_attention_layernorm.weight"), dt, vec![h]),
        (p("mlp.gate_proj.weight"), dt, vec![ff, h]),
        (p("mlp.up_proj.weight"), dt, vec![ff, h]),
        (p("mlp.down_proj.weight"), dt, vec![h, ff]),
    ]
}

/// AWQ GEMM packing: int32 `qweight [in, out/8]`, `qzeros [in/g, out/8]`, f16 `scales [in/g, out]`.
fn awq_linear(name: &str, input: u64, output: u64) -> Vec<T> {
    let g = 128;
    vec![
        (format!("{name}.qweight"), "I32", vec![input, output / 8]),
        (format!("{name}.qzeros"), "I32", vec![input / g, output / 8]),
        (format!("{name}.scales"), "F16", vec![input / g, output]),
    ]
}

pub struct Dir {
    pub name: &'static str,
    pub files: BTreeMap<String, Vec<u8>>,
}

fn json_file(v: &Value) -> Vec<u8> {
    let mut s = serde_json::to_string_pretty(v).unwrap();
    s.push('\n');
    s.into_bytes()
}

/// Split `all` into `n` shards named like transformers does, plus the index file.
fn sharded(all: &[T], n: usize) -> BTreeMap<String, Vec<u8>> {
    let per = all.len().div_ceil(n);
    let mut files = BTreeMap::new();
    let mut weight_map = serde_json::Map::new();
    for (i, chunk) in all.chunks(per).enumerate() {
        let file = format!("model-{:05}-of-{n:05}.safetensors", i + 1);
        for (name, ..) in chunk {
            weight_map.insert(name.clone(), json!(file));
        }
        files.insert(file, safetensors(chunk));
    }
    files.insert(
        "model.safetensors.index.json".into(),
        json_file(&json!({ "metadata": {}, "weight_map": weight_map })),
    );
    files
}

fn llama_bf16_sharded() -> Dir {
    let (h, q, kv, ff, vocab, n_layer) = (4096, 4096, 1024, 14336, 128256, 32);
    let mut all: Vec<T> = vec![("model.embed_tokens.weight".into(), "BF16", vec![vocab, h])];
    for il in 0..n_layer {
        all.extend(dense_layer(
            &format!("model.layers.{il}"),
            h,
            q,
            kv,
            ff,
            "BF16",
        ));
    }
    all.push(("model.norm.weight".into(), "BF16", vec![h]));
    all.push(("lm_head.weight".into(), "BF16", vec![vocab, h]));
    // Four shards, split by tensor count like the real checkpoint.
    let mut files = sharded(&all, 4);
    files.insert(
        "config.json".into(),
        json_file(&json!({
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": h, "intermediate_size": ff,
            "num_hidden_layers": n_layer, "num_attention_heads": 32, "num_key_value_heads": 8,
            "max_position_embeddings": 131072, "vocab_size": vocab,
            "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "rope_theta": 500000.0
        })),
    );
    files.insert(
        "tokenizer_config.json".into(),
        json_file(&json!({ "chat_template": "{{ bos_token }}{% for m in messages %}<|start_header_id|>{{ m.role }}<|end_header_id|>\n\n{{ m.content }}<|eot_id|>{% endfor %}" })),
    );
    Dir {
        name: "Meta-Llama-3.1-8B-Instruct",
        files,
    }
}

fn qwen3_awq() -> Dir {
    let (h, q, kv, ff, vocab, n_layer) = (4096, 4096, 1024, 12288, 151936, 36);
    let mut all: Vec<T> = vec![("model.embed_tokens.weight".into(), "F16", vec![vocab, h])];
    for il in 0..n_layer {
        let p = format!("model.layers.{il}");
        all.push((format!("{p}.input_layernorm.weight"), "F16", vec![h]));
        all.push((
            format!("{p}.post_attention_layernorm.weight"),
            "F16",
            vec![h],
        ));
        all.push((format!("{p}.self_attn.q_norm.weight"), "F16", vec![128]));
        all.push((format!("{p}.self_attn.k_norm.weight"), "F16", vec![128]));
        all.extend(awq_linear(&format!("{p}.self_attn.q_proj"), h, q));
        all.extend(awq_linear(&format!("{p}.self_attn.k_proj"), h, kv));
        all.extend(awq_linear(&format!("{p}.self_attn.v_proj"), h, kv));
        all.extend(awq_linear(&format!("{p}.self_attn.o_proj"), q, h));
        all.extend(awq_linear(&format!("{p}.mlp.gate_proj"), h, ff));
        all.extend(awq_linear(&format!("{p}.mlp.up_proj"), h, ff));
        all.extend(awq_linear(&format!("{p}.mlp.down_proj"), ff, h));
    }
    all.push(("model.norm.weight".into(), "F16", vec![h]));
    all.push(("lm_head.weight".into(), "F16", vec![vocab, h]));
    let mut files = sharded(&all, 2);
    files.insert(
        "config.json".into(),
        json_file(&json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": h, "intermediate_size": ff, "head_dim": 128,
            "num_hidden_layers": n_layer, "num_attention_heads": 32, "num_key_value_heads": 8,
            "max_position_embeddings": 40960, "vocab_size": vocab,
            "tie_word_embeddings": false, "torch_dtype": "float16",
            "sliding_window": null, "use_sliding_window": false,
            "quantization_config": { "quant_method": "awq", "bits": 4, "group_size": 128, "version": "gemm", "zero_point": true }
        })),
    );
    files.insert(
        "tokenizer_config.json".into(),
        json_file(&json!({ "chat_template": "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}" })),
    );
    Dir {
        name: "Qwen3-8B-AWQ",
        files,
    }
}

fn gemma3_multimodal() -> Dir {
    let (h, q, kv, ff, vocab, n_layer) = (3840, 4096, 2048, 15360, 262208, 48);
    let mut all: Vec<T> = vec![(
        "model.language_model.embed_tokens.weight".into(),
        "BF16",
        vec![vocab, h],
    )];
    for il in 0..n_layer {
        let p = format!("model.language_model.layers.{il}");
        all.extend(dense_layer(&p, h, q, kv, ff, "BF16"));
        all.push((
            format!("{p}.pre_feedforward_layernorm.weight"),
            "BF16",
            vec![h],
        ));
        all.push((
            format!("{p}.post_feedforward_layernorm.weight"),
            "BF16",
            vec![h],
        ));
        all.push((format!("{p}.self_attn.q_norm.weight"), "BF16", vec![256]));
        all.push((format!("{p}.self_attn.k_norm.weight"), "BF16", vec![256]));
    }
    all.push(("model.language_model.norm.weight".into(), "BF16", vec![h]));
    let (vh, vff) = (1152, 4304);
    all.push((
        "model.vision_tower.vision_model.embeddings.patch_embedding.weight".into(),
        "BF16",
        vec![vh, 3, 14, 14],
    ));
    for il in 0..27 {
        let p = format!("model.vision_tower.vision_model.encoder.layers.{il}");
        for n in ["q_proj", "k_proj", "v_proj", "out_proj"] {
            all.push((format!("{p}.self_attn.{n}.weight"), "BF16", vec![vh, vh]));
        }
        all.push((format!("{p}.mlp.fc1.weight"), "BF16", vec![vff, vh]));
        all.push((format!("{p}.mlp.fc2.weight"), "BF16", vec![vh, vff]));
    }
    all.push((
        "model.multi_modal_projector.mm_input_projection_weight".into(),
        "BF16",
        vec![vh, h],
    ));
    let mut files = sharded(&all, 5);
    files.insert(
        "config.json".into(),
        json_file(&json!({
            "architectures": ["Gemma3ForConditionalGeneration"],
            "model_type": "gemma3",
            "mm_tokens_per_image": 256,
            "text_config": {
                "model_type": "gemma3_text",
                "hidden_size": h, "intermediate_size": ff, "head_dim": 256,
                "num_hidden_layers": n_layer, "num_attention_heads": 16, "num_key_value_heads": 8,
                "max_position_embeddings": 131072, "vocab_size": vocab,
                "sliding_window": 1024, "sliding_window_pattern": 6
            },
            "vision_config": { "model_type": "siglip_vision_model", "image_size": 896, "patch_size": 14, "hidden_size": vh },
            "torch_dtype": "bfloat16"
        })),
    );
    files.insert(
        "chat_template.jinja".into(),
        b"{% for m in messages %}<start_of_turn>{{ m.role }}\n{{ m.content }}<end_of_turn>\n{% endfor %}".to_vec(),
    );
    Dir {
        name: "gemma-3-12b-it",
        files,
    }
}

/// EXL3 linear: an int16 trellis of `in/16 x out/16 x 16*bits` plus f16 input/output
/// sign vectors, i.e. `bits / 8` bytes per weight.
fn exl3_linear(name: &str, input: u64, output: u64, bits: u64) -> Vec<T> {
    vec![
        (
            format!("{name}.trellis"),
            "I16",
            vec![input / 16, output / 16, 16 * bits],
        ),
        (format!("{name}.suh"), "F16", vec![input]),
        (format!("{name}.svh"), "F16", vec![output]),
    ]
}

fn qwen3_exl3() -> Dir {
    let (h, q, kv, ff, vocab, n_layer, bits) = (4096, 4096, 1024, 12288, 151936, 36, 4);
    let mut all: Vec<T> = vec![("model.embed_tokens.weight".into(), "F16", vec![vocab, h])];
    for il in 0..n_layer {
        let p = format!("model.layers.{il}");
        all.push((format!("{p}.input_layernorm.weight"), "F16", vec![h]));
        all.push((
            format!("{p}.post_attention_layernorm.weight"),
            "F16",
            vec![h],
        ));
        all.extend(exl3_linear(&format!("{p}.self_attn.q_proj"), h, q, bits));
        all.extend(exl3_linear(&format!("{p}.self_attn.k_proj"), h, kv, bits));
        all.extend(exl3_linear(&format!("{p}.self_attn.v_proj"), h, kv, bits));
        all.extend(exl3_linear(&format!("{p}.self_attn.o_proj"), q, h, bits));
        all.extend(exl3_linear(&format!("{p}.mlp.gate_proj"), h, ff, bits));
        all.extend(exl3_linear(&format!("{p}.mlp.up_proj"), h, ff, bits));
        all.extend(exl3_linear(&format!("{p}.mlp.down_proj"), ff, h, bits));
    }
    all.push(("model.norm.weight".into(), "F16", vec![h]));
    all.extend(exl3_linear("lm_head", h, vocab, 6));
    let mut files = sharded(&all, 2);
    files.insert(
        "config.json".into(),
        json_file(&json!({
            "architectures": ["Qwen3ForCausalLM"],
            "model_type": "qwen3",
            "hidden_size": h, "intermediate_size": ff, "head_dim": 128,
            "num_hidden_layers": n_layer, "num_attention_heads": 32, "num_key_value_heads": 8,
            "max_position_embeddings": 40960, "vocab_size": vocab,
            "tie_word_embeddings": false, "torch_dtype": "bfloat16",
            "quantization_config": { "quant_method": "exl3", "bits": 4.0, "head_bits": 6, "calibration": { "rows": 250, "cols": 2048 } }
        })),
    );
    Dir {
        name: "Qwen3-8B-exl3-4.0bpw",
        files,
    }
}

pub fn dirs() -> Vec<Dir> {
    vec![
        llama_bf16_sharded(),
        qwen3_awq(),
        gemma3_multimodal(),
        qwen3_exl3(),
    ]
}
