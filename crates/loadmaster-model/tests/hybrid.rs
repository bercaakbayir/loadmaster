//! Hybrid models (Qwen3.5, Qwen3-Next): which layers are recurrent, and their state.

use std::path::{Path, PathBuf};

use loadmaster_model::gguf::writer::GgufWriter;
use loadmaster_model::gguf::{GgmlType, Value};
use loadmaster_model::{LayerKind, ModelInfo};

/// Conv state 3 x 8192 plus recurrent state 128 x 4096, f32 (llama.cpp `n_embd_r`,
/// `n_embd_s`) for Qwen3.5-9B.
const QWEN35_9B_STATE: u64 = (3 * 8192 + 128 * 4096) * 4;

/// The metadata of Lucebox/Qwen3.5-9B-GGUF `Qwen3.5-9B-Q8_0.gguf`, with one tensor of
/// each layer kind.
fn qwen35_9b_header() -> Vec<u8> {
    let mut w = GgufWriter::new();
    let u = |n| Value::U32(n);
    w.kv("general.architecture", Value::String("qwen35".into()));
    w.kv("qwen35.block_count", u(32));
    w.kv("qwen35.context_length", u(262_144));
    w.kv("qwen35.embedding_length", u(4096));
    w.kv("qwen35.feed_forward_length", u(12_288));
    w.kv("qwen35.attention.head_count", u(16));
    w.kv("qwen35.attention.head_count_kv", u(4));
    w.kv("qwen35.attention.key_length", u(256));
    w.kv("qwen35.attention.value_length", u(256));
    w.kv("qwen35.ssm.conv_kernel", u(4));
    w.kv("qwen35.ssm.state_size", u(128));
    w.kv("qwen35.ssm.group_count", u(16));
    w.kv("qwen35.ssm.time_step_rank", u(32));
    w.kv("qwen35.ssm.inner_size", u(4096));
    w.kv("qwen35.full_attention_interval", u(4));
    w.tensor("token_embd.weight", &[4096, 248_320], GgmlType::Q8_0);
    w.tensor("blk.0.ssm_conv1d.weight", &[4, 8192], GgmlType::F32);
    w.tensor("blk.3.attn_k.weight", &[4096, 1024], GgmlType::Q8_0);
    w.finish_header_only()
}

fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
    let path = dir.join(name);
    std::fs::write(&path, bytes).unwrap();
    path
}

fn assert_qwen35_9b(info: &ModelInfo) {
    let kinds: Vec<LayerKind> = info.layers.iter().map(|l| l.kind()).collect();
    let attention: Vec<usize> = (0..32)
        .filter(|&il| kinds[il] == LayerKind::Attention)
        .collect();
    assert_eq!(attention, [3, 7, 11, 15, 19, 23, 27, 31]);
    assert_eq!(info.n_recurrent_layers(), 24);
    assert!(!kinds.contains(&LayerKind::Sliding));
    for l in &info.layers {
        if l.is_recurrent() {
            assert_eq!(
                (l.k_dim, l.v_dim, l.recurrent_bytes),
                (0, 0, QWEN35_9B_STATE)
            );
        } else {
            assert_eq!((l.k_dim, l.v_dim, l.recurrent_bytes), (1024, 1024, 0));
        }
    }
    assert_eq!(info.recurrent_state_bytes(), 24 * QWEN35_9B_STATE);
    assert!(info.arch_known);
    assert!(info.has_kv_cache());
}

#[test]
fn qwen35_gguf_has_8_attention_and_24_recurrent_layers() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), "Qwen3.5-9B-Q8_0.gguf", &qwen35_9b_header());
    assert_qwen35_9b(&ModelInfo::from_gguf(path).unwrap());
}

/// The real header, when `lm plan hf://Lucebox/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf` has
/// cached it.
#[test]
#[ignore = "needs the Qwen3.5-9B header in ~/.loadmaster/hf"]
fn qwen35_9b_cached_header() {
    let home = std::env::var_os("HOME").expect("HOME");
    let path = Path::new(&home)
        .join(".loadmaster/hf/Lucebox/Qwen3.5-9B-GGUF/main/Qwen3.5-9B-Q8_0.gguf.header");
    let g = loadmaster_model::gguf::Gguf::open(&path).unwrap();
    let info = ModelInfo::from_gguf_parts("Qwen3.5-9B-Q8_0.gguf", &[g]).unwrap();
    assert_qwen35_9b(&info);
}

#[test]
fn explicit_recurrent_layers_flags_win_over_the_interval() {
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String("qwen3next".into()));
    w.kv("qwen3next.block_count", Value::U32(4));
    w.kv("qwen3next.context_length", Value::U32(4096));
    w.kv("qwen3next.embedding_length", Value::U32(256));
    w.kv("qwen3next.attention.head_count", Value::U32(4));
    w.kv("qwen3next.full_attention_interval", Value::U32(4));
    w.kv(
        "qwen3next.attention.recurrent_layers",
        Value::Array(vec![
            Value::Bool(false),
            Value::Bool(true),
            Value::Bool(true),
            Value::Bool(false),
        ]),
    );
    for (k, v) in [
        ("conv_kernel", 4),
        ("state_size", 16),
        ("group_count", 2),
        ("inner_size", 64),
    ] {
        w.kv(&format!("qwen3next.ssm.{k}"), Value::U32(v));
    }
    w.tensor("token_embd.weight", &[256, 1000], GgmlType::F16);
    let dir = tempfile::tempdir().unwrap();
    let info = ModelInfo::from_gguf(write(dir.path(), "m.gguf", &w.finish_header_only())).unwrap();
    let recurrent: Vec<bool> = info.layers.iter().map(|l| l.is_recurrent()).collect();
    assert_eq!(recurrent, [false, true, true, false]);
    assert_eq!(
        info.layers[1].recurrent_bytes,
        (3 * (64 + 2 * 2 * 16) + 16 * 64) * 4
    );
}

#[test]
fn hybrid_gguf_without_ssm_keys_is_an_error() {
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String("qwen35".into()));
    w.kv("qwen35.block_count", Value::U32(4));
    w.kv("qwen35.context_length", Value::U32(4096));
    w.kv("qwen35.embedding_length", Value::U32(256));
    w.kv("qwen35.attention.head_count", Value::U32(4));
    w.tensor("token_embd.weight", &[256, 1000], GgmlType::F16);
    let dir = tempfile::tempdir().unwrap();
    let err =
        ModelInfo::from_gguf(write(dir.path(), "m.gguf", &w.finish_header_only())).unwrap_err();
    assert!(err.to_string().contains("qwen35.ssm."), "{err}");
}

#[test]
fn llama_has_no_recurrent_layers() {
    let info = ModelInfo::from_gguf(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/gguf/llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
    )
    .unwrap();
    assert_eq!(info.n_recurrent_layers(), 0);
    assert_eq!(info.recurrent_state_bytes(), 0);
    assert!(info.layers.iter().all(|l| l.kind() == LayerKind::Attention));
}

#[test]
fn gemma3_sliding_layers_are_sliding() {
    let info = ModelInfo::from_gguf(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/gguf/gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf"),
    )
    .unwrap();
    assert_eq!(info.n_recurrent_layers(), 0);
    assert_eq!(info.layers[0].kind(), LayerKind::Sliding);
    assert_eq!(info.layers[5].kind(), LayerKind::Attention);
}

/// A config.json shaped like Qwen/Qwen3.5-9B (the LM nested in `text_config`).
#[test]
fn qwen35_safetensors_reads_layer_types_and_linear_keys() {
    let layer_types: Vec<&str> = (0..32)
        .map(|il| {
            if (il + 1) % 4 == 0 {
                "full_attention"
            } else {
                "linear_attention"
            }
        })
        .collect();
    let config = serde_json::json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "num_hidden_layers": 32,
            "hidden_size": 4096,
            "intermediate_size": 12288,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "max_position_embeddings": 262144,
            "vocab_size": 248320,
            "layer_types": layer_types,
            "full_attention_interval": 4,
            "linear_conv_kernel_dim": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_num_key_heads": 16,
            "linear_num_value_heads": 32
        }
    });
    let header = serde_json::json!({
        "model.language_model.embed_tokens.weight":
            {"dtype": "BF16", "shape": [248320, 4096], "data_offsets": [0, 0]}
    })
    .to_string();
    let mut st = (header.len() as u64).to_le_bytes().to_vec();
    st.extend_from_slice(header.as_bytes());

    let dir = tempfile::tempdir().unwrap();
    write(dir.path(), "config.json", config.to_string().as_bytes());
    write(dir.path(), "model.safetensors", &st);
    let info = ModelInfo::open(dir.path()).unwrap();
    assert_eq!(info.arch, "qwen35");
    assert_qwen35_9b(&info);
}
