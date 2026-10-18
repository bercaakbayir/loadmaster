use std::fmt::Write;
use std::path::PathBuf;

use loadmaster_model::gguf::writer::GgufWriter;
use loadmaster_model::gguf::{GgmlType, Value};
use loadmaster_model::{Error, ModelInfo};

fn fixture(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gguf")
        .join(rel)
}

fn summary(info: &ModelInfo) -> String {
    let gib = |b: u64| format!("{:.3} GiB", b as f64 / (1u64 << 30) as f64);
    let mut s = String::new();
    let l0 = &info.layers[0];
    let last = info.layers.last().unwrap();
    writeln!(s, "name: {:?}", info.name).unwrap();
    writeln!(
        s,
        "arch: {} (known: {}) kv: {:?}",
        info.arch, info.arch_known, info.kv_layout
    )
    .unwrap();
    writeln!(s, "quant: {}  params: {:.2}B", info.quant, info.params_b()).unwrap();
    writeln!(
        s,
        "layers: {}  embd: {}  vocab: {}  heads: {}/{} x {}  ctx_train: {}",
        info.n_layer,
        info.n_embd,
        info.n_vocab,
        info.n_head,
        info.n_head_kv,
        info.head_dim,
        info.n_ctx_train
    )
    .unwrap();
    writeln!(
        s,
        "experts: {:?} used {:?}",
        info.n_expert, info.n_expert_used
    )
    .unwrap();
    writeln!(
        s,
        "swa: window {:?}, {} of {} layers",
        info.swa_window,
        info.layers.iter().filter(|l| l.swa).count(),
        info.n_layer
    )
    .unwrap();
    writeln!(
        s,
        "weights: {}  ({} bytes)",
        gib(info.weight_bytes),
        info.weight_bytes
    )
    .unwrap();
    writeln!(
        s,
        "embd: {}  output: {} (tied: {})  other: {}",
        gib(info.embd_bytes),
        gib(info.output_bytes),
        info.output_tied,
        info.other_bytes
    )
    .unwrap();
    writeln!(
        s,
        "layer 0: {} bytes ({} expert) k_dim {} v_dim {}",
        l0.bytes, l0.expert_bytes, l0.k_dim, l0.v_dim
    )
    .unwrap();
    writeln!(
        s,
        "layer {}: {} bytes ({} expert)",
        info.n_layer - 1,
        last.bytes,
        last.expert_bytes
    )
    .unwrap();
    writeln!(
        s,
        "template blake3: {}",
        info.chat_template
            .as_ref()
            .map_or("-", |t| t.blake3.as_str())
    )
    .unwrap();
    match &info.mmproj {
        Some(m) => writeln!(
            s,
            "mmproj: {} {} tokens ({})",
            gib(m.weight_bytes),
            m.n_img_tokens,
            m.img_tokens_basis
        )
        .unwrap(),
        None => writeln!(s, "mmproj: none").unwrap(),
    }
    writeln!(s, "complete: {}", info.complete).unwrap();
    s
}

macro_rules! fixture_test {
    ($name:ident, $rel:literal) => {
        #[test]
        fn $name() {
            let info = ModelInfo::from_gguf(fixture($rel)).unwrap();
            assert_eq!(info.layers.len(), info.n_layer as usize);
            assert_eq!(
                info.weight_bytes,
                info.layers.iter().map(|l| l.bytes).sum::<u64>()
                    + info.embd_bytes
                    + info.other_bytes
                    + if info.output_tied {
                        info.output_bytes - info.embd_bytes
                    } else {
                        info.output_bytes
                    },
            );
            insta::assert_snapshot!(summary(&info));
        }
    };
}

fixture_test!(qwen3_8b, "qwen3-8b/Qwen3-8B-Q4_K_M.gguf");
fixture_test!(
    llama_3_1_8b,
    "llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"
);
fixture_test!(qwen3_30b_a3b, "qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf");
fixture_test!(gemma_3_12b, "gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf");
fixture_test!(
    deepseek_v2_lite,
    "deepseek-v2-lite/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"
);
fixture_test!(gpt_oss_20b, "gpt-oss-20b/gpt-oss-20b-mxfp4.gguf");

#[test]
fn mmproj_is_not_a_model() {
    let err =
        ModelInfo::from_gguf(fixture("gemma-3-12b/mmproj-gemma-3-12b-it-F16.gguf")).unwrap_err();
    assert!(matches!(err, Error::IsProjector { .. }), "{err}");
}

fn split_part(no: u16, count: u16, tensors: &[&str]) -> Vec<u8> {
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String("llama".into()));
    w.kv("llama.block_count", Value::U32(2));
    w.kv("llama.context_length", Value::U32(4096));
    w.kv("llama.embedding_length", Value::U32(256));
    w.kv("llama.attention.head_count", Value::U32(4));
    w.kv("split.no", Value::U16(no));
    w.kv("split.count", Value::U16(count));
    for name in tensors {
        let dims: &[u64] = if name.starts_with("token_embd") || name.starts_with("output.") {
            &[256, 1000]
        } else {
            &[256, 256]
        };
        w.tensor(name, dims, GgmlType::F16);
    }
    w.finish_with_data()
}

#[test]
fn split_gguf_loads_all_parts_from_any_part() {
    let dir = tempfile::tempdir().unwrap();
    let parts = [
        split_part(0, 3, &["token_embd.weight", "blk.0.attn_q.weight"]),
        split_part(1, 3, &["blk.1.attn_q.weight"]),
        split_part(2, 3, &["output.weight"]),
    ];
    for (i, bytes) in parts.iter().enumerate() {
        std::fs::write(
            dir.path().join(format!("m-{:05}-of-00003.gguf", i + 1)),
            bytes,
        )
        .unwrap();
    }
    let info = ModelInfo::from_gguf(dir.path().join("m-00002-of-00003.gguf")).unwrap();
    assert_eq!(info.files.len(), 3);
    assert_eq!(info.n_vocab, 1000);
    assert_eq!(info.layers[1].bytes, 256 * 256 * 2);
    assert!(!info.output_tied);
    assert!(info.complete);

    std::fs::remove_file(dir.path().join("m-00003-of-00003.gguf")).unwrap();
    let err = ModelInfo::from_gguf(dir.path().join("m-00001-of-00003.gguf")).unwrap_err();
    assert!(err.to_string().contains("part 3 of 3 is missing"), "{err}");
}

fn hf(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/hf")
        .join(name)
}

macro_rules! hf_test {
    ($name:ident, $dir:literal) => {
        #[test]
        fn $name() {
            let info = ModelInfo::open(hf($dir)).unwrap();
            assert_eq!(info.format, loadmaster_model::Format::Safetensors);
            assert!(!info.complete, "header-only fixtures are incomplete");
            insta::assert_snapshot!(summary(&info));
        }
    };
}

hf_test!(hf_llama_3_1_8b_bf16_sharded, "Meta-Llama-3.1-8B-Instruct");
hf_test!(hf_qwen3_8b_awq, "Qwen3-8B-AWQ");
hf_test!(hf_gemma3_12b_multimodal, "gemma-3-12b-it");

#[test]
fn hf_config_json_path_works_too() {
    let a = ModelInfo::open(hf("Qwen3-8B-AWQ")).unwrap();
    let b = ModelInfo::open(hf("Qwen3-8B-AWQ").join("config.json")).unwrap();
    assert_eq!(a.weight_bytes, b.weight_bytes);
}

#[test]
fn hf_missing_shard_is_reported() {
    let dir = tempfile::tempdir().unwrap();
    for f in [
        "config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
    ] {
        std::fs::copy(hf("Qwen3-8B-AWQ").join(f), dir.path().join(f)).unwrap();
    }
    let err = ModelInfo::open(dir.path()).unwrap_err();
    assert!(
        err.to_string().contains("missing; download all shards"),
        "{err}"
    );
}
