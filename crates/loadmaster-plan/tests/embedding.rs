//! Embedding models: encoders get no KV cache and a batch as long as the context, causal
//! embedding models keep their KV cache, and the commands ask for embeddings.

use loadmaster_hw::FakeHw;
use loadmaster_model::gguf::writer::GgufWriter;
use loadmaster_model::gguf::{GgmlType, Gguf, Value};
use loadmaster_model::{ModelInfo, ModelKind, Pooling};
use loadmaster_plan::{PlanRequest, plan};

/// A header-only GGUF of a small transformer, `arch` with `pooling_type` when given.
fn model(arch: &str, pooling: Option<i32>, ctx: u32) -> ModelInfo {
    let (n_layer, e, ff, vocab) = (12u32, 384u64, 1536u64, 30522u64);
    let mut w = GgufWriter::new();
    w.kv("general.architecture", Value::String(arch.into()));
    let k = |s: &str| format!("{arch}.{s}");
    w.kv(&k("block_count"), Value::U32(n_layer));
    w.kv(&k("embedding_length"), Value::U32(e as u32));
    w.kv(&k("feed_forward_length"), Value::U32(ff as u32));
    w.kv(&k("context_length"), Value::U32(ctx));
    w.kv(&k("attention.head_count"), Value::U32(12));
    if let Some(p) = pooling {
        w.kv(&k("pooling_type"), Value::I32(p));
    }
    w.tensor("token_embd.weight", &[e, vocab], GgmlType::F16);
    for il in 0..n_layer {
        for t in ["attn_q", "attn_k", "attn_v", "attn_output"] {
            w.tensor(&format!("blk.{il}.{t}.weight"), &[e, e], GgmlType::F16);
        }
        w.tensor(&format!("blk.{il}.ffn_up.weight"), &[e, ff], GgmlType::F16);
        w.tensor(
            &format!("blk.{il}.ffn_down.weight"),
            &[ff, e],
            GgmlType::F16,
        );
    }
    let bytes = w.finish_header_only();
    let mut g = Gguf::parse_bytes(&bytes).unwrap();
    let end = g
        .tensors
        .iter()
        .map(|t| t.offset + t.n_bytes)
        .max()
        .unwrap();
    g.file_size = g.data_offset + end;
    g.path = format!("{arch}.gguf").into();
    ModelInfo::from_gguf_parts(g.path.clone(), &[g]).unwrap()
}

fn hw() -> loadmaster_hw::HwSnapshot {
    let mut hw = FakeHw::rtx_4090().0;
    hw.gpus[0].bandwidth_bytes_s = Some(1_008_000_000_000);
    hw
}

#[test]
fn an_encoder_has_no_kv_cache_and_one_batch() {
    let m = model("bert", Some(2), 512);
    assert_eq!(
        m.kind,
        ModelKind::Embedding {
            pooling: Pooling::Cls,
            encoder: true
        }
    );
    assert!(!m.output_tied && m.output_bytes == 0, "no output head");
    let hw = hw();
    let p = plan(&m, &hw, &hw.gpus[0], &PlanRequest::default()).unwrap();
    assert_eq!(p.ctx, 512, "the training context");
    assert_eq!(p.footprint.term("kv").unwrap().bytes, 0);
    assert_eq!(p.n_ubatch, 512);
    let argv = p.argv.join(" ");
    for flag in ["--embeddings", "--pooling cls", "-ub 512", "-b 512"] {
        assert!(argv.contains(flag), "{argv}");
    }
    assert!(!argv.contains("--jinja"));
    assert_eq!(p.speed, None, "no generation, no tok/s");
    let summary = loadmaster_plan::explain::summary(&p);
    assert!(summary.contains("all 12 layers on GPU, no KV"), "{summary}");
    assert!(
        p.warnings
            .iter()
            .all(|w| !w.contains("not in loadmaster's table"))
    );
}

#[test]
fn a_causal_embedding_model_keeps_its_kv_cache() {
    // Qwen3-Embedding: arch qwen3 with a pooling type.
    let m = model("qwen3", Some(3), 32768);
    assert_eq!(
        m.kind,
        ModelKind::Embedding {
            pooling: Pooling::Last,
            encoder: false
        }
    );
    let hw = hw();
    let req = PlanRequest {
        ctx: Some(8192),
        ..PlanRequest::default()
    };
    let p = plan(&m, &hw, &hw.gpus[0], &req).unwrap();
    assert!(p.footprint.term("kv").unwrap().bytes > 0);
    assert_eq!(p.n_ubatch, loadmaster_plan::DEFAULT_UBATCH);
    let argv = p.argv.join(" ");
    assert!(
        argv.contains("--embeddings --pooling last") && !argv.contains(" -b "),
        "{argv}"
    );
}

#[test]
fn chat_models_are_unchanged() {
    let m = model("qwen3", None, 32768);
    assert_eq!(m.kind, ModelKind::Chat);
    let hw = hw();
    let p = plan(&m, &hw, &hw.gpus[0], &PlanRequest::default()).unwrap();
    assert!(!p.argv.contains(&"--embeddings".to_owned()));
    assert!(p.speed.is_some());
}
