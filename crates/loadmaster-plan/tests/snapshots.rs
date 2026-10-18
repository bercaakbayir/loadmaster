//! Pinned plans for (model, GPU, ctx) cases. Review every snapshot change by hand.

mod common;

use common::*;
use loadmaster_hw::FakeHw;
use loadmaster_plan::{Error, KvType, PlanRequest, explain, plan};

fn render(model: &str, hw: FakeHw, ctx: Option<u32>, kv: Option<KvType>) -> String {
    let m = common::model(model);
    let gpu = hw.0.best_gpu().unwrap().clone();
    let req = PlanRequest {
        ctx,
        kv,
        ..PlanRequest::default()
    };
    match plan(&m, &hw.0, &gpu, &req) {
        Ok(p) => explain::explain(&p),
        Err(Error::CannotFit(n)) => explain::no_fit(&n, true),
        Err(e) => panic!("{e}"),
    }
}

macro_rules! case {
    ($name:ident, $model:expr, $hw:expr, $ctx:expr) => {
        case!($name, $model, $hw, $ctx, None);
    };
    ($name:ident, $model:expr, $hw:expr, $ctx:expr, $kv:expr) => {
        #[test]
        fn $name() {
            insta::assert_snapshot!(render($model, $hw, $ctx, $kv));
        }
    };
}

case!(
    qwen3_8b_4090_default_ctx,
    QWEN3_8B,
    FakeHw::rtx_4090(),
    None
);
case!(qwen3_8b_4090_32k, QWEN3_8B, FakeHw::rtx_4090(), Some(32768));
case!(
    qwen3_8b_3060_beyond_training_ctx,
    QWEN3_8B,
    FakeHw::rtx_3060_12g(),
    Some(65536)
);
case!(
    llama_8b_3060_default_ctx,
    LLAMA_8B,
    FakeHw::rtx_3060_12g(),
    None
);
case!(
    llama_8b_m2_8g_8k_short_on_free_ram,
    LLAMA_8B,
    FakeHw::m2_8g(),
    Some(8192)
);
case!(
    llama_8b_4090_forced_q4_0,
    LLAMA_8B,
    FakeHw::rtx_4090(),
    Some(131072),
    Some(KvType::Q4_0)
);
case!(
    qwen3_30b_a3b_4090_32k,
    QWEN3_30B_A3B,
    FakeHw::rtx_4090(),
    Some(32768)
);
case!(
    qwen3_30b_a3b_3060_32k_expert_offload,
    QWEN3_30B_A3B,
    FakeHw::rtx_3060_12g(),
    Some(32768)
);
case!(
    qwen3_30b_a3b_m2_8g_cannot_fit,
    QWEN3_30B_A3B,
    FakeHw::m2_8g(),
    Some(8192)
);
case!(
    gemma3_12b_3060_default_ctx_mmproj,
    GEMMA3_12B,
    FakeHw::rtx_3060_12g(),
    None
);
case!(
    deepseek_v2_lite_4090_default_ctx,
    DSV2_LITE,
    FakeHw::rtx_4090(),
    None
);
case!(
    gpt_oss_20b_3060_128k,
    GPT_OSS_20B,
    FakeHw::rtx_3060_12g(),
    Some(131072)
);
case!(
    gpt_oss_20b_m3_max_default_ctx,
    GPT_OSS_20B,
    FakeHw::m3_max_64g(),
    None
);

#[test]
fn json_shape() {
    let m = common::model(QWEN3_8B);
    let hw = FakeHw::rtx_4090();
    let p = plan(
        &m,
        &hw.0,
        &hw.0.gpus[0],
        &PlanRequest {
            ctx: Some(8192),
            ..Default::default()
        },
    )
    .unwrap();
    insta::assert_snapshot!(serde_json::to_string_pretty(&p).unwrap());
}

#[test]
fn summary_output() {
    let m = common::model(GEMMA3_12B);
    let hw = FakeHw::rtx_3060_12g();
    let p = plan(
        &m,
        &hw.0,
        &hw.0.gpus[0],
        &PlanRequest {
            ctx: Some(16384),
            ..Default::default()
        },
    )
    .unwrap();
    insta::assert_snapshot!(explain::summary(&p));
}

#[test]
fn output_is_deterministic() {
    let a = render(QWEN3_30B_A3B, FakeHw::rtx_3060_12g(), Some(32768), None);
    let b = render(QWEN3_30B_A3B, FakeHw::rtx_3060_12g(), Some(32768), None);
    assert_eq!(a, b);
}

#[test]
fn incomplete_download_warns() {
    let mut m = common::model(QWEN3_8B);
    m.complete = false;
    let hw = FakeHw::rtx_4090();
    let p = plan(&m, &hw.0, &hw.0.gpus[0], &PlanRequest::default()).unwrap();
    assert!(p.warnings.iter().any(|w| w.contains("incomplete download")));
}

#[test]
fn cannot_fit_suggests_freeing_memory_held_by_other_processes() {
    let m = common::model(LLAMA_8B);
    let mut hw = FakeHw::rtx_3060_12g();
    // Another process holds 6.5 GiB of VRAM; host RAM is too tight for any offload.
    hw.0.gpus[0].usable_bytes = 5632 << 20;
    hw.0.host.available_bytes = 256 << 20;
    let req = PlanRequest {
        ctx: Some(8192),
        ..Default::default()
    };
    let Err(Error::CannotFit(n)) = plan(&m, &hw.0, &hw.0.gpus[0], &req) else {
        panic!("expected CannotFit");
    };
    assert_eq!(n.limit, "gpu");
    assert_eq!(n.fixes.len(), 2, "{:?}", n.fixes);
    assert!(
        n.fixes[0].summary.contains("held by other processes"),
        "{:?}",
        n.fixes
    );
}
