//! The unified-memory RAM safety margin: plans that fit but leave under
//! `UNIFIED_RAM_MARGIN_PCT` of available RAM unused are warned about, and the default
//! context backs off to one that leaves the margin.

mod common;

use common::*;
use loadmaster_hw::{FakeHw, HwSnapshot};
use loadmaster_plan::{KvType, PlanRequest, UNIFIED_RAM_MARGIN_PCT, plan};

fn with_available(mut hw: HwSnapshot, bytes: u64) -> HwSnapshot {
    hw.host.available_bytes = bytes;
    hw
}

fn at(ctx: Option<u32>) -> PlanRequest {
    PlanRequest {
        ctx,
        ..PlanRequest::default()
    }
}

fn tight(warnings: &[String]) -> bool {
    warnings.iter().any(|w| w.starts_with("tight on RAM"))
}

/// RAM a Qwen3 8B plan at 8192 needs on a 64 GB Mac.
fn need_8k() -> u64 {
    let hw = FakeHw::m3_max_64g().0;
    plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(Some(8192)))
        .unwrap()
        .host_need
}

#[test]
fn warns_when_the_plan_only_just_fits() {
    let need = need_8k();
    let hw = with_available(FakeHw::m3_max_64g().0, need + need / 20);
    let p = plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(Some(8192))).unwrap();
    let w = p.warnings.iter().find(|w| w.starts_with("tight on RAM"));
    let w = w.expect("a tight-on-RAM warning");
    assert!(w.contains(&format!("{UNIFIED_RAM_MARGIN_PCT}%")), "{w}");
    assert!(
        w.contains("--ctx "),
        "suggests a context that leaves the margin: {w}"
    );
}

#[test]
fn no_warning_with_room_to_spare() {
    let need = need_8k();
    let hw = with_available(FakeHw::m3_max_64g().0, need * 2);
    let p = plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(Some(8192))).unwrap();
    assert!(!tight(&p.warnings), "{:?}", p.warnings);
}

#[test]
fn never_on_a_dedicated_gpu() {
    // Host RAM is not shared with a dedicated GPU, so the margin does not apply.
    let hw = FakeHw::rtx_4090().0;
    for ctx in [None, Some(8192), Some(40960)] {
        if let Ok(p) = plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(ctx)) {
            assert!(!tight(&p.warnings), "{ctx:?}: {:?}", p.warnings);
        }
    }
}

#[test]
fn prefers_a_smaller_kv_cache_that_leaves_the_margin() {
    // f16 KV fits at 40960 but leaves under the margin; q8_0 leaves it.
    let hw = FakeHw::m3_max_64g().0;
    let m = model(QWEN3_8B);
    let f16 = plan(&m, &hw, &hw.gpus[0], &at(Some(40960))).unwrap();
    assert_eq!(f16.kv, KvType::F16);
    let hw = with_available(hw, f16.host_need + f16.host_need / 20);
    let p = plan(&m, &hw, &hw.gpus[0], &at(Some(40960))).unwrap();
    assert_eq!(p.kv, KvType::Q8_0, "{}", p.kv_reason);
    assert!(p.kv_reason.contains("RAM"), "{}", p.kv_reason);
    assert!(!tight(&p.warnings), "{:?}", p.warnings);
}

#[test]
fn default_context_leaves_the_margin() {
    // Enough RAM for the training context with q8_0 KV, but not with the margin on top.
    let hw = FakeHw::m3_max_64g().0;
    let m = model(QWEN3_8B);
    let req = PlanRequest {
        kv: Some(KvType::Q8_0),
        ..at(Some(m.n_ctx_train))
    };
    let full = plan(&m, &hw, &hw.gpus[0], &req).unwrap();
    let hw = with_available(hw, full.host_need + full.host_need / 20);
    let p = plan(&m, &hw, &hw.gpus[0], &at(None)).unwrap();
    assert!(p.ctx < m.n_ctx_train, "picked {}", p.ctx);
    assert!(!tight(&p.warnings), "{:?}", p.warnings);
    assert!(
        p.ctx_reason.contains("fits, without that margin"),
        "{}",
        p.ctx_reason
    );
}
