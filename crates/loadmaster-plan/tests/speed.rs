//! Speed estimates through the planner: present only with a known bandwidth, ordered bands,
//! slower with offload and with a fuller context, and MoE reads only the experts in use.

mod common;

use common::*;
use loadmaster_hw::{FakeHw, HwSnapshot};
use loadmaster_plan::{PlanRequest, plan};

fn with_bandwidth(mut hw: HwSnapshot, gbs: u64) -> HwSnapshot {
    hw.gpus[0].bandwidth_bytes_s = Some(gbs * 1_000_000_000);
    hw.gpus[0].bandwidth_basis = "test".into();
    hw
}

fn at(ctx: u32) -> PlanRequest {
    PlanRequest {
        ctx: Some(ctx),
        ..PlanRequest::default()
    }
}

#[test]
fn only_with_a_known_bandwidth() {
    let hw = FakeHw::rtx_4090().0;
    let p = plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(8192)).unwrap();
    assert_eq!(p.speed, None);
    let hw = with_bandwidth(hw, 1008);
    let p = plan(&model(QWEN3_8B), &hw, &hw.gpus[0], &at(8192)).unwrap();
    let s = p.speed.unwrap();
    assert_eq!(s.at.len(), 2);
    let (a, b) = (&s.at[0], &s.at[1]);
    assert_eq!((a.fill, b.fill), (4096, 8192));
    assert!(a.tok_s_x10.0 < a.tok_s_x10.1);
    // A fuller context reads more KV per token.
    assert!(b.gpu_bytes > a.gpu_bytes && b.tok_s_x10.1 < a.tok_s_x10.1);
    // Dense 8B Q4_K_M on a 4090: within the range llama.cpp users report (~100-150).
    assert!(
        a.tok_s_x10.0 >= 800 && a.tok_s_x10.1 <= 1800,
        "{:?}",
        a.tok_s_x10
    );
}

#[test]
fn moe_reads_only_active_experts_and_offload_is_slower() {
    let hw = with_bandwidth(FakeHw::rtx_4090().0, 1008);
    let m = model(QWEN3_30B_A3B);
    let full = plan(&m, &hw, &hw.gpus[0], &at(4096)).unwrap();
    let s = full.speed.unwrap();
    // 8 of 128 experts per token: far less than the whole file is read.
    assert!(
        s.at[0].gpu_bytes < m.weight_bytes / 4,
        "{}",
        s.at[0].gpu_bytes
    );
    assert_eq!(s.at[0].cpu_bytes, 0);

    let small = with_bandwidth(FakeHw::rtx_3060_12g().0, 360);
    let off = plan(&m, &small, &small.gpus[0], &at(4096)).unwrap();
    assert!(!off.placement.is_full(m.n_layer));
    let o = off.speed.unwrap();
    assert!(o.at[0].cpu_bytes > 0 && o.host_basis.is_some());
    assert!(o.at[0].tok_s_x10.1 < s.at[0].tok_s_x10.0);
}
