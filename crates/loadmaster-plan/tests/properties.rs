//! Properties of the search over every fixture model.

mod common;

use loadmaster_hw::{FakeHw, GIB, MIB};
use loadmaster_plan::footprint::{self, Config, Headroom};
use loadmaster_plan::{Error, KvType, Placement, PlanRequest, plan};
use proptest::prelude::*;

fn hw(usable_mib: u64, host_gib: u64) -> loadmaster_hw::HwSnapshot {
    let mut snap = FakeHw::rtx_4090().0;
    snap.gpus[0].usable_bytes = usable_mib * MIB;
    snap.gpus[0].total_bytes = snap.gpus[0].total_bytes.max(usable_mib * MIB);
    snap.host.available_bytes = host_gib * GIB;
    snap
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(256))]

    /// A plan always fits both budgets; otherwise the result is CannotFit.
    #[test]
    fn fits_or_cannot_fit(model in 0usize..6, ctx in 256u32..=262_144, usable in 1024u64..=49_152, host in 1u64..=128) {
        let m = &common::all_models()[model];
        let snap = hw(usable, host);
        let req = PlanRequest { ctx: Some(ctx), ..Default::default() };
        match plan(m, &snap, &snap.gpus[0], &req) {
            Ok(p) => {
                prop_assert!(p.footprint.gpu_bytes <= p.budget);
                prop_assert!(p.footprint.host_bytes <= p.host_budget);
                prop_assert_eq!(p.footprint.gpu_bytes, p.footprint.gpu.iter().map(|t| t.bytes).sum::<u64>());
            }
            Err(Error::CannotFit(n)) => prop_assert!(n.short_by > 0),
            Err(e) => prop_assert!(false, "unexpected error {e}"),
        }
    }

    /// More context never needs less memory for the same placement and KV type.
    #[test]
    fn footprint_monotonic_in_ctx(model in 0usize..6, a in 256u32..=200_000, b in 256u32..=200_000, ngl in 0u32..=100, kv in 0usize..3) {
        let m = &common::all_models()[model];
        let (lo, hi) = (a.min(b), a.max(b));
        let kv = KvType::ALL[kv];
        let cfg = |ctx| Config { ctx, n_seq: 1, n_ubatch: 512, kv, flash_attn: true, mmproj_on_gpu: true, headroom: Headroom { bytes: 0, basis: String::new() } };
        let p = Placement { n_gpu_layers: ngl, n_cpu_moe: 0 };
        if let (Some(x), Some(y)) = (footprint::footprint(m, p, &cfg(lo)), footprint::footprint(m, p, &cfg(hi))) {
            prop_assert!(x.gpu_bytes <= y.gpu_bytes);
            prop_assert!(x.host_bytes <= y.host_bytes);
        }
    }

    /// More context never moves more layers onto the GPU.
    #[test]
    fn offload_monotonic_in_ctx(model in 0usize..6, a in 256u32..=200_000, b in 256u32..=200_000, usable in 4096u64..=32_768) {
        let m = &common::all_models()[model];
        let snap = hw(usable, 256);
        let (lo, hi) = (a.min(b), a.max(b));
        let run = |ctx| plan(m, &snap, &snap.gpus[0], &PlanRequest { ctx: Some(ctx), ..Default::default() });
        if let (Ok(x), Ok(y)) = (run(lo), run(hi)) {
            prop_assert!(y.placement.n_gpu_layers <= x.placement.n_gpu_layers);
            prop_assert!(y.placement.n_cpu_moe >= x.placement.n_cpu_moe || y.placement.n_gpu_layers < x.placement.n_gpu_layers);
        }
    }
}
