//! Splitting a model across several GPUs: llama.cpp `--split-mode layer` and vLLM tensor
//! parallel, on pretend hardware from the catalog.

mod common;

use common::*;
use loadmaster_hw::{HwSnapshot, pretend};
use loadmaster_plan::{PlanRequest, Runtime, plan};

fn split_request(hw: &HwSnapshot, ctx: u32) -> PlanRequest {
    PlanRequest {
        ctx: Some(ctx),
        extra_gpus: hw.gpus[1..].to_vec(),
        ..PlanRequest::default()
    }
}

#[test]
fn layers_split_across_two_gpus() {
    // Qwen3 30B-A3B Q4_K_M (~17 GiB) at 32k does not fit one 12 GB card; two do.
    let hw = pretend("2x rtx 3060").unwrap();
    let m = model(QWEN3_30B_A3B);
    let one = plan(
        &m,
        &hw,
        &hw.gpus[0],
        &PlanRequest {
            ctx: Some(32768),
            ..PlanRequest::default()
        },
    )
    .unwrap();
    assert!(!one.placement.is_full(m.n_layer), "one card has to offload");

    let p = plan(&m, &hw, &hw.gpus[0], &split_request(&hw, 32768)).unwrap();
    assert!(p.placement.is_full(m.n_layer), "two cards hold every layer");
    let d = &p.footprint.per_device;
    assert_eq!(d.len(), 2);
    // Every layer slot is placed once: the layers plus the output head.
    assert_eq!(d.iter().map(|x| x.layers).sum::<u32>(), m.n_layer + 1);
    for (side, gpu) in d.iter().zip(&p.gpus) {
        assert!(
            side.bytes <= gpu.usable_bytes,
            "{} > {}",
            side.bytes,
            gpu.usable_bytes
        );
    }
    assert_eq!(
        d.iter().map(|x| x.bytes).sum::<u64>(),
        p.footprint.gpu_bytes
    );
    let argv = p.argv.join(" ");
    let ts = format!("--tensor-split {},{}", d[0].layers, d[1].layers);
    assert!(
        argv.contains("--split-mode layer") && argv.contains(&ts),
        "{argv}"
    );
    assert!(argv.contains("--device CUDA0,CUDA1"), "{argv}");
}

#[test]
fn one_gpu_of_several_is_pinned() {
    let hw = pretend("2x rtx 3090").unwrap();
    let p = plan(
        &model(QWEN3_8B),
        &hw,
        &hw.gpus[1],
        &PlanRequest {
            ctx: Some(8192),
            ..PlanRequest::default()
        },
    )
    .unwrap();
    assert!(p.footprint.per_device.is_empty());
    assert!(p.argv.join(" ").contains("--device CUDA1"));
}

#[test]
fn different_cards_warn_about_cuda_order() {
    let hw = pretend("rtx 4090 + rtx 3060").unwrap();
    let p = plan(
        &model(QWEN3_30B_A3B),
        &hw,
        &hw.gpus[0],
        &split_request(&hw, 32768),
    )
    .unwrap();
    assert!(
        p.warnings
            .iter()
            .any(|w| w.contains("CUDA_DEVICE_ORDER=PCI_BUS_ID"))
    );
    // The smaller card is not filled past its own budget.
    let d = &p.footprint.per_device;
    assert!(d[1].bytes <= p.gpus[1].usable_bytes);
}

#[test]
fn vllm_uses_tensor_parallel() {
    let m = hf_model(HF_LLAMA_8B);
    let hw = pretend("2x rtx 3060").unwrap();
    let req = PlanRequest {
        runtime: Runtime::Vllm,
        ..split_request(&hw, 32768)
    };
    let p = plan(&m, &hw, &hw.gpus[0], &req).unwrap();
    assert!(p.argv.join(" ").contains("--tensor-parallel-size 2"));
    // A BF16 8B does not fit one 12 GB card at all.
    let one = PlanRequest {
        runtime: Runtime::Vllm,
        ctx: Some(32768),
        ..PlanRequest::default()
    };
    assert!(plan(&m, &hw, &hw.gpus[0], &one).is_err());

    // Three GPUs: 32 heads are not divisible by 3, so two are used.
    let hw = pretend("3x rtx 3060").unwrap();
    let req = PlanRequest {
        runtime: Runtime::Vllm,
        ..split_request(&hw, 32768)
    };
    let p = plan(&m, &hw, &hw.gpus[0], &req).unwrap();
    assert!(p.argv.join(" ").contains("--tensor-parallel-size 2"));
    assert!(p.warnings.iter().any(|w| w.contains("using 2 of 3 GPUs")));
}
