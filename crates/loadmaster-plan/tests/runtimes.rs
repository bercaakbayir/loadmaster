//! vLLM and ExLlamaV3 plans, and runtime/format mismatches.

mod common;

use common::*;
use loadmaster_hw::FakeHw;
use loadmaster_plan::{Error, PlanRequest, Runtime, explain, plan};

fn render(
    model: loadmaster_model::ModelInfo,
    hw: FakeHw,
    runtime: Runtime,
    ctx: Option<u32>,
) -> String {
    let gpu = hw.0.best_gpu().unwrap().clone();
    let req = PlanRequest {
        ctx,
        runtime,
        ..Default::default()
    };
    match plan(&model, &hw.0, &gpu, &req) {
        Ok(p) => explain::explain(&p),
        Err(Error::CannotFit(n)) => explain::no_fit(&n, true),
        Err(e) => format!("error: {e}"),
    }
}

#[test]
fn vllm_llama_bf16_4090_default_ctx() {
    insta::assert_snapshot!(render(
        hf_model(HF_LLAMA_8B),
        FakeHw::rtx_4090(),
        Runtime::Vllm,
        None
    ));
}

#[test]
fn vllm_qwen3_awq_3060_32k() {
    insta::assert_snapshot!(render(
        hf_model(HF_QWEN3_AWQ),
        FakeHw::rtx_3060_12g(),
        Runtime::Vllm,
        Some(32768)
    ));
}

#[test]
fn vllm_gemma3_bf16_4090_cannot_fit() {
    insta::assert_snapshot!(render(
        hf_model(HF_GEMMA3_12B),
        FakeHw::rtx_4090(),
        Runtime::Vllm,
        None
    ));
}

#[test]
fn vllm_llama_bf16_3060_cannot_fit() {
    let out = render(
        hf_model(HF_LLAMA_8B),
        FakeHw::rtx_3060_12g(),
        Runtime::Vllm,
        None,
    );
    assert!(out.contains("does not fit"), "{out}");
    assert!(out.contains("smaller quant"), "{out}");
}

#[test]
fn exl3_qwen3_3060_default_ctx() {
    insta::assert_snapshot!(render(
        hf_model(HF_QWEN3_EXL3),
        FakeHw::rtx_3060_12g(),
        Runtime::Exl3,
        None
    ));
}

#[test]
fn runtime_mismatches_are_explained() {
    let cases = [
        (
            hf_model(HF_LLAMA_8B),
            FakeHw::m3_max_64g(),
            Runtime::Vllm,
            "NVIDIA GPU",
        ),
        (
            common::model(QWEN3_8B),
            FakeHw::rtx_4090(),
            Runtime::Vllm,
            "GGUF file",
        ),
        (
            hf_model(HF_LLAMA_8B),
            FakeHw::rtx_4090(),
            Runtime::LlamaCpp,
            "llama.cpp loads GGUF",
        ),
        (
            hf_model(HF_QWEN3_AWQ),
            FakeHw::rtx_4090(),
            Runtime::Exl3,
            "not an EXL3 quant",
        ),
        (
            hf_model(HF_QWEN3_EXL3),
            FakeHw::rtx_4090(),
            Runtime::Vllm,
            "--runtime exl3",
        ),
    ];
    for (model, hw, runtime, want) in cases {
        let out = render(model, hw, runtime, None);
        assert!(
            out.starts_with("error:") && out.contains(want),
            "{runtime:?}: {out}"
        );
    }
}

#[test]
fn vllm_plan_fits_its_budget() {
    let hw = FakeHw::rtx_4090();
    for ctx in [4096, 16384, 65536, 131072] {
        let req = PlanRequest {
            ctx: Some(ctx),
            runtime: Runtime::Vllm,
            ..Default::default()
        };
        if let Ok(p) = plan(&hf_model(HF_LLAMA_8B), &hw.0, &hw.0.gpus[0], &req) {
            assert!(p.footprint.gpu_bytes <= p.budget, "ctx {ctx}");
            assert_eq!(p.argv[0], "vllm");
        }
    }
}
