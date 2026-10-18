//! `vllm serve` command line.

use crate::{KvType, Plan};

pub const BIN: &str = "vllm";

pub fn argv(plan: &Plan, util_x100: u64) -> Vec<String> {
    // vLLM resolves a repo id from the HF cache itself; otherwise pass the directory.
    let model = plan
        .model
        .hf_repo
        .clone()
        .unwrap_or_else(|| plan.model.path.clone());
    let mut a: Vec<String> = vec![BIN.into(), "serve".into(), model];
    a.extend(["--max-model-len".into(), plan.ctx.to_string()]);
    a.extend([
        "--gpu-memory-utilization".into(),
        format!("0.{util_x100:02}"),
    ]);
    if plan.kv == KvType::Fp8 {
        a.extend(["--kv-cache-dtype".into(), "fp8".into()]);
    }
    a
}
