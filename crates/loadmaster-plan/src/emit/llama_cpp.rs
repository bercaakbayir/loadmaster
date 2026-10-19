//! `llama-server` command line.

use crate::{KvType, Plan};

pub const BIN: &str = "llama-server";

pub fn argv(plan: &Plan) -> Vec<String> {
    let mut a: Vec<String> = vec![BIN.into()];
    // Planned from Hub headers: llama-server fetches the exact file (and the projector).
    match (&plan.model.hf_repo, &plan.model.hf_file) {
        (Some(repo), Some(file)) => {
            a.extend([
                "--hf-repo".into(),
                repo.clone(),
                "--hf-file".into(),
                file.clone(),
            ]);
        }
        _ => a.extend(["-m".into(), plan.model.path.clone()]),
    }
    if let Some(mmproj) = &plan.model.mmproj {
        let flag = if mmproj.starts_with("https://") || mmproj.starts_with("http://") {
            "--mmproj-url"
        } else {
            "--mmproj"
        };
        a.extend([flag.into(), mmproj.clone()]);
    }
    a.extend(["-c".into(), plan.ctx.to_string()]);
    a.extend(["-ngl".into(), plan.placement.n_gpu_layers.to_string()]);
    if plan.placement.n_cpu_moe > 0 {
        a.extend(["--n-cpu-moe".into(), plan.placement.n_cpu_moe.to_string()]);
    }
    a.extend([
        "-fa".into(),
        if plan.flash_attn { "on" } else { "off" }.into(),
    ]);
    if plan.kv != KvType::F16 {
        a.extend(["-ctk".into(), plan.kv.flag().into()]);
        a.extend(["-ctv".into(), plan.kv.flag().into()]);
    }
    if plan.n_ubatch != crate::DEFAULT_UBATCH {
        a.extend(["-ub".into(), plan.n_ubatch.to_string()]);
    }
    // One slot: the KV cache is planned for a single sequence of `ctx` tokens.
    a.extend(["-np".into(), "1".into()]);
    if plan.model.chat_template_blake3.is_some() {
        a.push("--jinja".into());
    }
    a
}

pub fn command(plan: &Plan) -> String {
    super::shell_join(&plan.argv)
}
