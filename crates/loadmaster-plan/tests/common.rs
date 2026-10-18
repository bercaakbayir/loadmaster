#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use loadmaster_model::ModelInfo;

pub const QWEN3_8B: &str = "qwen3-8b/Qwen3-8B-Q4_K_M.gguf";
pub const LLAMA_8B: &str = "llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf";
pub const QWEN3_30B_A3B: &str = "qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf";
pub const GEMMA3_12B: &str = "gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf";
pub const DSV2_LITE: &str = "deepseek-v2-lite/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf";
pub const GPT_OSS_20B: &str = "gpt-oss-20b/gpt-oss-20b-mxfp4.gguf";
pub const ALL: [&str; 6] = [
    QWEN3_8B,
    LLAMA_8B,
    QWEN3_30B_A3B,
    GEMMA3_12B,
    DSV2_LITE,
    GPT_OSS_20B,
];

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gguf")
}

/// Load a fixture and rewrite its paths to be relative, so snapshots do not depend on
/// where the repository is checked out.
pub fn model(rel: &str) -> ModelInfo {
    let mut info = ModelInfo::from_gguf(root().join(rel)).unwrap();
    // Fixtures are header-only; treat them as complete downloads.
    assert!(!info.complete);
    info.complete = true;
    let rel_path = |p: &Path| PathBuf::from("fixtures/gguf").join(p.strip_prefix(root()).unwrap());
    info.path = rel_path(&info.path);
    info.files = info.files.iter().map(|p| rel_path(p)).collect();
    if let Some(m) = &mut info.mmproj {
        m.path = rel_path(&m.path);
    }
    info
}

pub fn all_models() -> &'static [ModelInfo] {
    static MODELS: OnceLock<Vec<ModelInfo>> = OnceLock::new();
    MODELS.get_or_init(|| ALL.iter().map(|m| model(m)).collect())
}

pub const HF_LLAMA_8B: &str = "Meta-Llama-3.1-8B-Instruct";
pub const HF_QWEN3_AWQ: &str = "Qwen3-8B-AWQ";
pub const HF_GEMMA3_12B: &str = "gemma-3-12b-it";
pub const HF_QWEN3_EXL3: &str = "Qwen3-8B-exl3-4.0bpw";

/// Load an HF fixture directory, with paths made relative and treated as complete.
pub fn hf_model(name: &str) -> ModelInfo {
    let hf_root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/hf");
    let mut info = ModelInfo::open(hf_root.join(name)).unwrap();
    let rel = |p: &Path| PathBuf::from("fixtures/hf").join(p.strip_prefix(&hf_root).unwrap());
    info.path = rel(&info.path);
    info.files = info.files.iter().map(|p| rel(p)).collect();
    if let Some(m) = &mut info.mmproj {
        m.path = rel(&m.path);
    }
    info.complete = true;
    info
}
