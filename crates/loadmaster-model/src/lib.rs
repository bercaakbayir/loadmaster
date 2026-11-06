//! GGUF and safetensors metadata parsers, chat-template extraction, quant tables.
//!
//! Reads headers only; tensor data is never loaded. [`ModelInfo::open`] takes a `.gguf`
//! file or a Hugging Face model directory.

pub mod arch;
pub mod gguf;
mod info;
pub mod safetensors;
pub mod vision;

use std::path::{Path, PathBuf};

pub use arch::{KvLayout, RecurrentPattern, SwaPattern};
pub use info::{
    ChatTemplate, ENCODER_ARCHS, Encoder, Format, Layer, LayerKind, Mmproj, ModelInfo, ModelKind,
    Pooling, Quant, Source, mmproj_candidates, split_part_names,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gguf(#[from] gguf::Error),
    #[error(
        "{path}: missing metadata key `{key}`; the file may come from an old or unsupported converter"
    )]
    MissingKey { path: PathBuf, key: String },
    #[error(
        "{path}: is a projector (mmproj), not a model; pass the model file and loadmaster picks up the mmproj next to it"
    )]
    IsProjector { path: PathBuf },
    #[error("{path}: split GGUF: {reason}")]
    Split { path: PathBuf, reason: String },
    #[error("{path}: not a model: {reason}")]
    NotAModel { path: PathBuf, reason: String },
    #[error("{path}: inconsistent metadata: {reason}")]
    Inconsistent { path: PathBuf, reason: String },
    /// A real model, but not a kind loadmaster plans (image generation, speech-to-text).
    #[error("{path}: {kind} model ({arch}): not supported, see docs/known-issues.md")]
    Unsupported {
        path: PathBuf,
        kind: &'static str,
        arch: String,
    },
}

impl Error {
    /// The file is a model loadmaster does not plan, not a broken one.
    pub fn is_unsupported(&self) -> bool {
        matches!(self, Error::Unsupported { .. })
    }
}

/// A diffusers repo or directory (`model_index.json`, no `config.json`): image or video
/// generation, named after its pipeline class (`Flux2Pipeline` gives `flux2`).
pub fn diffusers_unsupported(path: &Path, model_index: &serde_json::Value) -> Error {
    let class = model_index
        .get("_class_name")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("diffusers");
    let arch = class
        .strip_suffix("Pipeline")
        .unwrap_or(class)
        .to_lowercase();
    let kind = arch::non_llm_kind(&arch).unwrap_or(if arch.contains("video") {
        "video generation"
    } else {
        "image generation"
    });
    Error::Unsupported {
        path: path.to_path_buf(),
        kind,
        arch,
    }
}
