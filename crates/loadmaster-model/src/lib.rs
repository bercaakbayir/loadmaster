//! GGUF and safetensors metadata parsers, chat-template extraction, quant tables.
//!
//! Reads headers only; tensor data is never loaded. [`ModelInfo::open`] takes a `.gguf`
//! file or a Hugging Face model directory.

pub mod arch;
pub mod gguf;
mod info;
pub mod safetensors;
pub mod vision;

use std::path::PathBuf;

pub use arch::{KvLayout, SwaPattern};
pub use info::{
    ChatTemplate, Format, Layer, Mmproj, ModelInfo, Quant, Source, mmproj_candidates,
    split_part_names,
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
        "{path}: is a vision projector (mmproj), not a model; pass the model file and loadmaster picks up the mmproj next to it"
    )]
    IsProjector { path: PathBuf },
    #[error("{path}: split GGUF: {reason}")]
    Split { path: PathBuf, reason: String },
    #[error("{path}: not a model: {reason}")]
    NotAModel { path: PathBuf, reason: String },
    #[error("{path}: inconsistent metadata: {reason}")]
    Inconsistent { path: PathBuf, reason: String },
}
