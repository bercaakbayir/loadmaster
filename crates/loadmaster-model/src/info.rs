//! [`ModelInfo`]: the shared description of a model file, built from its real header.

use std::path::{Path, PathBuf};

use serde::Serialize;

use crate::Error;
use crate::arch::{self, KvLayout, SwaPattern};
use crate::gguf::{GgmlType, Gguf, Value, file_type_name};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Format {
    Gguf,
    /// A Hugging Face model directory: `config.json` plus `*.safetensors`.
    Safetensors,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Quant {
    /// llama.cpp file type, e.g. `Q4_K_M`.
    Gguf(String),
    /// Unquantised safetensors, by dominant dtype: `BF16`, `F16`, `F32`.
    Float(String),
    Awq {
        bits: u32,
    },
    Gptq {
        bits: u32,
    },
    Fp8,
    /// ExLlamaV3, bits per weight x 100.
    Exl3 {
        bpw_x100: u32,
    },
    /// Another `quantization_config.quant_method` (e.g. `compressed-tensors`, `mxfp4`).
    Other(String),
}

impl std::fmt::Display for Quant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Quant::Gguf(name) | Quant::Float(name) | Quant::Other(name) => f.write_str(name),
            Quant::Awq { bits } => write!(f, "AWQ-{bits}bit"),
            Quant::Gptq { bits } => write!(f, "GPTQ-{bits}bit"),
            Quant::Fp8 => f.write_str("FP8"),
            Quant::Exl3 { bpw_x100 } => {
                write!(f, "EXL3-{}.{:02}bpw", bpw_x100 / 100, bpw_x100 % 100)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Source {
    /// A file or directory passed by path, or found in a `--path` directory.
    Loose,
    /// Hugging Face hub cache: `models--org--name/snapshots/<revision>`.
    HfCache { repo: String, revision: String },
    /// Ollama store: a model blob named by its manifest.
    Ollama { name: String, digest: String },
    /// LM Studio models directory: `<publisher>/<repo>/<file>`.
    LmStudio { repo: String },
    /// Headers fetched from the Hugging Face Hub; nothing is on disk. `file` is the GGUF
    /// (first part if split) in the repo, `None` for a safetensors repo.
    HfRemote {
        repo: String,
        revision: String,
        file: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ChatTemplate {
    #[serde(skip)]
    pub source: String,
    /// blake3 of the template text, hex.
    pub blake3: String,
}

impl ChatTemplate {
    pub fn new(source: String) -> Self {
        let blake3 = blake3::hash(source.as_bytes()).to_hex().to_string();
        Self { source, blake3 }
    }
}

/// One repeating block (`blk.N.*`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Layer {
    /// All tensor bytes in this block.
    pub bytes: u64,
    /// Bytes of routed-expert tensors (`*_exps*`), included in `bytes`. These are what
    /// llama.cpp `--n-cpu-moe` moves to the CPU.
    pub expert_bytes: u64,
    /// KV-cache elements per token for K and V in this layer.
    pub k_dim: u32,
    pub v_dim: u32,
    /// Sliding-window attention layer: its cache holds at most the window.
    pub swa: bool,
}

/// A projector loaded alongside the model (`--mmproj`): one file holding a vision encoder,
/// an audio encoder, or both (Qwen2.5-Omni, Gemma 4).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Mmproj {
    pub path: PathBuf,
    pub weight_bytes: u64,
    pub vision: Option<Encoder>,
    pub audio: Option<Encoder>,
}

/// One encoder of a projector file.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Encoder {
    /// The projector type (`gemma3`, `qwen2.5vl_merger`, `ultravox`, ...), or `config.json`
    /// for a safetensors model's built-in vision tower.
    pub projector: String,
    /// False when the projector type is not in loadmaster's table.
    pub known: bool,
    /// Embeddings produced for one image or one 30 s audio clip.
    pub n_tokens: u32,
    /// How `n_tokens` was derived, for `--explain`.
    pub tokens_basis: String,
    /// Upper-bound estimate of the encoder's compute buffer (unvalidated).
    pub compute_bytes: u64,
    pub compute_basis: String,
}

impl Encoder {
    pub(crate) fn new(p: &crate::vision::Projector, unit: &str) -> Self {
        Encoder {
            projector: p.kind.clone(),
            known: p.known,
            n_tokens: p.n_tokens,
            tokens_basis: format!("{}, {} projector", p.tokens_basis, p.kind),
            compute_bytes: p.compute_bytes(),
            compute_basis: format!(
                "estimate, unvalidated: 16 MiB + mask 2 B x {}^2 {unit} + peak of FFN 16 B x {unit} x ffn {} and attention (width {}); flash attention (llama.cpp's default)",
                p.warmup_positions, p.n_ff, p.n_embd
            ),
        }
    }
}

/// What a model is served for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum ModelKind {
    /// Text generation (chat, completion).
    Chat,
    /// Sentence embeddings. `encoder`: a non-causal encoder (BERT and friends), for which
    /// llama.cpp allocates no KV cache and processes each input in one batch.
    Embedding { pooling: Pooling, encoder: bool },
}

/// llama.cpp's `--pooling` (and the GGUF `{arch}.pooling_type` values, llama.h).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Pooling {
    /// Not stated: llama.cpp uses the model's own default.
    Default,
    None,
    Mean,
    Cls,
    Last,
    Rank,
}

impl Pooling {
    /// From `{arch}.pooling_type`: -1 unspecified, 0 none, 1 mean, 2 cls, 3 last, 4 rank.
    pub fn from_llama(v: i64) -> Self {
        match v {
            0 => Pooling::None,
            1 => Pooling::Mean,
            2 => Pooling::Cls,
            3 => Pooling::Last,
            4 => Pooling::Rank,
            _ => Pooling::Default,
        }
    }

    /// The `--pooling` value, when one is stated.
    pub fn flag(self) -> Option<&'static str> {
        Some(match self {
            Pooling::Default => return None,
            Pooling::None => "none",
            Pooling::Mean => "mean",
            Pooling::Cls => "cls",
            Pooling::Last => "last",
            Pooling::Rank => "rank",
        })
    }
}

/// Architectures for which llama.cpp creates no KV cache (`llama_model::create_memory`,
/// src/llama-model.cpp at f174553): non-causal encoders.
pub const ENCODER_ARCHS: &[&str] = &[
    "bert",
    "modern-bert",
    "nomic-bert",
    "nomic-bert-moe",
    "neo-bert",
    "jina-bert-v2",
    "jina-bert-v3",
    "eurobert",
    "gemma-embedding",
    "gemma-embedding2",
];

/// Embedding-only architectures that do keep a KV cache.
const OTHER_EMBEDDING_ARCHS: &[&str] = &["t5encoder", "llama-embed"];

impl ModelKind {
    /// From the architecture and a stated pooling type: encoders and embedding
    /// architectures always embed; any other architecture embeds when the file states a
    /// pooling type (Qwen3-Embedding is `qwen3` with `pooling_type = last`).
    pub fn detect(arch: &str, pooling: Option<Pooling>) -> Self {
        let encoder = ENCODER_ARCHS.contains(&arch);
        if encoder || OTHER_EMBEDDING_ARCHS.contains(&arch) || pooling.is_some() {
            ModelKind::Embedding {
                pooling: pooling.unwrap_or(Pooling::Default),
                encoder,
            }
        } else {
            ModelKind::Chat
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct ModelInfo {
    pub path: PathBuf,
    /// All files of a split GGUF in order; just `path` otherwise.
    pub files: Vec<PathBuf>,
    pub format: Format,
    pub name: Option<String>,
    pub arch: String,
    /// False when `arch` is not in the architecture table and defaults were assumed.
    pub arch_known: bool,
    pub n_params: u64,
    pub n_layer: u32,
    pub n_embd: u32,
    pub n_vocab: u32,
    pub n_head: u32,
    /// Largest KV head count over layers.
    pub n_head_kv: u32,
    pub head_dim: u32,
    pub n_ctx_train: u32,
    /// Dense FFN width (`feed_forward_length`, largest over layers); 0 when not stated.
    pub n_ff: u32,
    pub n_expert: Option<u32>,
    pub n_expert_used: Option<u32>,
    /// Width of one routed expert (`expert_feed_forward_length`).
    pub n_ff_exp: Option<u32>,
    pub n_expert_shared: Option<u32>,
    pub kv_layout: KvLayout,
    pub swa_window: Option<u32>,
    pub quant: Quant,
    /// Exact sum of all tensor sizes in the file(s).
    pub weight_bytes: u64,
    pub layers: Vec<Layer>,
    /// Input embedding tensors (`token_embd*`); llama.cpp keeps these on the CPU.
    pub embd_bytes: u64,
    /// Output head and final norm as loaded. When the head is tied to the embeddings
    /// llama.cpp loads a second copy on the output device, so this includes it.
    pub output_bytes: u64,
    pub output_tied: bool,
    /// Tensors outside layers, embeddings and output (e.g. `rope_freqs`, extra MTP blocks).
    pub other_bytes: u64,
    pub mmproj: Option<Mmproj>,
    pub chat_template: Option<ChatTemplate>,
    pub source: Source,
    /// False when the file is shorter than its tensor table (partial download).
    pub complete: bool,
    pub kind: ModelKind,
}

impl ModelInfo {
    pub fn params_b(&self) -> f64 {
        self.n_params as f64 / 1e9
    }

    /// FFN activation width per token: the dense width, or the routed experts in use plus
    /// shared experts, whichever is larger.
    pub fn ffn_width(&self) -> u32 {
        let moe = match (self.n_ff_exp, self.n_expert_used) {
            (Some(ff), Some(used)) => ff * (used + self.n_expert_shared.unwrap_or(0)),
            _ => 0,
        };
        self.n_ff.max(moe)
    }

    pub fn is_moe(&self) -> bool {
        self.n_expert.is_some_and(|n| n > 1)
    }

    pub fn is_embedding(&self) -> bool {
        matches!(self.kind, ModelKind::Embedding { .. })
    }

    /// `embed`, `vision`, `audio`, `omni` (vision and audio) or `chat`: what the model takes
    /// in, for listings.
    pub fn kind_label(&self) -> &'static str {
        if self.is_embedding() {
            return "embed";
        }
        match self
            .mmproj
            .as_ref()
            .map(|m| (m.vision.is_some(), m.audio.is_some()))
        {
            Some((true, true)) => "omni",
            Some((false, true)) => "audio",
            Some(_) => "vision",
            None => "chat",
        }
    }

    /// False for encoders: llama.cpp allocates no KV cache, and an input must fit one
    /// batch (`n_ubatch >= n_tokens`).
    pub fn has_kv_cache(&self) -> bool {
        !matches!(self.kind, ModelKind::Embedding { encoder: true, .. })
    }

    /// Read a GGUF model, all of its split parts, and a sibling `mmproj*.gguf` if present.
    pub fn from_gguf(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let first = Gguf::open(path)?;
        if first.get_str("general.architecture") == Some("clip") {
            return Err(Error::IsProjector {
                path: path.to_path_buf(),
            });
        }
        let parts = load_split(path, first)?;
        let mut info = build(path, &parts)?;
        info.mmproj = find_mmproj(path, &info.files)?;
        Ok(info)
    }

    /// Build from GGUF headers already in memory, every part of a split model in order (for
    /// headers fetched from the Hub). Each part's `path` and `file_size` must be set by the
    /// caller: `path` is what gets printed, `file_size` decides [`ModelInfo::complete`].
    pub fn from_gguf_parts(path: impl AsRef<Path>, parts: &[Gguf]) -> Result<Self, Error> {
        let path = path.as_ref();
        let first = parts.first().ok_or_else(|| Error::Split {
            path: path.to_path_buf(),
            reason: "no parts".into(),
        })?;
        if first.get_str("general.architecture") == Some("clip") {
            return Err(Error::IsProjector {
                path: path.to_path_buf(),
            });
        }
        let count = first.get_u32("split.count").unwrap_or(1) as usize;
        if count.max(1) != parts.len() {
            return Err(Error::Split {
                path: path.to_path_buf(),
                reason: format!("header says {count} parts, {} given", parts.len()),
            });
        }
        build(path, parts)
    }
}

/// The file names of every part of a split GGUF, given any one of them and its
/// `split.count`; `None` when the name does not follow `<name>-00001-of-00003.gguf`.
pub fn split_part_names(file_name: &str, count: u32) -> Option<Vec<String>> {
    let prefix = split_prefix(file_name, count)?;
    Some(
        (1..=count)
            .map(|no| format!("{prefix}-{no:05}-of-{count:05}.gguf"))
            .collect(),
    )
}

/// Vision projector candidates among `names` (files next to a model), best first: names
/// containing `mmproj` and ending in `.gguf`, F16 ones first, then by name.
pub fn mmproj_candidates<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<&'a str> {
    let mut c: Vec<&str> = names
        .into_iter()
        .filter(|n| {
            let n = n.to_ascii_lowercase();
            n.contains("mmproj") && n.ends_with(".gguf")
        })
        .collect();
    c.sort_by_key(|n| {
        let lower = n.to_ascii_lowercase();
        (!lower.contains("f16"), lower)
    });
    c
}

/// Load every part of a split GGUF (`name-00001-of-00003.gguf`). `first` is the file the
/// user named, which may be any part.
fn load_split(path: &Path, first: Gguf) -> Result<Vec<Gguf>, Error> {
    let count = first.get_u32("split.count").unwrap_or(1);
    if count <= 1 {
        return Ok(vec![first]);
    }
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or_default();
    let Some(prefix) = split_prefix(file_name, count) else {
        return Err(Error::Split {
            path: path.to_path_buf(),
            reason: format!(
                "header says {count} parts but the name does not match `<name>-00001-of-{count:05}.gguf`"
            ),
        });
    };
    let mut parts = Vec::with_capacity(count as usize);
    let mut first = Some(first);
    for no in 1..=count {
        let part = path.with_file_name(format!("{prefix}-{no:05}-of-{count:05}.gguf"));
        let gguf = match first.take_if(|g| g.path == part) {
            Some(g) => g,
            None => Gguf::open(&part).map_err(|e| match e {
                crate::gguf::Error::Io { source, .. }
                    if source.kind() == std::io::ErrorKind::NotFound =>
                {
                    Error::Split {
                        path: part.clone(),
                        reason: format!("part {no} of {count} is missing; download all parts"),
                    }
                }
                e => e.into(),
            })?,
        };
        parts.push(gguf);
    }
    Ok(parts)
}

fn split_prefix(file_name: &str, count: u32) -> Option<&str> {
    let stem = file_name.strip_suffix(".gguf")?;
    let suffix = format!("-of-{count:05}");
    let stem = stem.strip_suffix(&suffix)?;
    let (prefix, no) = stem.rsplit_once('-')?;
    (no.len() == 5 && no.bytes().all(|b| b.is_ascii_digit())).then_some(prefix)
}

fn build(path: &Path, parts: &[Gguf]) -> Result<ModelInfo, Error> {
    let g = &parts[0];
    let missing = |key: String| Error::MissingKey {
        path: path.to_path_buf(),
        key,
    };
    let arch = g
        .get_str("general.architecture")
        .ok_or_else(|| missing("general.architecture".into()))?
        .to_owned();
    let spec = arch::lookup(&arch);
    let key = |k: &str| format!("{arch}.{k}");
    let req = |k: &str| g.get_u32(&key(k)).ok_or_else(|| missing(key(k)));

    let n_layer = req("block_count")?;
    let n_embd = req("embedding_length")?;
    let n_ctx_train = req("context_length")?;
    let n_head_per_layer = per_layer(g, &key("attention.head_count"), n_layer)
        .ok_or_else(|| missing(key("attention.head_count")))?;
    let n_head = n_head_per_layer.iter().copied().max().unwrap_or(0);
    let n_head_kv_per_layer =
        per_layer(g, &key("attention.head_count_kv"), n_layer).unwrap_or(n_head_per_layer);
    let n_head_kv = n_head_kv_per_layer.iter().copied().max().unwrap_or(0);
    let default_head_dim = n_embd.checked_div(n_head).unwrap_or(0);
    let key_len = g
        .get_u32(&key("attention.key_length"))
        .unwrap_or(default_head_dim);
    let value_len = g
        .get_u32(&key("attention.value_length"))
        .unwrap_or(default_head_dim);

    let kv_layout = spec.map_or(KvLayout::Gqa, |s| s.kv);
    let mla = match kv_layout {
        KvLayout::Gqa => None,
        KvLayout::Mla => Some((req("attention.kv_lora_rank")?, req("rope.dimension_count")?)),
    };

    let swa_window = g
        .get_u32(&key("attention.sliding_window"))
        .filter(|&w| w > 0);
    let swa_layers: Vec<bool> = match (swa_window, g.get(&key("attention.sliding_window_pattern")))
    {
        (None, _) => vec![false; n_layer as usize],
        (Some(_), Some(Value::Array(flags))) => {
            let flags: Vec<bool> = flags
                .iter()
                .map(|v| matches!(v, Value::Bool(true)))
                .collect();
            if flags.len() != n_layer as usize {
                return Err(Error::Inconsistent {
                    path: path.to_path_buf(),
                    reason: format!(
                        "{} has {} entries for {n_layer} layers",
                        key("attention.sliding_window_pattern"),
                        flags.len()
                    ),
                });
            }
            flags
        }
        (Some(_), Some(v)) if v.as_u32().is_some() => {
            let p = SwaPattern::Every(v.as_u32().unwrap_or(0));
            (0..n_layer).map(|il| p.is_swa(il)).collect()
        }
        (Some(_), _) => {
            let p = spec.map_or(SwaPattern::None, |s| s.swa);
            (0..n_layer).map(|il| p.is_swa(il)).collect()
        }
    };
    let swa_window = swa_window.filter(|_| swa_layers.contains(&true));

    let mut layers: Vec<Layer> = (0..n_layer as usize)
        .map(|il| {
            let (k_dim, v_dim) = match mla {
                Some((rank, rope)) => (rank + rope, rank),
                None => (
                    n_head_kv_per_layer[il] * key_len,
                    n_head_kv_per_layer[il] * value_len,
                ),
            };
            Layer {
                bytes: 0,
                expert_bytes: 0,
                k_dim,
                v_dim,
                swa: swa_layers[il],
            }
        })
        .collect();

    let mut n_params = 0u64;
    let mut weight_bytes = 0u64;
    let mut embd_bytes = 0u64;
    let mut token_embd_bytes = None;
    let mut output_bytes = 0u64;
    let mut has_output_head = false;
    let mut other_bytes = 0u64;
    let mut n_vocab = None;
    let mut by_type: Vec<(GgmlType, u64)> = Vec::new();

    for t in parts.iter().flat_map(|p| &p.tensors) {
        n_params += t.n_elements();
        weight_bytes += t.n_bytes;
        let layer = t
            .name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(n, _)| n.parse::<usize>().ok())
            .filter(|&il| il < layers.len());
        if let Some(il) = layer {
            layers[il].bytes += t.n_bytes;
            if t.name.contains("_exps") {
                layers[il].expert_bytes += t.n_bytes;
            }
            if t.dims.len() >= 2 {
                match by_type.iter_mut().find(|(ty, _)| *ty == t.ty) {
                    Some((_, b)) => *b += t.n_bytes,
                    None => by_type.push((t.ty, t.n_bytes)),
                }
            }
        } else if t.name.starts_with("token_embd") {
            embd_bytes += t.n_bytes;
            if t.name == "token_embd.weight" {
                token_embd_bytes = Some(t.n_bytes);
                n_vocab = t.dims.get(1).and_then(|&d| u32::try_from(d).ok());
            }
        } else if t.name.starts_with("output") {
            output_bytes += t.n_bytes;
            if t.name == "output.weight" {
                has_output_head = true;
                n_vocab = n_vocab.or_else(|| t.dims.get(1).and_then(|&d| u32::try_from(d).ok()));
            }
        } else {
            other_bytes += t.n_bytes;
        }
    }
    let pooling = g
        .get(&key("pooling_type"))
        .and_then(|v| match *v {
            Value::I32(i) => Some(i64::from(i)),
            ref v => v.as_u64().map(|u| u as i64),
        })
        .map(Pooling::from_llama);
    let kind = ModelKind::detect(&arch, pooling);
    // Embedding models end at pooling: no output head, so no tied copy either.
    let output_tied = !has_output_head && kind == ModelKind::Chat;
    if output_tied {
        output_bytes += token_embd_bytes.unwrap_or(0);
    }
    let n_vocab = n_vocab
        .or_else(|| g.get_u32(&key("vocab_size")))
        .or_else(|| {
            g.get("tokenizer.ggml.tokens")
                .and_then(Value::array_len)
                .and_then(|n| u32::try_from(n).ok())
        })
        .ok_or_else(|| missing("tokenizer.ggml.tokens".into()))?;

    let quant = match g.get_u32("general.file_type").and_then(file_type_name) {
        Some(name) => name.to_owned(),
        None => by_type
            .iter()
            .max_by_key(|(ty, b)| (*b, std::cmp::Reverse(ty.id())))
            .map_or("unknown".into(), |(ty, _)| ty.name().to_uppercase()),
    };

    let n_expert = g.get_u32(&key("expert_count")).filter(|&n| n > 0);
    Ok(ModelInfo {
        path: path.to_path_buf(),
        files: parts.iter().map(|p| p.path.clone()).collect(),
        format: Format::Gguf,
        name: g.get_str("general.name").map(str::to_owned),
        arch: arch.clone(),
        arch_known: spec.is_some(),
        n_params,
        n_layer,
        n_embd,
        n_vocab,
        n_head,
        n_head_kv,
        head_dim: key_len,
        n_ctx_train,
        n_ff: per_layer(g, &key("feed_forward_length"), n_layer)
            .and_then(|v| v.into_iter().max())
            .unwrap_or(0),
        n_expert,
        n_expert_used: n_expert.and(g.get_u32(&key("expert_used_count"))),
        n_ff_exp: n_expert.and(g.get_u32(&key("expert_feed_forward_length"))),
        n_expert_shared: n_expert.and(g.get_u32(&key("expert_shared_count"))),
        kv_layout,
        swa_window,
        quant: Quant::Gguf(quant),
        weight_bytes,
        layers,
        embd_bytes,
        output_bytes,
        output_tied,
        other_bytes,
        mmproj: None,
        chat_template: g
            .get_str("tokenizer.chat_template")
            .map(|s| ChatTemplate::new(s.to_owned())),
        source: Source::Loose,
        complete: parts.iter().all(|p| !p.is_truncated()),
        kind,
    })
}

/// A per-layer hyperparameter stored either as a scalar or as an array of `n_layer`.
fn per_layer(g: &Gguf, key: &str, n_layer: u32) -> Option<Vec<u32>> {
    match g.get(key)? {
        Value::Array(items) if items.len() == n_layer as usize => {
            items.iter().map(Value::as_u32).collect()
        }
        v => v.as_u32().map(|n| vec![n; n_layer as usize]),
    }
}

/// Find a vision projector next to the model: a `*mmproj*.gguf` sibling whose architecture
/// is `clip`. With several, prefer one marked F16, then the first by name.
fn find_mmproj(path: &Path, model_files: &[PathBuf]) -> Result<Option<Mmproj>, Error> {
    let Some(dir) = path.parent() else {
        return Ok(None);
    };
    let dir = if dir.as_os_str().is_empty() {
        Path::new(".")
    } else {
        dir
    };
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(None);
    };
    let names: Vec<String> = entries
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| {
            !model_files
                .iter()
                .any(|m| m.file_name().is_some_and(|f| f == n.as_str()))
        })
        .collect();
    Ok(mmproj_candidates(names.iter().map(String::as_str))
        .into_iter()
        .find_map(|n| Mmproj::open(&path.with_file_name(n)).ok()))
}

impl Mmproj {
    /// Read a projector GGUF (architecture `clip`).
    pub fn open(path: &Path) -> Result<Self, Error> {
        Self::from_gguf(&Gguf::open(path)?, path)
    }

    /// From a projector's GGUF header already in memory; `path` is what gets printed.
    pub fn from_gguf(g: &Gguf, path: &Path) -> Result<Self, Error> {
        if g.get_str("general.architecture") != Some("clip") {
            return Err(Error::NotAModel {
                path: path.to_path_buf(),
                reason: "not a projector (architecture is not `clip`)".into(),
            });
        }
        let (vision, audio) = crate::vision::modalities(g);
        Ok(Mmproj {
            weight_bytes: g.tensor_bytes(),
            path: path.to_path_buf(),
            vision: vision.then(|| Encoder::new(&crate::vision::Projector::vision(g), "patches")),
            audio: audio.then(|| Encoder::new(&crate::vision::Projector::audio(g), "frames")),
        })
    }

    /// Both encoders' compute buffers: llama.cpp reserves one per encoder at warmup.
    pub fn compute_bytes(&self) -> u64 {
        self.encoders().map(|e| e.compute_bytes).sum()
    }

    pub fn encoders(&self) -> impl Iterator<Item = &Encoder> {
        self.vision.iter().chain(self.audio.iter())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_prefix_parses_llama_cpp_names() {
        assert_eq!(
            split_prefix("Qwen3-235B-Q4_K_M-00002-of-00003.gguf", 3),
            Some("Qwen3-235B-Q4_K_M")
        );
        assert_eq!(split_prefix("model.gguf", 3), None);
        assert_eq!(split_prefix("m-00001-of-00004.gguf", 3), None);
    }
}
