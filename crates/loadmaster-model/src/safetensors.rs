//! Hugging Face model directories: `config.json` for hyperparameters, safetensors headers
//! for exact tensor sizes, `tokenizer_config.json` for the chat template.
//!
//! A safetensors file starts with a little-endian u64 header length and a JSON header whose
//! `data_offsets` give every tensor's exact byte range; only that header is read.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::arch::{self, KvLayout, SwaPattern};
use crate::{ChatTemplate, Error, Format, Layer, Mmproj, ModelInfo, Quant, Source};

const MAX_HEADER: u64 = 256 << 20;

/// What [`ModelInfo::from_hf_parts`] builds a model from.
pub struct HfParts {
    /// Shown as the model's path (a directory, or an `hf://` spec).
    pub path: PathBuf,
    pub name: Option<String>,
    pub config: Value,
    /// Labels errors about `config`.
    pub config_path: PathBuf,
    /// One per shard, in order.
    pub headers: Vec<StHeader>,
    pub chat_template: Option<ChatTemplate>,
}

/// One tensor from a safetensors header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StTensor {
    pub name: String,
    pub dtype: String,
    pub n_elements: u64,
    pub n_bytes: u64,
}

/// Parsed header of one `.safetensors` file.
#[derive(Debug, Clone)]
pub struct StHeader {
    pub path: PathBuf,
    pub tensors: Vec<StTensor>,
    /// False when the file is shorter than its header says.
    pub complete: bool,
}

impl StHeader {
    pub fn open(path: &Path) -> Result<Self, Error> {
        let io = |source| crate::gguf::Error::Io {
            path: path.to_path_buf(),
            source,
        };
        let mut f = File::open(path).map_err(io)?;
        let file_size = f.metadata().map_err(io)?.len();
        let mut len = [0u8; 8];
        f.read_exact(&mut len).map_err(io)?;
        let n = u64::from_le_bytes(len);
        if n > MAX_HEADER || n + 8 > file_size {
            return Err(bad_length(path, n, file_size));
        }
        let mut buf = len.to_vec();
        buf.resize(8 + n as usize, 0);
        f.read_exact(&mut buf[8..]).map_err(io)?;
        Self::from_bytes(path, &buf, file_size)
    }

    /// Parse a header already in memory: the 8-byte length and the JSON after it. `path`
    /// labels errors; `file_size` is the whole file's size, for [`StHeader::complete`].
    pub fn from_bytes(path: &Path, bytes: &[u8], file_size: u64) -> Result<Self, Error> {
        let n = bytes
            .get(..8)
            .map_or(0, |b| u64::from_le_bytes(b.try_into().expect("8 bytes")));
        let Some(buf) = bytes
            .get(8..8 + n as usize)
            .filter(|_| n > 0 && n <= MAX_HEADER)
        else {
            return Err(bad_length(path, n, file_size));
        };
        let header: serde_json::Map<String, Value> =
            serde_json::from_slice(buf).map_err(|e| Error::Inconsistent {
                path: path.to_path_buf(),
                reason: format!("safetensors header is not JSON: {e}"),
            })?;
        let mut tensors = Vec::with_capacity(header.len());
        let mut end = 0u64;
        for (name, info) in &header {
            if name == "__metadata__" {
                continue;
            }
            let bad = || Error::Inconsistent {
                path: path.to_path_buf(),
                reason: format!("tensor `{name}` has a malformed header entry"),
            };
            let offsets = info["data_offsets"].as_array().ok_or_else(bad)?;
            let (a, b) = match offsets.as_slice() {
                [a, b] => (a.as_u64().ok_or_else(bad)?, b.as_u64().ok_or_else(bad)?),
                _ => return Err(bad()),
            };
            if b < a {
                return Err(bad());
            }
            let n_elements = info["shape"]
                .as_array()
                .ok_or_else(bad)?
                .iter()
                .map(|d| d.as_u64().ok_or_else(bad))
                .product::<Result<u64, _>>()?;
            end = end.max(b);
            tensors.push(StTensor {
                name: name.clone(),
                dtype: info["dtype"].as_str().unwrap_or("?").to_owned(),
                n_elements,
                n_bytes: b - a,
            });
        }
        tensors.sort_by(|x, y| x.name.cmp(&y.name));
        Ok(StHeader {
            path: path.to_path_buf(),
            tensors,
            complete: file_size >= 8 + n + end,
        })
    }
}

fn bad_length(path: &Path, n: u64, file_size: u64) -> Error {
    Error::Inconsistent {
        path: path.to_path_buf(),
        reason: format!("safetensors header length {n} is invalid for a {file_size}-byte file"),
    }
}

/// The shard file names `model.safetensors.index.json` lists, sorted and deduplicated.
pub fn index_shards(index: &Value) -> Vec<String> {
    let mut files: Vec<String> = index["weight_map"]
        .as_object()
        .map(|m| {
            m.values()
                .filter_map(|f| f.as_str().map(str::to_owned))
                .collect()
        })
        .unwrap_or_default();
    files.sort();
    files.dedup();
    files
}

/// The safetensors files of a model directory: from `model.safetensors.index.json` when
/// present, else every `*.safetensors` file, sorted by name.
fn shard_files(dir: &Path) -> Result<Vec<PathBuf>, Error> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let files: Vec<PathBuf> = index_shards(&read_json(&index)?)
            .into_iter()
            .map(|f| dir.join(f))
            .collect();
        if let Some(missing) = files.iter().find(|f| !f.exists()) {
            return Err(Error::Split {
                path: missing.clone(),
                reason: "listed in model.safetensors.index.json but missing; download all shards"
                    .into(),
            });
        }
        return Ok(files);
    }
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|source| crate::gguf::Error::Io {
            path: dir.to_path_buf(),
            source,
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
        .collect();
    files.sort();
    if files.is_empty() {
        return Err(Error::NotAModel {
            path: dir.to_path_buf(),
            reason: "no .safetensors files".into(),
        });
    }
    Ok(files)
}

fn read_json(path: &Path) -> Result<Value, Error> {
    let text = std::fs::read_to_string(path).map_err(|source| crate::gguf::Error::Io {
        path: path.to_path_buf(),
        source,
    })?;
    serde_json::from_str(&text).map_err(|e| Error::Inconsistent {
        path: path.to_path_buf(),
        reason: format!("not valid JSON: {e}"),
    })
}

/// HF `model_type` to the GGUF architecture name the arch table uses.
fn gguf_arch(model_type: &str) -> &str {
    match model_type {
        "mistral" | "llama" => "llama",
        "qwen3_moe" => "qwen3moe",
        "qwen2_moe" => "qwen2moe",
        "gemma3" | "gemma3_text" => "gemma3",
        "gpt_oss" => "gpt-oss",
        "deepseek_v2" | "deepseek_v3" | "kimi_k2" => "deepseek2",
        "glm4_moe" => "glm4moe",
        other => other,
    }
}

/// Look up a key in `text_config` first (multimodal configs nest the LM there), then at the
/// top level.
struct Cfg<'a> {
    top: &'a Value,
    text: &'a Value,
}

impl Cfg<'_> {
    fn get(&self, k: &str) -> Option<&Value> {
        self.text
            .get(k)
            .filter(|v| !v.is_null())
            .or_else(|| self.top.get(k).filter(|v| !v.is_null()))
    }

    fn u32(&self, k: &str) -> Option<u32> {
        self.get(k)?.as_u64().and_then(|v| u32::try_from(v).ok())
    }

    fn first_u32(&self, keys: &[&str]) -> Option<u32> {
        keys.iter().find_map(|k| self.u32(k))
    }
}

/// Which part of the model a tensor belongs to.
enum Part {
    Layer(usize),
    Embedding,
    Output,
    Vision,
    Other,
}

fn classify(name: &str) -> Part {
    let vision = [
        "vision",
        "visual",
        "multi_modal_projector",
        "mm_projector",
        "image_newline",
    ];
    if vision.iter().any(|v| name.contains(v)) {
        return Part::Vision;
    }
    let mut parts = name.split('.');
    while let Some(p) = parts.next() {
        if p == "layers" || p == "h" || p == "blocks" {
            if let Some(Ok(il)) = parts.next().map(str::parse::<usize>) {
                return Part::Layer(il);
            }
        }
    }
    if name.contains("embed_tokens") || name.contains("wte.") {
        Part::Embedding
    } else if name.starts_with("lm_head")
        || name.ends_with("model.norm.weight")
        || name.contains(".norm.") && !name.contains("layers")
    {
        Part::Output
    } else {
        Part::Other
    }
}

fn quant(cfg: &Cfg<'_>, dominant_dtype: &str) -> Quant {
    let Some(q) = cfg.top.get("quantization_config").filter(|q| q.is_object()) else {
        return Quant::Float(dominant_dtype.to_owned());
    };
    let method = q["quant_method"].as_str().unwrap_or("unknown");
    let bits = q["bits"]
        .as_u64()
        .and_then(|b| u32::try_from(b).ok())
        .unwrap_or(4);
    match method {
        "awq" => Quant::Awq { bits },
        "gptq" => Quant::Gptq { bits },
        "fp8" => Quant::Fp8,
        "exl3" => Quant::Exl3 {
            bpw_x100: (q["bits"].as_f64().unwrap_or(4.0) * 100.0).round() as u32,
        },
        other => Quant::Other(other.to_owned()),
    }
}

fn chat_template(dir: &Path) -> Option<ChatTemplate> {
    chat_template_from(
        std::fs::read_to_string(dir.join("chat_template.jinja")).ok(),
        read_json(&dir.join("tokenizer_config.json")).ok().as_ref(),
    )
}

/// The chat template from `chat_template.jinja` if there is one, else from
/// `tokenizer_config.json` (a string, or a list of named templates).
pub fn chat_template_from(
    jinja: Option<String>,
    tokenizer_config: Option<&Value>,
) -> Option<ChatTemplate> {
    if let Some(text) = jinja {
        return Some(ChatTemplate::new(text));
    }
    match &tokenizer_config?["chat_template"] {
        Value::String(s) => Some(ChatTemplate::new(s.clone())),
        Value::Array(named) => named
            .iter()
            .find(|t| t["name"] == "default")
            .or_else(|| named.first())
            .and_then(|t| t["template"].as_str())
            .map(|s| ChatTemplate::new(s.to_owned())),
        _ => None,
    }
}

impl ModelInfo {
    /// Read a Hugging Face model directory (or its `config.json`).
    pub fn from_hf_dir(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        let dir = if path.is_dir() {
            path
        } else {
            path.parent().unwrap_or(Path::new("."))
        };
        let config_path = dir.join("config.json");
        if !config_path.exists() {
            return Err(Error::NotAModel {
                path: dir.to_path_buf(),
                reason: "no config.json; pass a Hugging Face model directory or a .gguf file"
                    .into(),
            });
        }
        let top = read_json(&config_path)?;
        let files = shard_files(dir)?;
        let headers: Vec<StHeader> = files
            .iter()
            .map(|f| StHeader::open(f))
            .collect::<Result<_, _>>()?;
        let name = dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .filter(|n| !n.chars().all(|c| c.is_ascii_hexdigit()));
        Self::from_hf_parts(HfParts {
            path: dir.to_path_buf(),
            name,
            config: top,
            config_path,
            headers,
            chat_template: chat_template(dir),
        })
    }

    /// Build from a model's `config.json` and safetensors headers already in memory (for
    /// headers fetched from the Hub). Each header's `path` becomes an entry of `files`.
    pub fn from_hf_parts(parts: HfParts) -> Result<Self, Error> {
        let HfParts {
            path: dir,
            name,
            config: top,
            config_path,
            headers,
            chat_template,
        } = parts;
        let files: Vec<PathBuf> = headers.iter().map(|h| h.path.clone()).collect();
        let text = top.get("text_config").unwrap_or(&top);
        let cfg = Cfg { top: &top, text };
        let missing = |key: &str| Error::MissingKey {
            path: config_path.clone(),
            key: key.to_owned(),
        };

        let model_type = cfg
            .get("model_type")
            .and_then(Value::as_str)
            .ok_or_else(|| missing("model_type"))?;
        let arch = gguf_arch(model_type).to_owned();
        let spec = arch::lookup(&arch);
        let n_layer = cfg
            .u32("num_hidden_layers")
            .ok_or_else(|| missing("num_hidden_layers"))?;
        let n_embd = cfg
            .u32("hidden_size")
            .ok_or_else(|| missing("hidden_size"))?;
        let n_head = cfg
            .u32("num_attention_heads")
            .ok_or_else(|| missing("num_attention_heads"))?;
        let n_head_kv = cfg.u32("num_key_value_heads").unwrap_or(n_head);
        let head_dim = cfg
            .u32("head_dim")
            .unwrap_or_else(|| n_embd.checked_div(n_head).unwrap_or(0));
        let n_ctx_train = cfg
            .u32("max_position_embeddings")
            .ok_or_else(|| missing("max_position_embeddings"))?;
        let n_vocab = cfg.u32("vocab_size").ok_or_else(|| missing("vocab_size"))?;
        let n_expert = cfg
            .first_u32(&["num_experts", "num_local_experts", "n_routed_experts"])
            .filter(|&n| n > 1);
        let n_ff = cfg.u32("intermediate_size").unwrap_or(0);

        let kv_layout = spec.map_or(KvLayout::Gqa, |s| s.kv);
        let mla = match kv_layout {
            KvLayout::Mla => Some((
                cfg.u32("kv_lora_rank")
                    .ok_or_else(|| missing("kv_lora_rank"))?,
                cfg.u32("qk_rope_head_dim")
                    .ok_or_else(|| missing("qk_rope_head_dim"))?,
            )),
            KvLayout::Gqa => None,
        };

        let use_swa = cfg.get("use_sliding_window").and_then(Value::as_bool) != Some(false);
        let window = cfg.u32("sliding_window").filter(|&w| w > 0 && use_swa);
        let swa_layers: Vec<bool> = match (window, cfg.get("layer_types").and_then(Value::as_array))
        {
            (None, _) => vec![false; n_layer as usize],
            (Some(_), Some(types)) if types.len() == n_layer as usize => types
                .iter()
                .map(|t| t.as_str() == Some("sliding_attention"))
                .collect(),
            (Some(_), _) => {
                let p = match cfg.u32("sliding_window_pattern") {
                    Some(n) => SwaPattern::Every(n),
                    None => spec.map_or(SwaPattern::None, |s| s.swa),
                };
                (0..n_layer).map(|il| p.is_swa(il)).collect()
            }
        };
        let swa_window = window.filter(|_| swa_layers.contains(&true));

        let mut layers: Vec<Layer> = (0..n_layer as usize)
            .map(|il| {
                let (k_dim, v_dim) = match mla {
                    Some((rank, rope)) => (rank + rope, rank),
                    None => (n_head_kv * head_dim, n_head_kv * head_dim),
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
        let (mut n_params, mut weight_bytes, mut embd_bytes, mut output_bytes) = (0, 0, 0, 0);
        let (mut vision_bytes, mut other_bytes) = (0u64, 0u64);
        let mut has_lm_head = false;
        let mut by_dtype: Vec<(String, u64)> = Vec::new();
        let quant = quant(&cfg, "?");
        // AWQ/GPTQ pack 32/bits weights into each int32 of `qweight`; zeros, scales and
        // group indices are metadata, not parameters.
        let packed = match quant {
            Quant::Awq { bits } | Quant::Gptq { bits } if bits > 0 => Some(32 / bits),
            _ => None,
        };
        for t in headers.iter().flat_map(|h| &h.tensors) {
            n_params += match packed {
                Some(per) if t.name.ends_with(".qweight") => t.n_elements * u64::from(per),
                Some(_)
                    if [".qzeros", ".scales", ".g_idx"]
                        .iter()
                        .any(|x| t.name.ends_with(x)) =>
                {
                    0
                }
                _ => t.n_elements,
            };
            weight_bytes += t.n_bytes;
            match classify(&t.name) {
                Part::Layer(il) if il < layers.len() => {
                    layers[il].bytes += t.n_bytes;
                    if t.name.contains(".experts.") {
                        layers[il].expert_bytes += t.n_bytes;
                    }
                    match by_dtype.iter_mut().find(|(d, _)| *d == t.dtype) {
                        Some((_, b)) => *b += t.n_bytes,
                        None => by_dtype.push((t.dtype.clone(), t.n_bytes)),
                    }
                }
                Part::Embedding => embd_bytes += t.n_bytes,
                Part::Output => {
                    has_lm_head |= t.name.starts_with("lm_head");
                    output_bytes += t.n_bytes;
                }
                Part::Vision => vision_bytes += t.n_bytes,
                Part::Layer(_) | Part::Other => other_bytes += t.n_bytes,
            }
        }
        let output_tied = !has_lm_head;
        if output_tied {
            output_bytes += embd_bytes;
        }
        let dominant = by_dtype
            .iter()
            .max_by_key(|(d, b)| (*b, std::cmp::Reverse(d.clone())))
            .map_or("unknown".to_owned(), |(d, _)| d.clone());

        let mmproj = (vision_bytes > 0).then(|| {
            let vc = &top["vision_config"];
            let (tokens, basis) = match (
                top["mm_tokens_per_image"].as_u64(),
                vc["image_size"].as_u64(),
                vc["patch_size"].as_u64(),
            ) {
                (Some(t), _, _) => (t as u32, "mm_tokens_per_image in config.json".to_owned()),
                (None, Some(size), Some(patch)) if patch > 0 => {
                    let side = (size / patch) as u32;
                    (side * side, format!("({size}/{patch})^2 patches"))
                }
                _ => (576, "default 576, config states no image size".to_owned()),
            };
            let side = match (vc["image_size"].as_u64(), vc["patch_size"].as_u64()) {
                (Some(size), Some(patch)) if patch > 0 => (size / patch) as u32,
                _ => 24,
            };
            let u = |k: &str, d: u32| vc[k].as_u64().map_or(d, |v| v as u32);
            let n_embd = u("hidden_size", 1024);
            let p = crate::vision::Projector {
                kind: "config.json".into(),
                known: true,
                n_img_tokens: tokens,
                tokens_basis: basis.clone(),
                warmup_patches: side * side,
                n_embd,
                n_head: u("num_attention_heads", (n_embd / 64).max(1)),
                n_ff: u("intermediate_size", n_embd * 4),
            };
            Mmproj {
                path: dir.clone(),
                weight_bytes: vision_bytes,
                n_img_tokens: tokens,
                img_tokens_basis: basis,
                projector: p.kind.clone(),
                projector_known: true,
                compute_bytes: p.compute_bytes(),
                compute_basis: crate::info::compute_basis(&p),
            }
        });

        Ok(ModelInfo {
            path: dir,
            files,
            format: Format::Safetensors,
            name,
            arch,
            arch_known: spec.is_some(),
            n_params,
            n_layer,
            n_embd,
            n_vocab,
            n_head,
            n_head_kv,
            head_dim,
            n_ctx_train,
            n_ff,
            n_expert,
            n_expert_used: n_expert.and(cfg.u32("num_experts_per_tok")),
            n_ff_exp: n_expert.and(cfg.first_u32(&["moe_intermediate_size", "intermediate_size"])),
            n_expert_shared: n_expert
                .and(cfg.first_u32(&["n_shared_experts", "num_shared_experts"])),
            kv_layout,
            swa_window,
            quant: match quant {
                Quant::Float(_) => Quant::Float(dominant),
                q => q,
            },
            weight_bytes,
            layers,
            embd_bytes,
            output_bytes,
            output_tied,
            other_bytes,
            mmproj,
            chat_template,
            source: Source::Loose,
            complete: headers.iter().all(|h| h.complete),
        })
    }

    /// Open a model by path: a `.gguf` file, or a Hugging Face model directory.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, Error> {
        let path = path.as_ref();
        if path.is_dir() || path.file_name().is_some_and(|n| n == "config.json") {
            Self::from_hf_dir(path)
        } else {
            Self::from_gguf(path)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_tensor_names() {
        assert!(matches!(
            classify("model.layers.12.self_attn.q_proj.weight"),
            Part::Layer(12)
        ));
        assert!(matches!(
            classify("model.language_model.layers.3.mlp.down_proj.weight"),
            Part::Layer(3)
        ));
        assert!(matches!(
            classify("model.embed_tokens.weight"),
            Part::Embedding
        ));
        assert!(matches!(classify("lm_head.weight"), Part::Output));
        assert!(matches!(classify("model.norm.weight"), Part::Output));
        assert!(matches!(
            classify("vision_tower.vision_model.encoder.layers.0.mlp.fc1.weight"),
            Part::Vision
        ));
    }
}
