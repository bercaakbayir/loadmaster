//! Read-only scanners for local model stores: the Hugging Face hub cache, Ollama's
//! manifests and blobs, LM Studio's models directory, and loose directories.
//!
//! The scanners write nothing and follow nothing outside the roots they are given.
//! Environment variables are read by the caller and passed in as [`Roots`]. [`remote`]
//! fetches headers from the Hugging Face Hub and writes only into the cache it is given.

mod hf;
mod lmstudio;
mod loose;
mod ollama;
pub mod remote;

use std::path::{Path, PathBuf};

use loadmaster_model::{ModelInfo, Source};
use serde::Serialize;

/// Where to look.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Roots {
    pub hf_hub: Option<PathBuf>,
    pub ollama: Option<PathBuf>,
    pub lmstudio: Vec<PathBuf>,
    pub extra: Vec<PathBuf>,
}

impl Roots {
    /// Default locations, given `$HOME` and an environment lookup (`HF_HUB_CACHE`,
    /// `HF_HOME`, `OLLAMA_MODELS`).
    pub fn defaults(home: &Path, env: impl Fn(&str) -> Option<String>) -> Self {
        let hf_hub = env("HF_HUB_CACHE")
            .map(PathBuf::from)
            .or_else(|| env("HF_HOME").map(|h| PathBuf::from(h).join("hub")))
            .unwrap_or_else(|| home.join(".cache/huggingface/hub"));
        let ollama = env("OLLAMA_MODELS")
            .map(PathBuf::from)
            .unwrap_or_else(|| home.join(".ollama/models"));
        Roots {
            hf_hub: Some(hf_hub),
            ollama: Some(ollama),
            lmstudio: vec![
                home.join(".lmstudio/models"),
                home.join(".cache/lm-studio/models"),
            ],
            extra: Vec::new(),
        }
    }
}

/// One model found in a store.
#[derive(Debug, Clone, Serialize)]
pub struct Entry {
    /// What to pass to `loadmaster plan`: `llama3.2:1b`, `org/repo`, `org/repo/file.gguf`.
    pub name: String,
    pub path: PathBuf,
    pub source: Source,
    /// Bytes on disk (all parts of a split model), counted once per blob.
    pub size_bytes: u64,
    /// Vision projector next to the model, when the store records one separately.
    #[serde(skip)]
    pub mmproj: Option<PathBuf>,
    /// The model's header, or why it could not be read.
    #[serde(skip)]
    pub info: Result<Box<ModelInfo>, String>,
}

impl Entry {
    fn new(name: String, path: PathBuf, source: Source, mmproj: Option<PathBuf>) -> Self {
        let info = ModelInfo::open(&path)
            .map(|mut m| {
                m.source = source.clone();
                if let Some(p) = &mmproj
                    && let Ok(mm) = loadmaster_model::Mmproj::open(p)
                {
                    m.mmproj = Some(mm);
                }
                Box::new(m)
            })
            .map_err(|e| e.to_string());
        let size_bytes = match &info {
            Ok(m) => m
                .files
                .iter()
                .filter_map(|f| f.metadata().ok())
                .map(|md| md.len())
                .sum(),
            Err(_) => path.metadata().map_or(0, |m| m.len()),
        };
        Entry {
            name,
            path,
            source,
            size_bytes,
            mmproj,
            info,
        }
    }
}

/// A store problem worth telling the user about (broken symlink, unreadable manifest).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Issue {
    pub path: PathBuf,
    pub problem: String,
}

#[derive(Debug, Default)]
pub struct Scan {
    pub entries: Vec<Entry>,
    pub issues: Vec<Issue>,
}

/// Scan every root. Entries are sorted by size (largest first), then name; the same file
/// reached twice (a blob shared by two revisions or tags) is listed once.
pub fn scan(roots: &Roots) -> Scan {
    let mut out = Scan::default();
    if let Some(r) = &roots.hf_hub {
        hf::scan(r, &mut out);
    }
    if let Some(r) = &roots.ollama {
        ollama::scan(r, &mut out);
    }
    for r in &roots.lmstudio {
        lmstudio::scan(r, &mut out);
    }
    for r in &roots.extra {
        loose::scan(r, &mut out);
    }
    let mut seen = std::collections::HashSet::new();
    out.entries.retain(|e| {
        let key = std::fs::canonicalize(&e.path).unwrap_or_else(|_| e.path.clone());
        seen.insert(key)
    });
    out.entries.sort_by(|a, b| {
        b.size_bytes
            .cmp(&a.size_bytes)
            .then_with(|| a.name.cmp(&b.name))
    });
    out.issues.sort_by(|a, b| a.path.cmp(&b.path));
    out
}

/// Find an entry by `name`, `hf://name`, or a unique file-stem match.
pub fn resolve<'a>(entries: &'a [Entry], query: &str) -> Result<&'a Entry, String> {
    let q = query.strip_prefix("hf://").unwrap_or(query);
    if let Some(e) = entries.iter().find(|e| e.name == q) {
        return Ok(e);
    }
    let stem = |e: &Entry| {
        e.path
            .file_stem()
            .map(|s| s.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default()
    };
    let ql = q.to_ascii_lowercase();
    let matches: Vec<&Entry> = entries
        .iter()
        .filter(|e| stem(e) == ql || e.name.to_ascii_lowercase().ends_with(&format!("/{ql}")))
        .collect();
    match matches.as_slice() {
        [one] => Ok(one),
        [] => Err(format!(
            "no model named `{query}` in the scanned stores; run `loadmaster scan` to list them, or pass a path"
        )),
        many => Err(format!(
            "`{query}` matches {} models ({}); use the full name",
            many.len(),
            many.iter()
                .map(|e| e.name.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// Directory entries sorted by name; unreadable directories are empty.
fn sorted_dir(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).collect())
        .unwrap_or_default();
    v.sort();
    v
}

fn is_gguf(p: &Path) -> bool {
    p.extension()
        .is_some_and(|x| x.eq_ignore_ascii_case("gguf"))
}

fn is_mmproj(p: &Path) -> bool {
    p.file_name()
        .is_some_and(|n| n.to_string_lossy().to_ascii_lowercase().contains("mmproj"))
}

/// Skip non-first parts of split GGUFs; the first part stands for the model.
fn is_later_split_part(p: &Path) -> bool {
    let name = p
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    match name.find("-of-") {
        Some(i) if i >= 5 => {
            let part = &name[i - 5..i];
            part.bytes().all(|b| b.is_ascii_digit()) && part != "00001"
        }
        _ => false,
    }
}

/// A directory holding an HF model: `config.json` plus at least one `.safetensors`.
fn is_hf_model_dir(dir: &Path) -> bool {
    dir.join("config.json").exists()
        && sorted_dir(dir)
            .iter()
            .any(|p| p.extension().is_some_and(|x| x == "safetensors"))
}

/// The declared architecture when `config.json` names one that does not generate text
/// (encoders, embedding and classification models).
fn non_generative(dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(dir.join("config.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let archs: Vec<&str> = v["architectures"]
        .as_array()?
        .iter()
        .filter_map(|a| a.as_str())
        .collect();
    let generative = archs.iter().any(|a| {
        a.ends_with("ForCausalLM")
            || a.ends_with("ForConditionalGeneration")
            || a.ends_with("LMHeadModel")
    });
    (!archs.is_empty() && !generative).then(|| archs.join(", "))
}

/// GGUF models in `dir` (not projectors, not later split parts) and an HF model if the
/// directory is one. `name_of` builds the entry name from a file name.
fn models_in_dir(
    dir: &Path,
    source: &Source,
    out: &mut Scan,
    hf_name: Option<String>,
    name_of: impl Fn(&str) -> String,
) {
    for p in sorted_dir(dir) {
        if !is_gguf(&p) || is_mmproj(&p) || is_later_split_part(&p) {
            continue;
        }
        if !p.exists() {
            out.issues.push(Issue {
                path: p.clone(),
                problem: "broken symlink (partial or deleted download)".into(),
            });
            continue;
        }
        let file = p
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        out.entries
            .push(Entry::new(name_of(&file), p, source.clone(), None));
    }
    if let Some(name) = hf_name
        && is_hf_model_dir(dir)
    {
        match non_generative(dir) {
            Some(arch) => out.issues.push(Issue {
                path: dir.to_path_buf(),
                problem: format!("{arch} is not a text-generation model; skipped"),
            }),
            None => out
                .entries
                .push(Entry::new(name, dir.to_path_buf(), source.clone(), None)),
        }
    }
}
