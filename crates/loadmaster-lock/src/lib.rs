//! `models.lock`: the committed record of which models a machine serves and how.
//!
//! Plan-derived fields (`ctx`, `footprint_gb`, `flags`) are frozen when an entry is written
//! by `init`/`add` and change only with `loadmaster plan --update <name>`. Files written by
//! [`Lockfile::to_toml`] parse back to the same value and re-serialise byte for byte.
//! Unknown keys are errors.

mod diff;
pub mod import;
mod write;

use std::collections::HashSet;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};

use loadmaster_model::Source;
use loadmaster_plan::{Plan, Runtime};
use serde::{Deserialize, Serialize};

pub use diff::{Change, diff};

pub const VERSION: u32 = 1;
pub const FILE_NAME: &str = "models.lock";
pub const DEFAULT_LISTEN: &str = "127.0.0.1:8080";
pub const DEFAULT_IDLE_UNLOAD_S: u64 = 600;
pub const DEFAULT_LOAD_TIMEOUT_S: u64 = 300;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lockfile {
    pub version: u32,
    pub host: Host,
    #[serde(default)]
    pub proxy: Proxy,
    #[serde(default, rename = "model")]
    pub models: Vec<Model>,
}

/// The machine the entries were planned on. Informational: a mismatch at `up` is a warning.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Host {
    pub gpu: String,
    pub vram_gb: f64,
    pub headroom_gb: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Proxy {
    #[serde(default = "default_listen")]
    pub listen: String,
    /// Unload a model after this many seconds without requests (0 = never). `keep_warm`
    /// models are exempt.
    #[serde(default = "default_idle")]
    pub idle_unload_s: u64,
    /// How long a request waits for its model to load before a 503.
    #[serde(default = "default_load_timeout")]
    pub load_timeout_s: u64,
}

fn default_listen() -> String {
    DEFAULT_LISTEN.into()
}
fn default_idle() -> u64 {
    DEFAULT_IDLE_UNLOAD_S
}
fn default_load_timeout() -> u64 {
    DEFAULT_LOAD_TIMEOUT_S
}

impl Default for Proxy {
    fn default() -> Self {
        Proxy {
            listen: default_listen(),
            idle_unload_s: DEFAULT_IDLE_UNLOAD_S,
            load_timeout_s: DEFAULT_LOAD_TIMEOUT_S,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Model {
    pub name: String,
    /// `hf://org/repo[/file.gguf]`, `ollama://name:tag`, `lmstudio://publisher/repo/file`,
    /// or a filesystem path.
    pub source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub runtime: String,
    /// Resolved on `PATH`, or absolute.
    pub runtime_bin: String,
    pub ctx: u32,
    /// Derived and frozen. Absent for imported entries whose model file was not found;
    /// `loadmaster plan --update <name>` fills it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub footprint_gb: Option<f64>,
    /// Runtime flags, without the model path, context, host and port, which the supervisor
    /// adds at launch.
    #[serde(default)]
    pub flags: Vec<String>,
    /// Vision projector source, passed as `--mmproj` (llama.cpp).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mmproj: Option<String>,
    /// Extra environment for the runtime process, `KEY=VALUE`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub env: Vec<String>,
    #[serde(default)]
    pub keep_warm: bool,
    /// Names the proxy maps to this model.
    #[serde(default)]
    pub aliases: Vec<String>,
    /// Per-model override of `proxy.idle_unload_s`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub idle_unload_s: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub verify: Option<Verify>,
}

/// Written by `loadmaster up --verify`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Verify {
    pub probe_version: String,
    /// `pass` | `partial` | `fail`.
    pub verdict: String,
    #[serde(default)]
    pub failures: Vec<String>,
    /// Runtime and version the verdict applies to, e.g. `llama.cpp b6700`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub runtime_version: Option<String>,
    pub checked_at: String,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error("{path}: {message}")]
    Parse { path: PathBuf, message: String },
    #[error(
        "{path}: lockfile version {found} is newer than this loadmaster understands ({VERSION}); upgrade loadmaster"
    )]
    TooNew { path: PathBuf, found: u32 },
    #[error(
        "{path}: lockfile version {found} is not supported; re-create it with `loadmaster init`"
    )]
    TooOld { path: PathBuf, found: u32 },
    #[error("{path}: {problems}", problems = .problems.join("; "))]
    Invalid {
        path: PathBuf,
        problems: Vec<String>,
    },
    #[error("{path} already exists; pass --force to overwrite it")]
    Exists { path: PathBuf },
}

impl Lockfile {
    pub fn new(host: Host) -> Self {
        Lockfile {
            version: VERSION,
            host,
            proxy: Proxy::default(),
            models: Vec::new(),
        }
    }

    pub fn load(path: &Path) -> Result<Self, Error> {
        let text = std::fs::read_to_string(path).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })?;
        Self::parse(&text, path)
    }

    /// Parse, migrate and validate. `path` is only used in messages.
    pub fn parse(text: &str, path: &Path) -> Result<Self, Error> {
        let parse_err = |message: String| Error::Parse {
            path: path.to_path_buf(),
            message,
        };
        let raw: toml::Table = toml::from_str(text).map_err(|e| parse_err(e.to_string()))?;
        let version = raw
            .get("version")
            .and_then(toml::Value::as_integer)
            .ok_or_else(|| parse_err("missing `version`".into()))?;
        let found =
            u32::try_from(version).map_err(|_| parse_err(format!("bad version {version}")))?;
        if found > VERSION {
            return Err(Error::TooNew {
                path: path.to_path_buf(),
                found,
            });
        }
        if found < VERSION {
            // No older versions exist yet; migrations from `found` go here.
            return Err(Error::TooOld {
                path: path.to_path_buf(),
                found,
            });
        }
        let lock: Lockfile = toml::from_str(text).map_err(|e| parse_err(e.to_string()))?;
        let problems = lock.problems();
        if !problems.is_empty() {
            return Err(Error::Invalid {
                path: path.to_path_buf(),
                problems,
            });
        }
        Ok(lock)
    }

    /// Everything wrong with this lockfile, as messages naming the entry.
    pub fn problems(&self) -> Vec<String> {
        let mut out = Vec::new();
        if self.proxy.listen.parse::<SocketAddr>().is_err() {
            out.push(format!(
                "proxy.listen `{}` is not an address like 127.0.0.1:8080",
                self.proxy.listen
            ));
        }
        let mut names = HashSet::new();
        for m in &self.models {
            let id = format!("model `{}`", m.name);
            if m.name.is_empty() || m.name.chars().any(char::is_whitespace) {
                out.push(format!("{id}: name must be non-empty with no spaces"));
            }
            if !names.insert(m.name.as_str()) {
                out.push(format!("{id}: duplicate name"));
            }
            if runtime(&m.runtime).is_none() {
                out.push(format!(
                    "{id}: runtime `{}` is not llama.cpp, vllm or exl3",
                    m.runtime
                ));
            }
            if m.ctx == 0 {
                out.push(format!("{id}: ctx must be positive"));
            }
            if m.runtime_bin.is_empty() {
                out.push(format!("{id}: runtime_bin is empty"));
            }
            if m.footprint_gb.is_some_and(|f| !(f.is_finite() && f > 0.0)) {
                out.push(format!("{id}: footprint_gb must be positive"));
            }
            if let Some(e) = m.env.iter().find(|e| !e.contains('=')) {
                out.push(format!("{id}: env entry `{e}` is not KEY=VALUE"));
            }
            for f in &m.flags {
                if ["--port", "--host", "-m", "--model"].contains(&f.as_str()) {
                    out.push(format!(
                        "{id}: flags must not contain `{f}`; the supervisor sets it"
                    ));
                }
            }
        }
        for m in &self.models {
            for a in &m.aliases {
                if !names.insert(a.as_str()) {
                    out.push(format!(
                        "model `{}`: alias `{a}` collides with another name or alias",
                        m.name
                    ));
                }
            }
        }
        out
    }

    pub fn model(&self, name: &str) -> Option<&Model> {
        self.models.iter().find(|m| m.name == name)
    }

    /// The entry a request's `model` field refers to: by name, then by alias.
    pub fn route(&self, requested: &str) -> Option<&Model> {
        self.model(requested).or_else(|| {
            self.models
                .iter()
                .find(|m| m.aliases.iter().any(|a| a == requested))
        })
    }

    /// Canonical TOML; see the module docs.
    pub fn to_toml(&self) -> String {
        write::to_toml(self)
    }

    /// Write to `path`, refusing to replace an existing file unless `force`.
    pub fn save(&self, path: &Path, force: bool) -> Result<(), Error> {
        if path.exists() && !force {
            return Err(Error::Exists {
                path: path.to_path_buf(),
            });
        }
        std::fs::write(path, self.to_toml()).map_err(|source| Error::Io {
            path: path.to_path_buf(),
            source,
        })
    }
}

pub fn runtime(name: &str) -> Option<Runtime> {
    match name {
        "llama.cpp" => Some(Runtime::LlamaCpp),
        "vllm" => Some(Runtime::Vllm),
        "exl3" => Some(Runtime::Exl3),
        _ => None,
    }
}

/// GiB with at most two decimals, the unit of `vram_gb`, `headroom_gb`, `footprint_gb`.
pub fn gb(bytes: u64) -> f64 {
    (bytes as f64 / (1u64 << 30) as f64 * 100.0).ceil() / 100.0
}

pub fn gb_to_bytes(gb: f64) -> u64 {
    (gb * (1u64 << 30) as f64).round() as u64
}

/// The lockfile `source` string for a model.
pub fn source_string(source: &Source, path: &Path) -> String {
    match source {
        Source::HfCache { repo, .. } => {
            // Inside `snapshots/<revision>/`: a GGUF keeps its relative path, a model
            // directory is the repo itself.
            let parts: Vec<String> = path
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect();
            let rel = parts
                .iter()
                .position(|c| c == "snapshots")
                .map(|i| parts[(i + 2).min(parts.len())..].join("/"))
                .unwrap_or_default();
            if rel.is_empty() || path.is_dir() {
                format!("hf://{repo}")
            } else {
                format!("hf://{repo}/{rel}")
            }
        }
        Source::Ollama { name, .. } => format!("ollama://{name}"),
        Source::LmStudio { .. } => path.display().to_string(),
        Source::Loose => path.display().to_string(),
    }
}

impl Model {
    /// A lockfile entry from a plan. Flags are the plan's argv without the binary, model,
    /// projector, context, host and port.
    pub fn from_plan(name: &str, source: String, sha256: Option<String>, plan: &Plan) -> Model {
        let (runtime_bin, flags) = launch_flags(plan);
        Model {
            name: name.to_owned(),
            source,
            sha256,
            runtime: plan.runtime.name().to_owned(),
            runtime_bin,
            ctx: plan.ctx,
            footprint_gb: Some(gb(plan.footprint.gpu_bytes)),
            flags,
            mmproj: plan.model.mmproj.clone(),
            env: Vec::new(),
            keep_warm: false,
            aliases: Vec::new(),
            idle_unload_s: None,
            verify: None,
        }
    }
}

/// Split a plan's argv into the runtime binary and the frozen flags.
fn launch_flags(plan: &Plan) -> (String, Vec<String>) {
    let argv = &plan.argv;
    let bin = argv.first().cloned().unwrap_or_default();
    let mut flags = Vec::new();
    let mut i = 1;
    // `vllm serve <model>`: the subcommand and model are positional.
    if plan.runtime == Runtime::Vllm {
        i = 3;
    }
    while i < argv.len() {
        let a = argv[i].as_str();
        let takes_value = [
            "-m",
            "--model",
            "--mmproj",
            "-c",
            "--ctx-size",
            "--max-model-len",
            "--port",
            "--host",
        ];
        if takes_value.contains(&a) {
            i += 2;
            continue;
        }
        flags.push(argv[i].clone());
        i += 1;
    }
    match plan.runtime {
        Runtime::Exl3 => {
            let f = vec![
                "--cache-mode".to_owned(),
                loadmaster_plan::emit::exl3::cache_mode(plan.kv).to_owned(),
                "--chunk-size".to_owned(),
                plan.n_ubatch.to_string(),
            ];
            ("tabbyAPI".to_owned(), f)
        }
        _ => (bin, flags),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gb_rounds_up_to_hundredths() {
        assert_eq!(gb(1 << 30), 1.0);
        assert_eq!(gb((1 << 30) + 1), 1.01);
    }
}
