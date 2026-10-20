//! Model headers from the Hugging Face Hub, without downloading weights: the repo's file
//! list from the tree API, GGUF headers by growing HTTP range requests, safetensors headers
//! by two small ranges, and small JSON files whole. Everything fetched is cached under the
//! directory the caller gives, and that directory is the only place this module writes.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use loadmaster_model::gguf::{Error as GgufError, Gguf};
use loadmaster_model::safetensors::{self, HfParts, StHeader};
use loadmaster_model::{Mmproj, ModelInfo, Source, mmproj_candidates, split_part_names};
use serde::{Deserialize, Serialize};

pub const DEFAULT_ENDPOINT: &str = "https://huggingface.co";
/// First GGUF range; doubled until the header parses.
const GGUF_START: u64 = 4 << 20;
/// Give up past this: no real GGUF header is this large.
const GGUF_LIMIT: u64 = 256 << 20;
/// A safetensors JSON header larger than this is not one.
const ST_HEADER_LIMIT: u64 = 100 << 20;
/// How long a branch's file list is trusted; a commit hash never changes.
const TREE_TTL: Duration = Duration::from_secs(24 * 3600);
/// How long a search's results are reused.
const SEARCH_TTL: Duration = Duration::from_secs(3600);
/// Fields `GET /api/models` returns per model; with `expand[]` it returns only these
/// and `id` (checked against the live API, 2026-10-20; ADR 0014).
const SEARCH_EXPAND: [&str; 9] = [
    "downloads",
    "trendingScore",
    "createdAt",
    "tags",
    "pipeline_tag",
    "baseModels",
    "gguf",
    "safetensors",
    "gated",
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(
        "{0}: not found, or gated or private (the Hub answers both the same way); check the name, and for a gated repo accept its licence on huggingface.co and set HF_TOKEN"
    )]
    Gated(String),
    #[error("{0}: not found on the Hugging Face Hub")]
    NotFound(String),
    #[error("{url}: HTTP {status}")]
    Status { url: String, status: u16 },
    #[error("{url}: {reason}")]
    Http { url: String, reason: String },
    #[error("{url}: {reason}")]
    Format { url: String, reason: String },
    #[error(
        "{0}: Hugging Face rate limit reached; set HF_TOKEN for a higher limit, or retry in a few minutes"
    )]
    RateLimited(String),
    #[error("{0}: not in the local cache and --offline was given")]
    Offline(String),
    #[error("cache {}: {reason}", path.display())]
    Cache { path: PathBuf, reason: String },
    #[error("hf://{repo}: no file `{file}`{hint}")]
    NoSuchFile {
        repo: String,
        file: String,
        hint: String,
    },
    #[error(transparent)]
    Model(#[from] loadmaster_model::Error),
}

/// A file in a repo, as the tree API lists it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RemoteFile {
    pub path: String,
    pub size: u64,
}

/// `hf://org/repo[@rev][/path/in/repo]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spec {
    pub repo: String,
    pub rev: String,
    pub file: Option<String>,
}

impl Spec {
    pub fn parse(s: &str) -> Option<Spec> {
        let rest = s.strip_prefix("hf://")?;
        let mut parts = rest.splitn(3, '/');
        let org = parts.next().filter(|p| !p.is_empty())?;
        let name = parts.next().filter(|p| !p.is_empty())?;
        let file = parts.next().filter(|p| !p.is_empty()).map(str::to_owned);
        let (name, rev) = match name.split_once('@') {
            Some((n, r)) if !r.is_empty() => (n, r.to_owned()),
            _ => (name, "main".to_owned()),
        };
        Some(Spec {
            repo: format!("{org}/{name}"),
            rev,
            file,
        })
    }
}

/// A Hub model search (`GET /api/models`), newest-first or most-popular-first.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Search {
    /// Tags every result carries (`gguf`, `image-text-to-text`); the Hub ANDs them.
    pub filters: Vec<String>,
    /// Matched loosely against the repo id.
    pub text: Option<String>,
    /// `trendingScore`, `downloads` or `createdAt`, always descending.
    pub sort: &'static str,
    pub limit: u32,
}

/// One search result: what the listing says about a repo, before any file is read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HubModel {
    pub id: String,
    pub downloads: u64,
    pub trending: f64,
    pub created: Option<String>,
    /// Gated repos need an accepted licence and a token.
    pub gated: bool,
    pub pipeline: Option<String>,
    pub tags: Vec<String>,
    /// The model this one quantizes or fine-tunes, when the Hub records it.
    pub base: Option<String>,
    /// Parameter count (the Hub's `gguf.total` or `safetensors.total`).
    pub params: Option<u64>,
    /// GGUF architecture, when the Hub parsed one.
    pub arch: Option<String>,
    /// Safetensors parameter counts by dtype (`BF16`, `I32`, ...).
    pub dtypes: Vec<(String, u64)>,
}

impl HubModel {
    /// One entry of the `/api/models` response.
    pub fn from_json(v: &serde_json::Value) -> Option<HubModel> {
        let id = v.get("id")?.as_str()?.to_owned();
        let str_of = |k: &str| v.get(k).and_then(|x| x.as_str()).map(str::to_owned);
        let tags: Vec<String> = v
            .get("tags")
            .and_then(|t| t.as_array())
            .map(|t| {
                t.iter()
                    .filter_map(|x| x.as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default();
        let base = v
            .pointer("/baseModels/models/0/id")
            .and_then(|x| x.as_str())
            .map(str::to_owned)
            .or_else(|| {
                tags.iter().find_map(|t| {
                    t.strip_prefix("base_model:quantized:")
                        .or_else(|| t.strip_prefix("base_model:finetune:"))
                        .map(str::to_owned)
                })
            });
        let dtypes: Vec<(String, u64)> = v
            .pointer("/safetensors/parameters")
            .and_then(|p| p.as_object())
            .map(|m| {
                m.iter()
                    .filter_map(|(k, n)| Some((k.clone(), n.as_u64()?)))
                    .collect()
            })
            .unwrap_or_default();
        Some(HubModel {
            downloads: v.get("downloads").and_then(|x| x.as_u64()).unwrap_or(0),
            trending: v
                .get("trendingScore")
                .and_then(|x| x.as_f64())
                .unwrap_or(0.0),
            created: str_of("createdAt"),
            // `false`, or the kind of gate (`"auto"`, `"manual"`).
            gated: v.get("gated").is_some_and(|g| g.as_bool() != Some(false)),
            pipeline: str_of("pipeline_tag"),
            base,
            params: v
                .pointer("/gguf/total")
                .or_else(|| v.pointer("/safetensors/total"))
                .and_then(|x| x.as_u64()),
            arch: v
                .pointer("/gguf/architecture")
                .and_then(|x| x.as_str())
                .map(str::to_owned),
            dtypes,
            tags,
            id,
        })
    }
}

pub struct Hub {
    http: reqwest::blocking::Client,
    endpoint: String,
    token: Option<String>,
    cache: PathBuf,
    offline: bool,
}

impl Hub {
    /// `endpoint` is `$HF_ENDPOINT` or [`DEFAULT_ENDPOINT`]; `cache` is where headers go.
    pub fn new(endpoint: &str, token: Option<String>, cache: PathBuf, offline: bool) -> Self {
        Hub {
            http: reqwest::blocking::Client::builder()
                .timeout(Duration::from_secs(60))
                .user_agent(concat!("loadmaster/", env!("CARGO_PKG_VERSION")))
                .build()
                .expect("default client"),
            endpoint: endpoint.trim_end_matches('/').to_owned(),
            token,
            cache,
            offline,
        }
    }

    /// Whether requests carry a token (gated repos need one).
    pub fn has_token(&self) -> bool {
        self.token.is_some()
    }

    fn cache_path(&self, repo: &str, rev: &str, name: &str) -> PathBuf {
        self.cache.join(repo).join(rev).join(name)
    }

    fn read_cache(path: &Path, ttl: Option<Duration>) -> Option<Vec<u8>> {
        let meta = std::fs::metadata(path).ok()?;
        if let Some(ttl) = ttl {
            let age = SystemTime::now()
                .duration_since(meta.modified().ok()?)
                .unwrap_or_default();
            if age > ttl {
                return None;
            }
        }
        std::fs::read(path).ok()
    }

    fn write_cache(path: &Path, bytes: &[u8]) -> Result<(), Error> {
        let err = |e: std::io::Error| Error::Cache {
            path: path.to_owned(),
            reason: e.to_string(),
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(err)?;
        }
        // Write then rename, so an interrupted fetch never leaves a short header behind.
        let tmp = path.with_extension("partial");
        std::fs::write(&tmp, bytes).map_err(err)?;
        std::fs::rename(&tmp, path).map_err(err)
    }

    /// A commit hash names an immutable tree; anything else is a branch or tag.
    fn ttl(rev: &str) -> Option<Duration> {
        let pinned = rev.len() == 40 && rev.bytes().all(|b| b.is_ascii_hexdigit());
        (!pinned).then_some(TREE_TTL)
    }

    fn get(
        &self,
        url: &str,
        range: Option<(u64, u64)>,
    ) -> Result<reqwest::blocking::Response, Error> {
        let mut req = self.http.get(url);
        if let Some(t) = &self.token {
            req = req.bearer_auth(t);
        }
        if let Some((a, b)) = range {
            req = req.header(reqwest::header::RANGE, format!("bytes={a}-{b}"));
        }
        let resp = req.send().map_err(|e| Error::Http {
            url: url.to_owned(),
            reason: e.to_string(),
        })?;
        match resp.status().as_u16() {
            200..=299 => Ok(resp),
            401 | 403 => Err(Error::Gated(url.to_owned())),
            404 => Err(Error::NotFound(url.to_owned())),
            429 => Err(Error::RateLimited(url.to_owned())),
            status => Err(Error::Status {
                url: url.to_owned(),
                status,
            }),
        }
    }

    /// The first `n` bytes of `url` (fewer if the file is shorter), even from a server that
    /// ignores `Range`.
    fn prefix(&self, url: &str, n: u64) -> Result<Vec<u8>, Error> {
        let resp = self.get(url, Some((0, n - 1)))?;
        let mut out = Vec::new();
        resp.take(n)
            .read_to_end(&mut out)
            .map_err(|e| Error::Http {
                url: url.to_owned(),
                reason: e.to_string(),
            })?;
        Ok(out)
    }

    fn file_url(&self, repo: &str, rev: &str, path: &str) -> String {
        format!("{}/{repo}/resolve/{rev}/{path}", self.endpoint)
    }

    /// Every file in the repo, sub-directories included.
    pub fn list(&self, repo: &str, rev: &str) -> Result<Vec<RemoteFile>, Error> {
        let cached = self.cache_path(repo, rev, ".tree.json");
        let ttl = if self.offline { None } else { Self::ttl(rev) };
        if let Some(bytes) = Self::read_cache(&cached, ttl)
            && let Ok(files) = serde_json::from_slice(&bytes)
        {
            return Ok(files);
        }
        if self.offline {
            return Err(Error::Offline(format!("hf://{repo}@{rev}")));
        }
        #[derive(Deserialize)]
        struct Entry {
            #[serde(rename = "type")]
            kind: String,
            path: String,
            #[serde(default)]
            size: u64,
        }
        let mut url = Some(format!(
            "{}/api/models/{repo}/tree/{rev}?recursive=true",
            self.endpoint
        ));
        let mut files = Vec::new();
        while let Some(u) = url.take() {
            let resp = self.get(&u, None).map_err(|e| match e {
                Error::NotFound(_) => Error::NotFound(format!("hf://{repo}@{rev}")),
                Error::Gated(_) => Error::Gated(format!("hf://{repo}")),
                e => e,
            })?;
            // The tree API pages with `Link: <...>; rel="next"`.
            url = resp
                .headers()
                .get(reqwest::header::LINK)
                .and_then(|v| v.to_str().ok())
                .and_then(next_link);
            let page: Vec<Entry> = resp.json().map_err(|e| Error::Format {
                url: u.clone(),
                reason: e.to_string(),
            })?;
            files.extend(
                page.into_iter()
                    .filter(|e| e.kind == "file")
                    .map(|e| RemoteFile {
                        path: e.path,
                        size: e.size,
                    }),
            );
        }
        files.sort_by(|a, b| a.path.cmp(&b.path));
        let json = serde_json::to_vec(&files).expect("serialisable");
        Self::write_cache(&cached, &json)?;
        Ok(files)
    }

    /// Search the Hub's model listing. Results are cached for an hour, and any cached
    /// result is used with `--offline`.
    pub fn search(&self, q: &Search) -> Result<Vec<HubModel>, Error> {
        let mut params: Vec<(&str, String)> =
            q.filters.iter().map(|f| ("filter", f.clone())).collect();
        if let Some(t) = &q.text {
            params.push(("search", t.clone()));
        }
        params.push(("sort", q.sort.to_owned()));
        params.push(("limit", q.limit.to_string()));
        params.extend(SEARCH_EXPAND.iter().map(|e| ("expand[]", (*e).to_owned())));
        let base = format!("{}/api/models", self.endpoint);
        let url = reqwest::Url::parse_with_params(&base, &params).map_err(|e| Error::Http {
            url: base.clone(),
            reason: e.to_string(),
        })?;
        // FNV-1a of the query: a short, stable file name (the query itself is too long).
        let key = url
            .query()
            .unwrap_or_default()
            .bytes()
            .fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
            });
        let cached = self.cache.join(".search").join(format!("{key:016x}.json"));
        let ttl = if self.offline { None } else { Some(SEARCH_TTL) };
        let body = match Self::read_cache(&cached, ttl) {
            Some(b) => b,
            None if self.offline => {
                return Err(Error::Offline(format!(
                    "search {}",
                    q.text.as_deref().unwrap_or("(popular models)")
                )));
            }
            None => {
                let resp = self.get(url.as_str(), None)?;
                let b = resp.bytes().map_err(|e| Error::Http {
                    url: url.to_string(),
                    reason: e.to_string(),
                })?;
                Self::write_cache(&cached, &b)?;
                b.to_vec()
            }
        };
        let v: serde_json::Value = serde_json::from_slice(&body).map_err(|e| Error::Format {
            url: url.to_string(),
            reason: e.to_string(),
        })?;
        let list = v.as_array().ok_or_else(|| Error::Format {
            url: url.to_string(),
            reason: "expected a JSON list of models".into(),
        })?;
        Ok(list.iter().filter_map(HubModel::from_json).collect())
    }

    /// A GGUF file's header (metadata and tensor table), truncated to where tensor data
    /// starts. Returns the bytes; parse them with [`Gguf::parse_bytes`].
    pub fn gguf_header(&self, repo: &str, rev: &str, path: &str) -> Result<Vec<u8>, Error> {
        let cached = self.cache_path(repo, rev, &format!("{path}.header"));
        if let Some(bytes) = Self::read_cache(&cached, None) {
            return Ok(bytes);
        }
        if self.offline {
            return Err(Error::Offline(format!("hf://{repo}@{rev}/{path}")));
        }
        let bytes = fetch_gguf_header(&self.file_url(repo, rev, path), |url, n| {
            self.prefix(url, n)
        })?;
        Self::write_cache(&cached, &bytes)?;
        Ok(bytes)
    }

    /// A safetensors file's header: the 8-byte length and the JSON that follows.
    pub fn safetensors_header(&self, repo: &str, rev: &str, path: &str) -> Result<Vec<u8>, Error> {
        let cached = self.cache_path(repo, rev, &format!("{path}.header"));
        if let Some(bytes) = Self::read_cache(&cached, None) {
            return Ok(bytes);
        }
        if self.offline {
            return Err(Error::Offline(format!("hf://{repo}@{rev}/{path}")));
        }
        let url = self.file_url(repo, rev, path);
        let head = self.prefix(&url, 8)?;
        let len: [u8; 8] = head.as_slice().try_into().map_err(|_| Error::Format {
            url: url.clone(),
            reason: "shorter than 8 bytes; not a safetensors file".into(),
        })?;
        let n = u64::from_le_bytes(len);
        if n == 0 || n > ST_HEADER_LIMIT {
            return Err(Error::Format {
                url,
                reason: format!("header length {n} is not plausible; not a safetensors file"),
            });
        }
        let bytes = self.prefix(&url, 8 + n)?;
        if bytes.len() as u64 != 8 + n {
            return Err(Error::Format {
                url,
                reason: "file ends inside its header".into(),
            });
        }
        Self::write_cache(&cached, &bytes)?;
        Ok(bytes)
    }

    /// A small file whole (`config.json`, `tokenizer_config.json`, ...).
    pub fn small_file(&self, repo: &str, rev: &str, path: &str) -> Result<Vec<u8>, Error> {
        let cached = self.cache_path(repo, rev, path);
        if let Some(bytes) = Self::read_cache(&cached, None) {
            return Ok(bytes);
        }
        if self.offline {
            return Err(Error::Offline(format!("hf://{repo}@{rev}/{path}")));
        }
        let url = self.file_url(repo, rev, path);
        let resp = self.get(&url, None)?;
        let mut bytes = Vec::new();
        // Bounded: a "small" file over 64 MiB is a mistake, not a config.
        resp.take(64 << 20)
            .read_to_end(&mut bytes)
            .map_err(|e| Error::Http {
                url: url.clone(),
                reason: e.to_string(),
            })?;
        Self::write_cache(&cached, &bytes)?;
        Ok(bytes)
    }
}

impl Hub {
    /// A GGUF model in a repo from its header (every part, if split) and the projector
    /// next to it, if any. Nothing but headers is downloaded.
    pub fn open_gguf(
        &self,
        repo: &str,
        rev: &str,
        file: &str,
        files: &[RemoteFile],
    ) -> Result<ModelInfo, Error> {
        let size_of = |path: &str| files.iter().find(|f| f.path == path).map(|f| f.size);
        let Some(size) = size_of(file) else {
            let ggufs: Vec<&str> = files
                .iter()
                .map(|f| f.path.as_str())
                .filter(|p| p.ends_with(".gguf"))
                .collect();
            return Err(Error::NoSuchFile {
                repo: repo.into(),
                file: file.into(),
                hint: if ggufs.is_empty() {
                    "; the repo has no .gguf files".into()
                } else {
                    format!(
                        "; plan hf://{repo} to compare its {} GGUF files",
                        ggufs.len()
                    )
                },
            });
        };
        let header = |path: &str, size: u64| -> Result<Gguf, Error> {
            let bytes = self.gguf_header(repo, rev, path)?;
            let mut g = Gguf::parse_bytes(&bytes).map_err(loadmaster_model::Error::from)?;
            g.path = PathBuf::from(format!("hf://{repo}/{path}"));
            g.file_size = size;
            Ok(g)
        };
        let first = header(file, size)?;
        let (dir, name) = match file.rsplit_once('/') {
            Some((d, n)) => (format!("{d}/"), n),
            None => (String::new(), file),
        };
        let count = first.get_u32("split.count").unwrap_or(1);
        let names = if count > 1 {
            split_part_names(name, count).ok_or_else(|| loadmaster_model::Error::Split {
                path: first.path.clone(),
                reason: format!(
                    "header says {count} parts but the name does not match `<name>-00001-of-{count:05}.gguf`"
                ),
            })?
        } else {
            vec![name.to_owned()]
        };
        let mut parts = Vec::with_capacity(names.len());
        let mut first = Some(first);
        for n in &names {
            let path = format!("{dir}{n}");
            match first.take_if(|g| g.path.ends_with(&path)) {
                Some(g) => parts.push(g),
                None => {
                    let size = size_of(&path).ok_or_else(|| loadmaster_model::Error::Split {
                        path: PathBuf::from(format!("hf://{repo}/{path}")),
                        reason: format!("part of {count} is not in the repo"),
                    })?;
                    parts.push(header(&path, size)?);
                }
            }
        }
        let display = format!("hf://{repo}/{dir}{}", names[0]);
        let mut info = ModelInfo::from_gguf_parts(&display, &parts)?;
        info.source = Source::HfRemote {
            repo: repo.into(),
            revision: rev.into(),
            file: Some(format!("{dir}{}", names[0])),
        };
        // A projector in the same directory, by the same rule as on disk.
        let siblings: Vec<&str> = files
            .iter()
            .filter_map(|f| f.path.strip_prefix(dir.as_str()))
            .filter(|n| !n.contains('/') && !names.iter().any(|m| m == n))
            .collect();
        for c in mmproj_candidates(siblings) {
            let path = format!("{dir}{c}");
            let g = header(&path, size_of(&path).unwrap_or(0))?;
            let url = PathBuf::from(self.file_url(repo, rev, &path));
            if let Ok(m) = Mmproj::from_gguf(&g, &url) {
                info.mmproj = Some(m);
                break;
            }
        }
        Ok(info)
    }
}

impl Hub {
    /// A safetensors model repo from `config.json`, the shard index, the chat template and
    /// each shard's header. Nothing but headers and small JSON files is downloaded.
    pub fn open_safetensors(
        &self,
        repo: &str,
        rev: &str,
        files: &[RemoteFile],
    ) -> Result<ModelInfo, Error> {
        let has = |p: &str| files.iter().any(|f| f.path == p);
        let size_of = |p: &str| files.iter().find(|f| f.path == p).map_or(0, |f| f.size);
        let json = |p: &str| -> Result<serde_json::Value, Error> {
            let bytes = self.small_file(repo, rev, p)?;
            serde_json::from_slice(&bytes).map_err(|e| Error::Format {
                url: format!("hf://{repo}/{p}"),
                reason: format!("not valid JSON: {e}"),
            })
        };
        let config = json("config.json")?;
        let shards: Vec<String> = if has("model.safetensors.index.json") {
            safetensors::index_shards(&json("model.safetensors.index.json")?)
        } else {
            files
                .iter()
                .map(|f| f.path.clone())
                .filter(|p| !p.contains('/') && p.ends_with(".safetensors"))
                .collect()
        };
        if shards.is_empty() {
            return Err(Error::NoSuchFile {
                repo: repo.into(),
                file: "*.safetensors".into(),
                hint: String::new(),
            });
        }
        let headers = shards
            .iter()
            .map(|s| {
                let bytes = self.safetensors_header(repo, rev, s)?;
                let path = PathBuf::from(format!("hf://{repo}/{s}"));
                Ok(StHeader::from_bytes(&path, &bytes, size_of(s))?)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let jinja = if has("chat_template.jinja") {
            String::from_utf8(self.small_file(repo, rev, "chat_template.jinja")?).ok()
        } else {
            None
        };
        let tokenizer = if jinja.is_none() && has("tokenizer_config.json") {
            json("tokenizer_config.json").ok()
        } else {
            None
        };
        let pooling_config = if has("1_Pooling/config.json") {
            json("1_Pooling/config.json").ok()
        } else {
            None
        };
        let pooling = safetensors::sentence_transformers_pooling(
            safetensors::SENTENCE_TRANSFORMERS_MARKERS
                .iter()
                .any(|m| has(m)),
            pooling_config.as_ref(),
        );
        let mut info = ModelInfo::from_hf_parts(HfParts {
            path: PathBuf::from(format!("hf://{repo}")),
            name: repo.rsplit('/').next().map(str::to_owned),
            config,
            config_path: PathBuf::from(format!("hf://{repo}/config.json")),
            headers,
            chat_template: safetensors::chat_template_from(jinja, tokenizer.as_ref()),
            pooling,
        })?;
        info.source = Source::HfRemote {
            repo: repo.into(),
            revision: rev.into(),
            file: None,
        };
        Ok(info)
    }
}

fn next_link(header: &str) -> Option<String> {
    header.split(',').find_map(|part| {
        let (url, rel) = part.split_once(';')?;
        rel.contains("rel=\"next\"").then(|| {
            url.trim()
                .trim_start_matches('<')
                .trim_end_matches('>')
                .to_owned()
        })
    })
}

/// Fetch growing prefixes of `url` with `fetch(url, n)` until the GGUF header parses, and
/// return it truncated to the start of tensor data. Shared with `cargo xtask fetch-headers`.
pub fn fetch_gguf_header(
    url: &str,
    mut fetch: impl FnMut(&str, u64) -> Result<Vec<u8>, Error>,
) -> Result<Vec<u8>, Error> {
    let mut n = GGUF_START;
    loop {
        let mut bytes = fetch(url, n)?;
        match Gguf::parse_bytes(&bytes) {
            Ok(g) => {
                bytes.truncate(g.data_offset as usize);
                return Ok(bytes);
            }
            Err(GgufError::Truncated { .. }) if n < GGUF_LIMIT && bytes.len() as u64 == n => n *= 2,
            Err(e) => {
                return Err(Error::Format {
                    url: url.to_owned(),
                    reason: e.to_string(),
                });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_results_parse_the_expanded_fields() {
        // Shapes as the live API returned them on 2026-10-20 (values shortened).
        let v: serde_json::Value = serde_json::from_str(
            r#"[
              {"_id":"x","id":"unsloth/Qwen3.8-27B-GGUF","downloads":120345,"trendingScore":88,
               "createdAt":"2026-09-01T00:00:00.000Z","gated":false,"pipeline_tag":"text-generation",
               "tags":["gguf","base_model:quantized:Qwen/Qwen3.8-27B"],
               "baseModels":{"relation":"quantized","models":[{"_id":"y","id":"Qwen/Qwen3.8-27B"}]},
               "gguf":{"total":27320697856,"architecture":"qwen35","context_length":262144}},
              {"_id":"z","id":"Qwen/Qwen3-8B-AWQ","gated":"auto",
               "tags":["safetensors","base_model:quantized:Qwen/Qwen3-8B"],
               "safetensors":{"parameters":{"I32":1048576000,"F16":800000000},"total":1848576000}},
              {"_id":"w"}
            ]"#,
        )
        .unwrap();
        let m: Vec<HubModel> = v
            .as_array()
            .unwrap()
            .iter()
            .filter_map(HubModel::from_json)
            .collect();
        assert_eq!(m.len(), 2, "an entry without an id is dropped");
        assert_eq!(m[0].base.as_deref(), Some("Qwen/Qwen3.8-27B"));
        assert_eq!(m[0].params, Some(27320697856));
        assert_eq!(m[0].arch.as_deref(), Some("qwen35"));
        assert!(!m[0].gated);
        assert_eq!(m[1].base.as_deref(), Some("Qwen/Qwen3-8B"), "from the tag");
        assert!(m[1].gated, "a gate kind counts as gated");
        assert_eq!(m[1].params, Some(1848576000));
        assert_eq!(m[1].dtypes.len(), 2);
        assert_eq!(m[1].downloads, 0);
    }

    #[test]
    fn parses_specs() {
        assert_eq!(
            Spec::parse("hf://org/repo"),
            Some(Spec {
                repo: "org/repo".into(),
                rev: "main".into(),
                file: None
            })
        );
        assert_eq!(
            Spec::parse("hf://org/repo@abc123/BF16/m-00001-of-00002.gguf"),
            Some(Spec {
                repo: "org/repo".into(),
                rev: "abc123".into(),
                file: Some("BF16/m-00001-of-00002.gguf".into())
            })
        );
        assert_eq!(Spec::parse("hf://org"), None);
        assert_eq!(Spec::parse("org/repo"), None);
    }

    #[test]
    fn next_link_from_header() {
        let h = r#"<https://huggingface.co/api/models/a/b/tree/main?cursor=x>; rel="next""#;
        assert_eq!(
            next_link(h).as_deref(),
            Some("https://huggingface.co/api/models/a/b/tree/main?cursor=x")
        );
        assert_eq!(next_link(r#"<https://x>; rel="prev""#), None);
    }

    #[test]
    fn grows_the_prefix_until_the_header_parses() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/gguf");
        let file = std::fs::read(root.join("qwen3-8b/Qwen3-8B-Q4_K_M.gguf")).unwrap();
        let g = Gguf::parse_bytes(&file).unwrap();
        // A file that ends inside its header (fewer bytes than asked for) is an error.
        let got = fetch_gguf_header("mem://x", |_, _| Ok(file[..1024].to_vec()));
        assert!(matches!(got, Err(Error::Format { .. })));
        let mut asked = Vec::new();
        let got = fetch_gguf_header("mem://x", |_, n| {
            asked.push(n);
            Ok(file[..(n as usize).min(file.len())].to_vec())
        })
        .unwrap();
        assert_eq!(got.len() as u64, g.data_offset);
        assert_eq!(asked, vec![GGUF_START]);
    }
}
