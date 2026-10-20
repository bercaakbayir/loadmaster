//! `plan hf://...` against a fake Hugging Face Hub (tree API and `Range` downloads served
//! from the fixtures), so no test touches the network.

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::Arc;

use loadmaster_model::gguf::Gguf;

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A file in a fake repo: the bytes served, and the full size the tree API reports.
#[derive(Clone)]
struct File {
    bytes: Vec<u8>,
    size: u64,
}

fn gguf(rel: &str) -> File {
    let bytes = std::fs::read(workspace().join("fixtures/gguf").join(rel)).unwrap();
    let g = Gguf::parse_bytes(&bytes).unwrap();
    let end = g
        .tensors
        .iter()
        .map(|t| t.offset + t.n_bytes)
        .max()
        .unwrap();
    File {
        size: g.data_offset + end,
        bytes,
    }
}

/// A file from `fixtures/hf/<dir>`; safetensors shards report their full size.
fn hf_file(dir: &str, name: &str) -> File {
    let bytes = std::fs::read(workspace().join("fixtures/hf").join(dir).join(name)).unwrap();
    let size = if name.ends_with(".safetensors") {
        let n = u64::from_le_bytes(bytes[..8].try_into().unwrap());
        let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + n as usize]).unwrap();
        let end = header
            .as_object()
            .unwrap()
            .values()
            .filter_map(|t| t["data_offsets"][1].as_u64())
            .max()
            .unwrap();
        8 + n + end
    } else {
        bytes.len() as u64
    };
    File { bytes, size }
}

/// Repos keyed by `org/name`, then by path in the repo. `private` repos need the token.
struct Hub {
    repos: HashMap<String, Vec<(String, File)>>,
    private: Vec<String>,
    /// What `GET /api/models` lists, in the Hub's (trending) order.
    listing: Vec<serde_json::Value>,
}

fn hub() -> Hub {
    let mut repos = HashMap::new();
    let q8 = gguf("qwen3-8b/Qwen3-8B-Q4_K_M.gguf");
    let size = q8.size;
    repos.insert(
        "fx/qwen3-8b-GGUF".into(),
        vec![
            ("Qwen3-8B-Q4_K_M.gguf".into(), q8),
            // Listed only: a bare-repo plan estimates it from the size.
            (
                "Qwen3-8B-Q8_0.gguf".into(),
                File {
                    bytes: Vec::new(),
                    size: size * 17 / 10,
                },
            ),
        ],
    );
    repos.insert(
        "fx/gemma-3-12b-GGUF".into(),
        vec![
            (
                "gemma-3-12b-it-Q4_K_M.gguf".into(),
                gguf("gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf"),
            ),
            (
                "mmproj-gemma-3-12b-it-F16.gguf".into(),
                gguf("gemma-3-12b/mmproj-gemma-3-12b-it-F16.gguf"),
            ),
        ],
    );
    repos.insert(
        "ggml-org/ultravox-8b-GGUF".into(),
        vec![
            (
                "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf".into(),
                gguf("ultravox-llama-3.1-8b/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"),
            ),
            (
                "mmproj-ultravox-v0_5-llama-3_1-8b-f16.gguf".into(),
                gguf("ultravox-llama-3.1-8b/mmproj-ultravox-v0_5-llama-3_1-8b-f16.gguf"),
            ),
        ],
    );
    let awq = "Qwen3-8B-AWQ";
    repos.insert(
        "fx/Qwen3-8B-AWQ".into(),
        std::fs::read_dir(workspace().join("fixtures/hf").join(awq))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .map(|n| {
                let f = hf_file(awq, &n);
                (n, f)
            })
            .collect(),
    );
    repos.insert(
        "someone/qwen3-8b-finetune-GGUF".into(),
        vec![(
            "Qwen3-8B-Q4_K_M.gguf".into(),
            gguf("qwen3-8b/Qwen3-8B-Q4_K_M.gguf"),
        )],
    );
    repos.insert(
        "fx/private".into(),
        vec![(
            "Qwen3-8B-Q4_K_M.gguf".into(),
            gguf("qwen3-8b/Qwen3-8B-Q4_K_M.gguf"),
        )],
    );
    // Listed only, and far too big: suggest must rule it out from its size, never
    // fetching the (empty, unparseable) header.
    repos.insert(
        "fx/huge-GGUF".into(),
        vec![(
            "huge-Q4_K_M.gguf".into(),
            File {
                bytes: Vec::new(),
                size: 500 << 30,
            },
        )],
    );
    // `suggest` lists known publishers only: the same fixtures under their names.
    for (alias, repo) in [
        ("ggml-org/qwen3-8b-GGUF", "fx/qwen3-8b-GGUF"),
        ("google/gemma-3-12b-GGUF", "fx/gemma-3-12b-GGUF"),
        ("Qwen/huge-GGUF", "fx/huge-GGUF"),
        ("Qwen/Qwen3-8B-AWQ", "fx/Qwen3-8B-AWQ"),
    ] {
        let files = repos[repo].clone();
        repos.insert(alias.into(), files);
    }
    let listing = vec![
        serde_json::json!({"id": "ggml-org/qwen3-8b-GGUF", "tags": ["gguf", "text-generation"],
            "baseModels": {"relation": "quantized", "models": [{"id": "fx/Qwen3-8B"}]},
            "gguf": {"total": 8_190_735_360u64, "architecture": "qwen3"}, "downloads": 900}),
        serde_json::json!({"id": "unsloth/qwen3-8b-GGUF", "tags": ["gguf", "base_model:quantized:fx/Qwen3-8B"],
            "downloads": 800}),
        // A community fine-tune: shown only with --any-publisher.
        serde_json::json!({"id": "someone/qwen3-8b-finetune-GGUF", "tags": ["gguf", "base_model:finetune:fx/Qwen3-8B-Chat"],
            "gguf": {"total": 8_190_735_360u64}}),
        serde_json::json!({"id": "google/gemma-3-12b-GGUF", "tags": ["gguf", "image-text-to-text"],
            "baseModels": {"relation": "quantized", "models": [{"id": "fx/gemma-3-12b-it"}]},
            "gguf": {"total": 12_187_000_000u64, "architecture": "gemma3"}}),
        serde_json::json!({"id": "ggml-org/ultravox-8b-GGUF", "tags": ["gguf", "audio-text-to-text"],
            "baseModels": {"relation": "quantized", "models": [{"id": "fx/ultravox-8b"}]},
            "gguf": {"total": 8_030_261_248u64, "architecture": "llama"}}),
        serde_json::json!({"id": "Qwen/huge-GGUF", "tags": ["gguf"], "gguf": {"total": 1_000_000_000_000u64}}),
        serde_json::json!({"id": "fx/private", "tags": ["gguf"], "gated": "manual"}),
        serde_json::json!({"id": "Qwen/Qwen3-8B-AWQ", "tags": ["safetensors", "text-generation"],
            "safetensors": {"parameters": {"I32": 1_000_000_000u64, "F16": 800_000_000u64}, "total": 1_800_000_000u64}}),
    ];
    Hub {
        repos,
        private: vec!["fx/private".into()],
        listing,
    }
}

/// `GET /api/models?filter=..&search=..`: listing entries carrying every filter tag whose
/// id contains the search text. Searching `ratelimit` answers 429.
fn search(hub: &Hub, query: &str) -> (&'static str, Vec<u8>) {
    let mut filters = Vec::new();
    let mut text = None;
    for pair in query.split('&') {
        let (k, v) = pair.split_once('=').unwrap_or((pair, ""));
        let v = v.replace("%20", " ").replace('+', " ");
        match k {
            "filter" => filters.push(v),
            "search" => text = Some(v.to_lowercase()),
            _ => {}
        }
    }
    if text.as_deref() == Some("ratelimit") {
        return ("429 Too Many Requests", Vec::new());
    }
    let hits: Vec<&serde_json::Value> = hub
        .listing
        .iter()
        .filter(|m| {
            let tags: Vec<&str> = m["tags"]
                .as_array()
                .unwrap()
                .iter()
                .map(|t| t.as_str().unwrap())
                .collect();
            filters.iter().all(|f| tags.contains(&f.as_str()))
                && text
                    .as_deref()
                    .is_none_or(|t| m["id"].as_str().unwrap().to_lowercase().contains(t))
        })
        .collect();
    ("200 OK", serde_json::to_vec(&hits).unwrap())
}

fn respond(mut stream: std::net::TcpStream, hub: &Hub) {
    let mut reader = BufReader::new(stream.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let target = line.split_whitespace().nth(1).unwrap_or("/").to_owned();
    let (mut range, mut auth) = (None, None);
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h).unwrap() == 0 || h == "\r\n" {
            break;
        }
        let (k, v) = h.split_once(':').unwrap_or((&h, ""));
        match k.to_ascii_lowercase().as_str() {
            "range" => range = Some(v.trim().to_owned()),
            "authorization" => auth = Some(v.trim().to_owned()),
            _ => {}
        }
    }
    let mut send = |status: &str, body: &[u8]| {
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n",
            body.len()
        );
        let _ = stream.write_all(body);
    };
    let path = target.split('?').next().unwrap();
    if path == "/api/models" {
        let (status, body) = search(hub, target.split_once('?').map_or("", |(_, q)| q));
        return send(status, &body);
    }
    let (repo, rest) = if let Some(p) = path.strip_prefix("/api/models/") {
        let mut it = p.splitn(3, '/');
        let repo = format!("{}/{}", it.next().unwrap(), it.next().unwrap_or(""));
        (repo, None)
    } else {
        let p = path.trim_start_matches('/');
        let mut it = p.splitn(5, '/');
        let repo = format!("{}/{}", it.next().unwrap(), it.next().unwrap_or(""));
        let _resolve = it.next();
        let _rev = it.next();
        (repo, it.next().map(str::to_owned))
    };
    let Some(files) = hub.repos.get(&repo) else {
        return send("401 Unauthorized", b"");
    };
    if hub.private.contains(&repo) && auth.as_deref() != Some("Bearer secret") {
        return send("401 Unauthorized", b"");
    }
    match rest {
        None => {
            let tree: Vec<_> = files
                .iter()
                .map(|(p, f)| serde_json::json!({"type": "file", "path": p, "size": f.size}))
                .collect();
            send("200 OK", serde_json::to_string(&tree).unwrap().as_bytes());
        }
        Some(file) => {
            let Some((_, f)) = files.iter().find(|(p, _)| *p == file) else {
                return send("404 Not Found", b"");
            };
            let (a, b) = range
                .as_deref()
                .and_then(|r| r.strip_prefix("bytes="))
                .and_then(|r| r.split_once('-'))
                .map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap()))
                .unwrap_or((0, f.bytes.len().saturating_sub(1)));
            let end = (b + 1).min(f.bytes.len());
            send("206 Partial Content", &f.bytes[a.min(end)..end]);
        }
    }
}

/// Serve the fake hub on a free port for the life of the test process.
fn serve() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let hub = Arc::new(hub());
    std::thread::spawn(move || {
        for s in listener.incoming().flatten() {
            let hub = hub.clone();
            std::thread::spawn(move || respond(s, &hub));
        }
    });
    url
}

fn run(endpoint: &str, home: &Path, token: Option<&str>, args: &[&str]) -> Output {
    let mut c = Command::new(env!("CARGO_BIN_EXE_loadmaster"));
    c.args(args)
        .env("HF_ENDPOINT", endpoint)
        .env("LOADMASTER_HOME", home.join("state"))
        .env("HOME", home)
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("OLLAMA_MODELS")
        .env_remove("HF_TOKEN");
    if let Some(t) = token {
        c.env("HF_TOKEN", t);
    }
    c.output().unwrap()
}

fn text(o: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

const PLAN: [&str; 5] = ["plan", "--pretend", "rtx 4090", "--ctx", "8192"];

#[test]
fn plans_a_hub_gguf_from_its_header() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let spec = "hf://fx/qwen3-8b-GGUF/Qwen3-8B-Q4_K_M.gguf";
    let o = run(&url, home.path(), None, &[&PLAN[..], &[spec]].concat());
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}{err}");
    assert!(
        out.contains("--hf-repo fx/qwen3-8b-GGUF --hf-file Qwen3-8B-Q4_K_M.gguf"),
        "{out}"
    );
    assert!(!out.contains("incomplete download"), "{out}");

    // Cached: the same plan offline, with the server unreachable.
    let o = run(
        "http://127.0.0.1:9",
        home.path(),
        None,
        &[&PLAN[..], &[spec, "--offline"]].concat(),
    );
    assert_eq!(text(&o).0, out);
}

#[test]
fn picks_up_the_projector_next_to_the_model() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let o = run(
        &url,
        home.path(),
        None,
        &[
            &PLAN[..],
            &["hf://fx/gemma-3-12b-GGUF/gemma-3-12b-it-Q4_K_M.gguf"],
        ]
        .concat(),
    );
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}{err}");
    assert!(
        out.contains(&format!(
            "--mmproj-url {url}/fx/gemma-3-12b-GGUF/resolve/main/mmproj-gemma-3-12b-it-F16.gguf"
        )),
        "{out}"
    );
    assert!(out.contains("on GPU, one image"), "{out}");

    // An audio projector the same way, sized for one 30 s clip.
    let o = run(
        &url,
        home.path(),
        None,
        &[
            &PLAN[..],
            &["hf://ggml-org/ultravox-8b-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf"],
        ]
        .concat(),
    );
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}{err}");
    assert!(
        out.contains("resolve/main/mmproj-ultravox-v0_5-llama-3_1-8b-f16.gguf"),
        "{out}"
    );
    assert!(out.contains("on GPU, one 30 s clip"), "{out}");
}

#[test]
fn gated_repos_need_the_token() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let args = [&PLAN[..], &["hf://fx/private/Qwen3-8B-Q4_K_M.gguf"]].concat();
    let o = run(&url, home.path(), None, &args);
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("HF_TOKEN"), "{}", text(&o).1);
    let o = run(&url, home.path(), Some("secret"), &args);
    assert_eq!(o.status.code(), Some(0), "{}", text(&o).1);
}

#[test]
fn missing_files_and_offline_misses_are_errors() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let o = run(
        &url,
        home.path(),
        None,
        &[&PLAN[..], &["hf://fx/qwen3-8b-GGUF/nope.gguf"]].concat(),
    );
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("no file `nope.gguf`"), "{}", text(&o).1);
    let o = run(
        &url,
        home.path(),
        None,
        &[
            &PLAN[..],
            &["hf://fx/qwen3-8b-GGUF/Qwen3-8B-Q4_K_M.gguf", "--offline"],
        ]
        .concat(),
    );
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("--offline"), "{}", text(&o).1);
}

#[test]
fn a_bare_repo_compares_its_quants() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let o = run(
        &url,
        home.path(),
        None,
        &[&PLAN[..], &["hf://fx/qwen3-8b-GGUF", "--json"]].concat(),
    );
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}{err}");
    let v: serde_json::Value = serde_json::from_str(&out).unwrap();
    let q = v["quants"].as_array().unwrap();
    assert_eq!(q.len(), 2);
    assert_eq!(
        (q[0]["quant"].as_str(), q[0]["exact"].as_bool()),
        (Some("Q4_K_M"), Some(true))
    );
    assert_eq!(
        (q[1]["quant"].as_str(), q[1]["exact"].as_bool()),
        (Some("Q8_0"), Some(false))
    );
    // The Q8_0 file is 1.7x the size, so its estimate is about 1.7x the weights.
    assert!(q[1]["gpu_bytes"].as_u64() > q[0]["gpu_bytes"].as_u64());

    let o = run(
        &url,
        home.path(),
        None,
        &[&PLAN[..], &["hf://fx/qwen3-8b-GGUF"]].concat(),
    );
    let out = text(&o).0;
    let q8 = out.lines().find(|l| l.starts_with("Q8_0")).unwrap();
    assert!(q8.contains("~"), "an estimated row: {out}");
    assert_eq!(out.matches("← best fit").count(), 1, "{out}");
    assert!(out.contains("~ = planned from the Q4_K_M header"), "{out}");
    assert!(
        out.contains("loadmaster plan hf://fx/qwen3-8b-GGUF/"),
        "{out}"
    );
}

#[test]
fn plans_a_safetensors_repo_for_vllm() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let args = [
        "plan",
        "hf://fx/Qwen3-8B-AWQ",
        "--runtime",
        "vllm",
        "--pretend",
        "rtx 3060",
        "--ctx",
        "8192",
    ];
    let o = run(&url, home.path(), None, &args);
    let (out, err) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{out}{err}");
    assert!(out.contains("AWQ-4bit"), "{out}");
    assert!(out.contains("vllm serve fx/Qwen3-8B-AWQ "), "{out}");
    assert!(!out.contains("incomplete download"), "{out}");

    // A GGUF repo cannot be planned for vLLM.
    let o = run(
        &url,
        home.path(),
        None,
        &[
            "plan",
            "hf://fx/qwen3-8b-GGUF",
            "--runtime",
            "vllm",
            "--pretend",
            "4090",
        ],
    );
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("a GGUF repo"), "{}", text(&o).1);
}

const SUGGEST: [&str; 4] = ["suggest", "--pretend", "rtx 4090", "--ctx"];

fn suggest(url: &str, home: &Path, extra: &[&str]) -> (Option<i32>, String, String) {
    let args = [&SUGGEST[..], &["8192"], extra].concat();
    let o = run(url, home, None, &args);
    let (out, err) = text(&o);
    (o.status.code(), out, err)
}

/// The result rows of `suggest`'s table: the lines that name a repo.
fn table_rows(out: &str) -> Vec<&str> {
    out.lines()
        .filter(|l| l.split_whitespace().next().is_some_and(|w| w.contains('/')))
        .collect()
}

#[test]
fn suggest_ranks_what_fits_and_explains_what_it_skipped() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let (code, out, err) = suggest(&url, home.path(), &[]);
    assert_eq!(code, Some(0), "{out}{err}");
    let rows = table_rows(&out);
    assert_eq!(rows.len(), 3, "{out}");
    // Biggest first: gemma 12B, qwen3 8B, then the 8.0B audio model (the duplicate and the
    // 500 GB repo gone).
    assert!(
        rows[0].starts_with("google/gemma-3-12b-GGUF") && rows[0].contains("12.2B"),
        "{out}"
    );
    assert!(
        rows[1].starts_with("ggml-org/qwen3-8b-GGUF") && rows[1].contains("gpu"),
        "{out}"
    );
    assert!(rows[2].starts_with("ggml-org/ultravox-8b-GGUF"), "{out}");
    assert!(
        out.contains(
            "searched 7 repos · 1 too large · 1 duplicates · 1 by other publishers (--any-publisher) · 1 gated (set HF_TOKEN)"
        ),
        "{out}"
    );
    assert!(
        out.contains(
            "exact plan: loadmaster plan hf://google/gemma-3-12b-GGUF/gemma-3-12b-it-Q4_K_M.gguf"
        ),
        "{out}"
    );

    // Biggest first, and no further once enough fit on the GPU: the 8B is never planned.
    let (_, one, _) = suggest(&url, home.path(), &["--limit", "1"]);
    assert_eq!(table_rows(&one).len(), 1, "{one}");
    assert!(one.contains("2 smaller, not needed"), "{one}");

    // Anyone's repos on request: the fine-tune joins (a different base model, so no dup).
    let (_, any, _) = suggest(&url, home.path(), &["--any-publisher"]);
    assert!(any.contains("someone/qwen3-8b-finetune-GGUF"), "{any}");

    // The same answer offline, from the cached search, trees and headers.
    let (code, offline, err) = suggest("http://127.0.0.1:9", home.path(), &["--offline"]);
    assert_eq!(code, Some(0), "{offline}{err}");
    assert_eq!(offline, out);
}

#[test]
fn suggest_filters_by_kind_and_runtime() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let (_, out, _) = suggest(&url, home.path(), &["--kind", "vision"]);
    let rows = table_rows(&out);
    assert_eq!(rows.len(), 1, "{out}");
    assert!(rows[0].starts_with("google/gemma-3-12b-GGUF"), "{out}");

    // Speech input: the audio projector decides, and speech-to-text is named as out of scope.
    let (code, out, err) = suggest(&url, home.path(), &["--kind", "audio"]);
    assert_eq!(code, Some(0), "{out}{err}");
    let rows = table_rows(&out);
    assert_eq!(rows.len(), 1, "{out}");
    assert!(rows[0].starts_with("ggml-org/ultravox-8b-GGUF"), "{out}");
    assert!(out.contains("Whisper and Parakeet"), "{out}");
    let (code, _, err) = suggest(&url, home.path(), &["--kind", "audio", "--runtime", "vllm"]);
    assert_ne!(code, Some(0));
    assert!(
        err.contains("--kind audio searches llama.cpp (GGUF) models only"),
        "{err}"
    );

    let (code, out, err) = suggest(&url, home.path(), &["--runtime", "vllm"]);
    assert_eq!(code, Some(0), "{out}{err}");
    assert!(out.contains("Qwen/Qwen3-8B-AWQ"), "{out}");
    assert!(
        out.contains("exact plan: loadmaster plan hf://Qwen/Qwen3-8B-AWQ --runtime vllm"),
        "{out}"
    );

    let (_, json, _) = suggest(&url, home.path(), &["--json"]);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["results"][0]["repo"], "google/gemma-3-12b-GGUF");
    assert_eq!(v["results"][0]["kind"], "vision");
    assert_eq!(v["skipped"]["duplicates"], 1);
    assert_eq!(v["skipped"]["too_large"], 1);
}

#[test]
fn suggest_names_the_rate_limit() {
    let url = serve();
    let home = tempfile::tempdir().unwrap();
    let (code, out, err) = suggest(&url, home.path(), &["ratelimit"]);
    assert_eq!(code, Some(3), "{out}{err}");
    assert!(
        err.contains("rate limit") && err.contains("HF_TOKEN"),
        "{err}"
    );
}
