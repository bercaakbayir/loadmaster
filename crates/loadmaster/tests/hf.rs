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

/// Repos keyed by `org/name`, then by path in the repo. `private` repos need the token.
struct Hub {
    repos: HashMap<String, Vec<(String, File)>>,
    private: Vec<String>,
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
        "fx/private".into(),
        vec![(
            "Qwen3-8B-Q4_K_M.gguf".into(),
            gguf("qwen3-8b/Qwen3-8B-Q4_K_M.gguf"),
        )],
    );
    Hub {
        repos,
        private: vec!["fx/private".into()],
    }
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
    assert!(out.contains("Q8_0") && out.contains("  *"), "{out}");
    assert!(
        out.contains("loadmaster plan hf://fx/qwen3-8b-GGUF/"),
        "{out}"
    );
}
