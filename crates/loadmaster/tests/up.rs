#![cfg(unix)]
//! End-to-end supervisor: `loadmaster up` with fake runtimes (`loadmaster __fake-runtime`
//! behind wrapper scripts), driven over HTTP and through `status`/`swap`/`down`.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

use loadmaster_hw::{FakeHw, GIB};

const BIN: &str = env!("CARGO_BIN_EXE_loadmaster");

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// An executable wrapper so a lockfile's `runtime_bin` can be the fake runtime.
fn script(dir: &Path, name: &str, body: &str) -> PathBuf {
    let p = dir.join(name);
    std::fs::write(&p, format!("#!/bin/sh\n{body}\n")).unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

struct Env {
    dir: tempfile::TempDir,
    lock: PathBuf,
    hw: PathBuf,
    base: String,
}

impl Env {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let fake = script(
            dir.path(),
            "fake.sh",
            &format!("exec {BIN} __fake-runtime \"$@\""),
        );
        let slow = script(
            dir.path(),
            "slow.sh",
            &format!("exec {BIN} __fake-runtime --load-delay 30 \"$@\""),
        );
        let crash = script(
            dir.path(),
            "crash.sh",
            "echo 'fake runtime: out of memory' >&2; exit 1",
        );
        let model = workspace().join("fixtures/gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf");
        let port = free_port();
        // 21 GiB budget after 1 GiB headroom: two of the 10 GiB models fit, not three.
        let mut snap = FakeHw::rtx_4090().0;
        snap.gpus[0].usable_bytes = 22 * GIB;
        let hw = dir.path().join("hw.json");
        std::fs::write(&hw, serde_json::to_string(&snap).unwrap()).unwrap();
        let entry = |name: &str, bin: &Path, keep_warm: bool, aliases: &str| {
            format!(
                "[[model]]\nname = \"{name}\"\nsource = \"{}\"\nruntime = \"llama.cpp\"\nruntime_bin = \"{}\"\nctx = 4096\nfootprint_gb = 10.0\nflags = [\"--name\", \"{name}\"]\nkeep_warm = {keep_warm}\naliases = [{aliases}]\n\n",
                model.display(),
                bin.display()
            )
        };
        let mut text = format!(
            "version = 1\n\n[host]\ngpu = \"NVIDIA GeForce RTX 4090\"\nvram_gb = 24.0\nheadroom_gb = 1.0\n\n[proxy]\nlisten = \"127.0.0.1:{port}\"\nidle_unload_s = 0\nload_timeout_s = 2\n\n"
        );
        text += &entry("warm", &fake, true, "");
        text += &entry("alpha", &fake, false, "\"gpt-4o\"");
        text += &entry("beta", &fake, false, "");
        text += &entry("slow", &slow, false, "");
        text += &entry("crashy", &crash, false, "");
        let lock = dir.path().join("models.lock");
        std::fs::write(&lock, text).unwrap();
        Env {
            dir,
            lock,
            hw,
            base: format!("http://127.0.0.1:{port}"),
        }
    }

    fn cli(&self, args: &[&str]) -> std::process::Output {
        Command::new(BIN)
            .args(args)
            .args(["--lock", self.lock.to_str().unwrap()])
            .env("LOADMASTER_HOME", self.dir.path())
            .output()
            .unwrap()
    }

    fn up(&self) -> Guard {
        Guard(
            Command::new(BIN)
                .args([
                    "up",
                    "--lock",
                    self.lock.to_str().unwrap(),
                    "--hw-file",
                    self.hw.to_str().unwrap(),
                ])
                .env("LOADMASTER_HOME", self.dir.path())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .spawn()
                .unwrap(),
        )
    }
}

/// Kills `loadmaster up` (SIGTERM, so it stops its runtimes) if the test panics.
struct Guard(Child);

impl Drop for Guard {
    fn drop(&mut self) {
        if self.0.try_wait().ok().flatten().is_none() {
            let _ = Command::new("kill").arg(self.0.id().to_string()).status();
            let _ = self.0.wait();
        }
    }
}

fn client() -> reqwest::blocking::Client {
    reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .unwrap()
}

fn wait_for(url: &str) {
    let start = Instant::now();
    while start.elapsed() < Duration::from_secs(20) {
        if client()
            .get(url)
            .send()
            .is_ok_and(|r| r.status().is_success())
        {
            return;
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    panic!("{url} never came up");
}

fn chat(model: &str, stream: bool) -> serde_json::Value {
    serde_json::json!({ "model": model, "stream": stream, "messages": [{ "role": "user", "content": "hi" }] })
}

fn states(env: &Env) -> std::collections::HashMap<String, String> {
    let v: serde_json::Value = client()
        .get(format!("{}/loadmaster/status", env.base))
        .send()
        .unwrap()
        .json()
        .unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|s| {
            (
                s["name"].as_str().unwrap().to_owned(),
                s["state"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

#[test]
fn supervisor_end_to_end() {
    let env = Env::new();
    let mut up = env.up();
    wait_for(&format!("{}/health", env.base));
    let http = client();
    let post = |body: serde_json::Value| {
        http.post(format!("{}/v1/chat/completions", env.base))
            .json(&body)
            .send()
            .unwrap()
    };

    // keep_warm models are preloaded.
    let start = Instant::now();
    while states(&env)["warm"] != "ready" {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "warm never preloaded"
        );
        std::thread::sleep(Duration::from_millis(100));
    }

    // Route by alias; non-streamed response.
    let r = post(chat("gpt-4o", false));
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().unwrap();
    assert_eq!(v["model"], "alpha", "backend sees the lockfile name");
    assert_eq!(states(&env)["alpha"], "ready");

    // Streaming passes through byte for byte: compare with a direct request to an
    // identical fake runtime (both answer their first chat request).
    let direct_port = free_port();
    let mut direct = Command::new(BIN)
        .args([
            "__fake-runtime",
            "--port",
            &direct_port.to_string(),
            "--name",
            "beta",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    wait_for(&format!("http://127.0.0.1:{direct_port}/health"));
    let via_proxy = post(chat("beta", true)).bytes().unwrap();
    let straight = http
        .post(format!(
            "http://127.0.0.1:{direct_port}/v1/chat/completions"
        ))
        .json(&chat("beta", true))
        .send()
        .unwrap()
        .bytes()
        .unwrap();
    direct.kill().unwrap();
    direct.wait().unwrap();
    assert!(via_proxy.ends_with(b"data: [DONE]\n\n"));
    assert_eq!(via_proxy, straight, "SSE bytes identical through the proxy");

    // beta made room by evicting alpha (LRU), not warm (keep_warm), and exactly one model.
    let s = states(&env);
    assert_eq!(
        (s["warm"].as_str(), s["alpha"].as_str(), s["beta"].as_str()),
        ("ready", "stopped", "ready"),
        "{s:?}"
    );

    // Unknown model: 404 in OpenAI error shape.
    let r = post(chat("nope", false));
    assert_eq!(r.status(), 404);
    assert_eq!(
        r.json::<serde_json::Value>().unwrap()["error"]["code"],
        "model_not_found"
    );

    // A slow load: 503 with Retry-After once load_timeout_s passes.
    let r = post(chat("slow", false));
    assert_eq!(r.status(), 503);
    assert_eq!(r.headers()["retry-after"], "5");

    // Make room again: a model still loading is never evicted, so unload it explicitly.
    let r = http
        .post(format!("{}/loadmaster/unload/slow", env.base))
        .send()
        .unwrap();
    assert!(r.status().is_success());

    // A crash-looping runtime: while it restarts with backoff, requests see 503; after
    // three restarts it is failed and requests get 502 naming the log.
    assert_eq!(post(chat("crashy", false)).status(), 503);
    let start = Instant::now();
    while states(&env)["crashy"] != "failed" {
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "crashy never failed"
        );
        std::thread::sleep(Duration::from_millis(200));
    }
    let r = post(chat("crashy", false));
    assert_eq!(r.status(), 502);
    let msg = r.json::<serde_json::Value>().unwrap()["error"]["message"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(
        msg.contains("gave up after 3 restarts") && msg.contains("crashy.log"),
        "{msg}"
    );
    let log = std::fs::read_to_string(env.dir.path().join("logs/crashy.log")).unwrap();
    assert_eq!(
        log.lines().filter(|l| l.contains("out of memory")).count(),
        4
    );

    // /v1/models and /metrics.
    let v: serde_json::Value = http
        .get(format!("{}/v1/models", env.base))
        .send()
        .unwrap()
        .json()
        .unwrap();
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["warm", "alpha", "beta", "slow", "crashy"]);
    let metrics = http
        .get(format!("{}/metrics", env.base))
        .send()
        .unwrap()
        .text()
        .unwrap();
    assert!(
        metrics.contains("loadmaster_model_evictions_total{model=\"alpha\"} 1"),
        "{metrics}"
    );
    assert!(
        metrics.contains("loadmaster_model_loads_total{model=\"beta\"} 1"),
        "{metrics}"
    );

    // CLI: status shows the failure and log; swap loads alpha back; down stops everything.
    let out = env.cli(&["status"]);
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("crashy") && text.contains("failed") && text.contains("crashy.log"),
        "{text}"
    );
    let out = env.cli(&["swap", "alpha"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let s = states(&env);
    assert_eq!(
        (s["alpha"].as_str(), s["warm"].as_str()),
        ("ready", "ready"),
        "{s:?}"
    );
    assert!(env.cli(&["down"]).status.success());
    let start = Instant::now();
    loop {
        if let Some(status) = up.0.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(
            start.elapsed() < Duration::from_secs(30),
            "up did not exit after down"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}
