#![cfg(unix)]
//! `loadmaster probe`, `probe eval`, `probe compare` and `up --verify` against fake runtimes.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::time::{Duration, Instant};

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

struct Proc(Child);

impl Drop for Proc {
    fn drop(&mut self) {
        let _ = Command::new("kill").arg(self.0.id().to_string()).status();
        let _ = self.0.wait();
    }
}

fn wait_for(url: &str) {
    let start = Instant::now();
    while !reqwest::blocking::get(url).is_ok_and(|r| r.status().is_success()) {
        assert!(
            start.elapsed() < Duration::from_secs(20),
            "{url} never came up"
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn fake(misbehave: &str) -> (Proc, String) {
    let port = free_port();
    let p = Command::new(BIN)
        .args([
            "__fake-runtime",
            "--port",
            &port.to_string(),
            "--misbehave",
            misbehave,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let url = format!("http://127.0.0.1:{port}");
    wait_for(&format!("{url}/health"));
    (Proc(p), url)
}

fn run(home: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .env("LOADMASTER_HOME", home)
        .output()
        .unwrap()
}

#[test]
fn probe_exit_codes_follow_the_verdict() {
    let home = tempfile::tempdir().unwrap();
    for (misbehave, code, verdict) in [
        ("", 0, "pass"),
        ("parallel_serialised", 1, "partial"),
        ("call_in_content", 2, "fail"),
    ] {
        let (_p, url) = fake(misbehave);
        let o = run(
            home.path(),
            &["probe", &url, "--model", "fake-model", "--repeat", "1"],
        );
        let out = String::from_utf8_lossy(&o.stdout);
        assert_eq!(
            o.status.code(),
            Some(code),
            "{misbehave}: {out}{}",
            String::from_utf8_lossy(&o.stderr)
        );
        assert!(out.contains(&format!("verdict: {verdict}")), "{out}");
        assert!(out.contains("llama.cpp b6700"), "runtime detected: {out}");
    }
    let (_p, url) = fake("call_in_content");
    let o = run(
        home.path(),
        &[
            "probe",
            &url,
            "--model",
            "m",
            "--json",
            "--only",
            "tool.single",
            "--repeat",
            "1",
        ],
    );
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 1);
    assert_eq!(v["results"][0]["outcome"]["detail"], "call_in_content");
    // Raw captures are written for debugging.
    let probe_dir = home.path().join("probe");
    let run_dir = std::fs::read_dir(&probe_dir)
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert!(run_dir.join("tool.single/run0-step0.response.raw").exists());
}

#[test]
fn eval_is_reproducible_and_compare_marks_noise() {
    let home = tempfile::tempdir().unwrap();
    let (_p, url) = fake("");
    let prompts = home.path().join("prompts.jsonl");
    std::fs::write(
        &prompts,
        concat!(
            r#"{"id": "weather", "prompt": "Weather in Paris?", "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}}], "expect_tool": "get_weather"}"#, "\n",
            r#"{"id": "person", "prompt": "Maria, 34", "schema": {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}}"#, "\n",
            r#"{"id": "greet", "prompt": "hi", "expect_contains": "Hello"}"#, "\n",
            r#"{"id": "wrong", "prompt": "hi", "expect_contains": "Bonjour"}"#, "\n",
        ),
    )
    .unwrap();
    let p = prompts.to_str().unwrap();
    for _ in 0..2 {
        let o = run(
            home.path(),
            &["probe", "eval", p, "--model", "fake-model", "--url", &url],
        );
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        assert!(String::from_utf8_lossy(&o.stdout).contains("accuracy 3/4  schema 1/1"));
    }
    let o = run(home.path(), &["probe", "compare", "1", "2"]);
    let out = String::from_utf8_lossy(&o.stdout).into_owned();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
    assert!(out.lines().count() <= 20, "{out}");
    assert!(
        out.contains("75.0%") && out.contains("+0.0pp") && out.contains("n.s."),
        "{out}"
    );
    assert!(out.contains("0 of 4 responses differ"), "{out}");
}

#[test]
fn up_verify_writes_verdicts_and_status_flags_stale_ones() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("fake.sh");
    std::fs::write(
        &script,
        format!("#!/bin/sh\nexec {BIN} __fake-runtime \"$@\"\n"),
    )
    .unwrap();
    std::fs::set_permissions(&script, std::fs::Permissions::from_mode(0o755)).unwrap();
    let model = workspace().join("fixtures/gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf");
    let port = free_port();
    let lock = dir.path().join("models.lock");
    std::fs::write(
        &lock,
        format!(
            "version = 1\n\n[host]\ngpu = \"NVIDIA GeForce RTX 4090\"\nvram_gb = 24.0\nheadroom_gb = 1.0\n\n[proxy]\nlisten = \"127.0.0.1:{port}\"\nidle_unload_s = 0\nload_timeout_s = 30\n\n[[model]]\nname = \"qwen\"\nsource = \"{}\"\nruntime = \"llama.cpp\"\nruntime_bin = \"{}\"\nctx = 4096\nfootprint_gb = 6.0\nflags = []\nkeep_warm = false\naliases = []\n",
            model.display(),
            script.display()
        ),
    )
    .unwrap();
    let hw = dir.path().join("hw.json");
    std::fs::write(
        &hw,
        serde_json::to_string(&loadmaster_hw::FakeHw::rtx_4090().0).unwrap(),
    )
    .unwrap();
    let up = Proc(
        Command::new(BIN)
            .args([
                "up",
                "--verify",
                "--lock",
                lock.to_str().unwrap(),
                "--hw-file",
                hw.to_str().unwrap(),
            ])
            .env("LOADMASTER_HOME", dir.path())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap(),
    );
    let start = Instant::now();
    let text = loop {
        let t = std::fs::read_to_string(&lock).unwrap();
        if t.contains("[model.verify]") {
            break t;
        }
        assert!(
            start.elapsed() < Duration::from_secs(60),
            "no verdict written"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(text.contains("verdict = \"pass\""), "{text}");
    assert!(
        text.contains("runtime_version = \"llama.cpp b6700-abc1234\""),
        "{text}"
    );
    loadmaster_lock::Lockfile::load(&lock).expect("still a valid lockfile");

    let o = run(dir.path(), &["status", "--lock", lock.to_str().unwrap()]);
    assert!(
        String::from_utf8_lossy(&o.stdout).contains("pass"),
        "{}",
        String::from_utf8_lossy(&o.stdout)
    );
    // Pretend the verdict came from another runtime build: status must call it stale.
    std::fs::write(
        &lock,
        text.replace("llama.cpp b6700-abc1234", "llama.cpp b6000-old"),
    )
    .unwrap();
    let o = run(dir.path(), &["status", "--lock", lock.to_str().unwrap()]);
    let out = String::from_utf8_lossy(&o.stdout);
    assert!(
        out.contains("stale: verified on llama.cpp b6000-old"),
        "{out}"
    );
    drop(up);
}

#[test]
fn probe_checks_embedding_endpoints() {
    let (_ok, url) = fake("");
    let o = Command::new(BIN)
        .args(["probe", &url, "--model", "emb", "--embeddings", "--json"])
        .output()
        .unwrap();
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["verdict"], "pass");
    let ids: Vec<&str> = v["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["embed.shape", "embed.deterministic", "embed.batch"]);

    let (_bad, url) = fake("embed_nondeterministic");
    let o = Command::new(BIN)
        .args(["probe", &url, "--model", "emb", "--embeddings", "--json"])
        .output()
        .unwrap();
    assert_eq!(o.status.code(), Some(2));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["failures"][0], "embed_nondeterministic");
}
