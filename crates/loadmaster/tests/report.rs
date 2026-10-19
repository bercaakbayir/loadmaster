#![cfg(unix)]
//! `loadmaster report` end to end against the fake runtime replaying llama-server log
//! fixtures: no model is ever loaded.

use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use loadmaster_hw::{FakeHw, GIB};

const BIN: &str = env!("CARGO_BIN_EXE_loadmaster");
const QWEN3_8B: &str = "fixtures/gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// A `llama-server` stand-in: the fake runtime printing a log fixture.
fn runtime(dir: &Path, fixture: &str, extra: &str) -> PathBuf {
    let p = dir.join(format!("{fixture}.sh"));
    let log = workspace().join("fixtures/llama-log").join(fixture);
    std::fs::write(
        &p,
        format!(
            "#!/bin/sh\nexec {BIN} __fake-runtime --log-fixture {} --speed 1830,41.2 {extra} \"$@\"\n",
            log.display()
        ),
    )
    .unwrap();
    std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    p
}

fn hw(dir: &Path, hw: FakeHw) -> PathBuf {
    let p = dir.join("hw.json");
    std::fs::write(&p, serde_json::to_string(&hw.0).unwrap()).unwrap();
    p
}

fn run(home: &Path, args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .current_dir(workspace())
        .env("HOME", home)
        .env("LOADMASTER_HOME", home.join("state"))
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("OLLAMA_MODELS")
        .stdin(Stdio::null())
        .output()
        .unwrap()
}

fn text(o: &Output) -> (String, String) {
    (
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn decode(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap());
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

fn the_report(dir: &Path) -> (String, toml::Value) {
    let f = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .find(|p| p.extension().is_some_and(|x| x == "toml"))
        .expect("a report file");
    let t = std::fs::read_to_string(f).unwrap();
    let v = toml::from_str(&t).unwrap();
    (t, v)
}

#[test]
fn a_run_that_fits_becomes_a_shareable_report() {
    let home = tempfile::tempdir().unwrap();
    let out = home.path().join("out");
    std::fs::create_dir_all(&out).unwrap();
    let rt = runtime(home.path(), "cuda-fits.log", "");
    let hw = hw(home.path(), FakeHw::rtx_4090());
    let o = run(
        home.path(),
        &[
            "report",
            QWEN3_8B,
            "--ctx",
            "8192",
            "--runtime-bin",
            rt.to_str().unwrap(),
            "--yes",
            "--hw-file",
            hw.to_str().unwrap(),
            "--out",
            out.to_str().unwrap(),
        ],
    );
    let (stdout, stderr) = text(&o);
    assert_eq!(o.status.code(), Some(0), "{stdout}{stderr}");
    let (file, v) = the_report(&out);
    assert_eq!(v["prediction"]["result"].as_str(), Some("fits"));
    assert_eq!(v["observed"]["result"].as_str(), Some("fits"));
    assert_eq!(v["observed"]["speed"]["tg_tok_s"].as_float(), Some(41.2));
    let gpu = &v["observed"]["buffers"][0];
    assert_eq!(gpu["device"].as_str(), Some("CUDA0"));
    assert_eq!(gpu["model"].as_float(), Some(4455.35));
    assert_eq!(v["model"]["sha256"].as_str().map(str::len), Some(64));
    assert_eq!(v["report"]["mode"].as_str(), Some("run"));
    // Privacy: no directory of this machine appears anywhere.
    for p in [
        home.path().display().to_string(),
        workspace().canonicalize().unwrap().display().to_string(),
        "fixtures/gguf".into(),
    ] {
        assert!(!file.contains(&p), "report names {p}:\n{file}");
    }
    assert!(file.contains("-m <model>"), "{file}");
    // The prefilled issue carries exactly the file.
    let url = stdout
        .lines()
        .find(|l| l.contains("/issues/new?"))
        .unwrap()
        .trim();
    assert!(url.contains("template=validation.yml"), "{url}");
    let report = url.split("&report=").nth(1).unwrap();
    assert_eq!(decode(report), file);
}

#[test]
fn a_failed_load_is_recorded_as_oom() {
    let home = tempfile::tempdir().unwrap();
    let rt = runtime(home.path(), "cuda-oom.log", "--oom-after 0.1");
    let hw = hw(home.path(), FakeHw::rtx_4090());
    let o = run(
        home.path(),
        &[
            "report",
            QWEN3_8B,
            "--ctx",
            "8192",
            "--runtime-bin",
            rt.to_str().unwrap(),
            "--yes",
            "--hw-file",
            hw.to_str().unwrap(),
            "--out",
            home.path().to_str().unwrap(),
        ],
    );
    assert_eq!(o.status.code(), Some(0), "{:?}", text(&o));
    let (file, v) = the_report(home.path());
    assert_eq!(v["observed"]["result"].as_str(), Some("oom"), "{file}");
    assert!(
        v["observed"]["reason"]
            .as_str()
            .unwrap()
            .contains("cudaMalloc failed")
    );
    assert!(v["observed"].get("speed").is_none());
}

#[test]
fn unified_memory_runs_only_plans_that_fit_safely() {
    let home = tempfile::tempdir().unwrap();
    let rt = runtime(home.path(), "metal-vision.log", "");
    // Qwen3 8B does not fit on an 8 GB Mac: refused, nothing started.
    let hw8 = hw(home.path(), FakeHw::m2_8g());
    let args = |hw: &Path| {
        vec![
            "report".to_owned(),
            QWEN3_8B.into(),
            "--ctx".into(),
            "8192".into(),
            "--runtime-bin".into(),
            rt.display().to_string(),
            "--yes".into(),
            "--hw-file".into(),
            hw.display().to_string(),
        ]
    };
    let a = args(&hw8);
    let o = run(
        home.path(),
        &a.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("unified memory"), "{:?}", text(&o));
    assert!(
        !home.path().join("state/report").exists(),
        "nothing started"
    );

    // Fits, but under the free-RAM margin: refused too.
    let mut tight = FakeHw::m3_max_64g();
    let o = run(
        home.path(),
        &[
            "plan",
            QWEN3_8B,
            "--ctx",
            "8192",
            "--json",
            "--hw-file",
            hw(home.path(), FakeHw(tight.0.clone())).to_str().unwrap(),
        ],
    );
    let p: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    let need = p["plan"]["host_need"].as_u64().unwrap();
    tight.0.host.available_bytes = need + need / 20;
    tight.0.host.total_bytes = 64 * GIB;
    let a = args(&hw(home.path(), tight));
    let o = run(
        home.path(),
        &a.iter().map(String::as_str).collect::<Vec<_>>(),
    );
    assert_eq!(o.status.code(), Some(3), "{:?}", text(&o));
    assert!(text(&o).1.contains("--unsafe-unified"), "{:?}", text(&o));
}

#[test]
fn asks_before_starting_and_refuses_pretend() {
    let home = tempfile::tempdir().unwrap();
    let hw = hw(home.path(), FakeHw::rtx_4090());
    let o = run(
        home.path(),
        &["report", QWEN3_8B, "--hw-file", hw.to_str().unwrap()],
    );
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("--yes"), "{:?}", text(&o));
    let o = run(home.path(), &["report", QWEN3_8B, "--pretend", "4090"]);
    assert_eq!(o.status.code(), Some(3));
    assert!(text(&o).1.contains("real hardware"));
}

#[test]
fn reads_the_last_load_from_an_up_log() {
    let home = tempfile::tempdir().unwrap();
    let model = workspace().join(QWEN3_8B);
    let lock = home.path().join("models.lock");
    std::fs::write(
        &lock,
        format!(
            "version = 1\n\n[host]\ngpu = \"NVIDIA GeForce RTX 4090\"\nvram_gb = 24.0\nheadroom_gb = 1.0\n\n[proxy]\nlisten = \"127.0.0.1:18000\"\nidle_unload_s = 0\nload_timeout_s = 2\n\n[[model]]\nname = \"qwen\"\nsource = \"{}\"\nruntime = \"llama.cpp\"\nruntime_bin = \"llama-server\"\nctx = 8192\nfootprint_gb = 10.0\nflags = [\"-lv\", \"4\"]\nkeep_warm = false\naliases = []\n",
            model.display()
        ),
    )
    .unwrap();
    let logs = home.path().join("state/logs");
    std::fs::create_dir_all(&logs).unwrap();
    let fits =
        std::fs::read_to_string(workspace().join("fixtures/llama-log/cuda-fits.log")).unwrap();
    let oom = std::fs::read_to_string(workspace().join("fixtures/llama-log/cuda-oom.log")).unwrap();
    // An earlier failed start, then the latest successful one.
    std::fs::write(
        logs.join("qwen.log"),
        format!(
            "--- loadmaster: starting llama-server -m {m} -c 8192\n{oom}--- loadmaster: starting llama-server -m {m} -c 8192 -lv 4\n{fits}",
            m = model.display()
        ),
    )
    .unwrap();
    let hw = hw(home.path(), FakeHw::rtx_4090());
    let o = run(
        home.path(),
        &[
            "report",
            "--from-log",
            "qwen",
            "--lock",
            lock.to_str().unwrap(),
            "--hw-file",
            hw.to_str().unwrap(),
            "--out",
            home.path().to_str().unwrap(),
        ],
    );
    assert_eq!(o.status.code(), Some(0), "{:?}", text(&o));
    let (file, v) = the_report(home.path());
    assert_eq!(v["report"]["mode"].as_str(), Some("log"));
    assert_eq!(v["observed"]["result"].as_str(), Some("fits"), "{file}");
    assert!(!file.contains(&model.display().to_string()), "{file}");
}
