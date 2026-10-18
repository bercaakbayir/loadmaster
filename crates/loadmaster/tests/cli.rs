//! End-to-end runs of the binary against fixtures and replayed hardware snapshots.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use loadmaster_hw::FakeHw;

const QWEN3_8B: &str = "fixtures/gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf";
const QWEN3_30B: &str = "fixtures/gguf/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf";

fn workspace() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn hw_file(dir: &tempfile::TempDir, hw: FakeHw) -> PathBuf {
    let path = dir.path().join("hw.json");
    std::fs::write(&path, serde_json::to_string(&hw.0).unwrap()).unwrap();
    path
}

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_loadmaster"))
        .args(args)
        .current_dir(workspace())
        .env_remove("LOADMASTER_LOG")
        .output()
        .unwrap()
}

fn stdout(o: &Output) -> String {
    String::from_utf8(o.stdout.clone()).unwrap()
}

#[test]
fn plan_prints_summary_and_command() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::rtx_4090());
    let o = run(&[
        "plan",
        QWEN3_8B,
        "--ctx",
        "32768",
        "--hw-file",
        hw.to_str().unwrap(),
    ]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    insta::assert_snapshot!(stdout(&o));
}

#[test]
fn plan_json_is_one_object_on_stdout() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::rtx_4090());
    let o = run(&[
        "-v",
        "plan",
        QWEN3_8B,
        "--json",
        "--hw-file",
        hw.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["result"], "fits");
    assert_eq!(v["plan"]["argv"][0], "llama-server");
}

#[test]
fn cannot_fit_exits_1() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::m2_8g());
    let o = run(&[
        "plan",
        QWEN3_30B,
        "--ctx",
        "8192",
        "--hw-file",
        hw.to_str().unwrap(),
    ]);
    assert_eq!(o.status.code(), Some(1));
    assert!(stdout(&o).contains("does not fit"), "{}", stdout(&o));

    let o = run(&[
        "plan",
        QWEN3_30B,
        "--ctx",
        "8192",
        "--json",
        "--hw-file",
        hw.to_str().unwrap(),
    ]);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["result"], "cannot_fit");
}

#[test]
fn gpu_json_round_trips_as_hw_file() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::rtx_3060_12g());
    let o = run(&["gpu", "--json", "--hw-file", hw.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(0));
    let back: loadmaster_hw::HwSnapshot = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(back, FakeHw::rtx_3060_12g().0);
}

#[test]
fn errors_name_the_file_and_exit_2() {
    let o = run(&["plan", "Cargo.toml"]);
    assert_eq!(o.status.code(), Some(2));
    let err = String::from_utf8_lossy(&o.stderr);
    assert!(
        err.contains("Cargo.toml") && err.contains("not a GGUF file"),
        "{err}"
    );
}

#[test]
fn no_gpu_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::cpu_only());
    let o = run(&["plan", QWEN3_8B, "--hw-file", hw.to_str().unwrap()]);
    assert_eq!(o.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&o.stderr).contains("no GPU found"));
}
