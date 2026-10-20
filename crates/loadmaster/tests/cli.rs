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
    run_in(args, &workspace().join("target/no-such-home"))
}

/// Run with `home` as `$HOME` and no store overrides, so real stores are never scanned.
fn run_in(args: &[&str], home: &Path) -> Output {
    Command::new(env!("CARGO_BIN_EXE_loadmaster"))
        .args(args)
        .current_dir(workspace())
        .env_remove("LOADMASTER_LOG")
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("OLLAMA_MODELS")
        .env_remove("NO_COLOR")
        .env_remove("CLICOLOR")
        .env_remove("CLICOLOR_FORCE")
        .env("HOME", home)
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
fn piped_output_is_plain_and_colour_is_opt_in() {
    let dir = tempfile::tempdir().unwrap();
    let hw = hw_file(&dir, FakeHw::rtx_4090());
    let hw = hw.to_str().unwrap();
    let plan = ["plan", QWEN3_8B, "--hw-file", hw];
    // The test harness is never a terminal: no styles, and the command on one line.
    let out = stdout(&run(&plan));
    assert!(!out.contains('\x1b'), "{out}");
    let cmd = out.lines().find(|l| l.contains("llama-server")).unwrap();
    assert!(cmd.trim_start().starts_with("llama-server") && !cmd.ends_with('\\'));

    let always = stdout(&run(&[&plan[..], &["--color", "always"]].concat()));
    assert!(always.contains("\x1b[1;32m✓ Fits"), "{always}");
    let never = Command::new(env!("CARGO_BIN_EXE_loadmaster"))
        .args(plan)
        .current_dir(workspace())
        .env("NO_COLOR", "1")
        .env("HOME", workspace().join("target/no-such-home"))
        .output()
        .unwrap();
    assert!(!stdout(&never).contains('\x1b'));

    // --command: exactly the launch line, for scripts.
    let only = stdout(&run(&[&plan[..], &["--command"]].concat()));
    assert_eq!(only.lines().count(), 1, "{only}");
    assert!(only.starts_with("llama-server -m "), "{only}");
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
    assert!(stdout(&o).contains("✗ Doesn't fit"), "{}", stdout(&o));

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
fn errors_name_the_file_and_exit_3() {
    let o = run(&["plan", "Cargo.toml"]);
    assert_eq!(o.status.code(), Some(3));
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
    assert_eq!(o.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&o.stderr).contains("no GPU found"));
}

#[test]
fn scan_lists_stores_and_plan_resolves_names() {
    let home = tempfile::tempdir().unwrap();
    let blobs = home.path().join(".ollama/models/blobs");
    let manifests = home
        .path()
        .join(".ollama/models/manifests/registry.ollama.ai/library/qwen3");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::copy(workspace().join(QWEN3_8B), blobs.join("sha256-abcd")).unwrap();
    std::fs::write(
        manifests.join("8b"),
        r#"{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:abcd"}]}"#,
    )
    .unwrap();
    let hw = hw_file(&home, FakeHw::rtx_4090());
    let hw = hw.to_str().unwrap();

    let o = run_in(&["scan", "--hw-file", hw], home.path());
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let out = stdout(&o);
    assert!(
        out.contains("qwen3:8b") && out.contains("ollama") && out.contains("Q4_K_M"),
        "{out}"
    );

    let o = run_in(&["scan", "--json", "--hw-file", hw], home.path());
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["models"][0]["name"], "qwen3:8b");
    assert_eq!(v["models"][0]["fits"][0], "gpu");
    assert_eq!(v["models"][0]["caveats"][0], serde_json::Value::Null);

    let o = run_in(
        &["plan", "qwen3:8b", "--ctx", "8192", "--hw-file", hw],
        home.path(),
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(stdout(&o).contains("sha256-abcd"));
}

#[test]
fn scan_marks_fits_under_the_ram_margin() {
    let home = ollama_home();
    let mut hw = FakeHw::m3_max_64g();
    // RAM Qwen3 8B needs at 32k on a 64 GB Mac with `available` bytes free.
    let need = |hw: &FakeHw| {
        let path = hw_file(&home, FakeHw(hw.0.clone()));
        let o = run(&[
            "plan",
            QWEN3_8B,
            "--ctx",
            "32768",
            "--json",
            "--hw-file",
            path.to_str().unwrap(),
        ]);
        let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
        v["plan"]["host_need"].as_u64().unwrap()
    };
    // f16 KV, then q8_0 once f16 no longer leaves the margin; then just above q8_0's need,
    // so only a tight q8_0 fit is left (q4_0 is never picked to keep the margin).
    hw.0.host.available_bytes = need(&hw);
    let q8 = need(&hw);
    hw.0.host.available_bytes = q8 + q8 / 20;
    let path = hw_file(&home, hw);
    let hw = path.to_str().unwrap();

    let o = run_in(&["scan", "--json", "--hw-file", hw], home.path());
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["models"][0]["fits"][1], "gpu");
    assert_eq!(v["models"][0]["caveats"][1], "tight_ram");
    let out = stdout(&run_in(&["scan", "--hw-file", hw], home.path()));
    assert!(
        out.contains("gpu*") && out.contains("free-RAM margin"),
        "{out}"
    );
}

/// A fake home with one Ollama model (the Qwen3 8B fixture).
fn ollama_home() -> tempfile::TempDir {
    let home = tempfile::tempdir().unwrap();
    let blobs = home.path().join(".ollama/models/blobs");
    let manifests = home
        .path()
        .join(".ollama/models/manifests/registry.ollama.ai/library/qwen3");
    std::fs::create_dir_all(&blobs).unwrap();
    std::fs::create_dir_all(&manifests).unwrap();
    std::fs::copy(
        workspace().join(QWEN3_8B),
        blobs.join(format!("sha256-{}", "ab".repeat(32))),
    )
    .unwrap();
    std::fs::write(
        manifests.join("8b"),
        format!(r#"{{"schemaVersion":2,"layers":[{{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:{}"}}]}}"#, "ab".repeat(32)),
    )
    .unwrap();
    home
}

#[test]
fn init_add_update_and_up_accepts_it() {
    let home = ollama_home();
    let hw = hw_file(&home, FakeHw::rtx_4090());
    let hw = hw.to_str().unwrap();
    let lock = home.path().join("models.lock");
    let lock_s = lock.to_str().unwrap();
    let port = std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port();
    let listen = format!("127.0.0.1:{port}");

    let o = run_in(
        &[
            "init",
            "qwen3:8b",
            "--ctx",
            "8192",
            "--listen",
            &listen,
            "--lock",
            lock_s,
            "--hw-file",
            hw,
        ],
        home.path(),
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let text = std::fs::read_to_string(&lock).unwrap();
    assert!(text.contains("name = \"qwen3:8b\""), "{text}");
    assert!(text.contains("source = \"ollama://qwen3:8b\""), "{text}");
    assert!(
        text.contains(&format!("sha256 = \"{}\"", "ab".repeat(32))),
        "digest reused, not re-hashed"
    );
    assert!(text.contains("ctx = 8192"));

    // init refuses to overwrite; add appends under a new name with an alias.
    let o = run_in(
        &["init", "qwen3:8b", "--lock", lock_s, "--hw-file", hw],
        home.path(),
    );
    assert_ne!(o.status.code(), Some(0));
    let o = run_in(
        &[
            "add",
            "qwen3:8b",
            "--as",
            "qwen-long",
            "--ctx",
            "32768",
            "--alias",
            "gpt-4o",
            "--keep-warm",
            "--lock",
            lock_s,
            "--hw-file",
            hw,
        ],
        home.path(),
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let text = std::fs::read_to_string(&lock).unwrap();
    assert!(
        text.contains("name = \"qwen-long\"")
            && text.contains("aliases = [\"gpt-4o\"]")
            && text.contains("keep_warm = true")
    );

    // plan --update re-plans one entry and prints what changed.
    let o = run_in(
        &[
            "plan",
            "qwen3:8b",
            "--update",
            "--ctx",
            "16384",
            "--lock",
            lock_s,
            "--hw-file",
            hw,
        ],
        home.path(),
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        stdout(&o).contains("~ qwen3:8b.ctx: 8192 -> 16384"),
        "{}",
        stdout(&o)
    );
    let o = run_in(
        &[
            "plan",
            "qwen3:8b",
            "--update",
            "--ctx",
            "16384",
            "--lock",
            lock_s,
            "--hw-file",
            hw,
        ],
        home.path(),
    );
    assert!(stdout(&o).contains("up to date"));

    // up accepts it unedited (models load on demand, so no runtime is started here).
    let mut up = Command::new(env!("CARGO_BIN_EXE_loadmaster"))
        .args(["up", "--lock", lock_s, "--hw-file", hw])
        .env("HOME", home.path())
        .env("LOADMASTER_HOME", home.path())
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("OLLAMA_MODELS")
        .stderr(std::process::Stdio::null())
        .spawn()
        .unwrap();
    let url = format!("http://{listen}/v1/models");
    let start = std::time::Instant::now();
    let models = loop {
        if let Ok(r) = reqwest::blocking::get(&url) {
            break r.json::<serde_json::Value>().unwrap();
        }
        if let Some(s) = up.try_wait().unwrap() {
            panic!("up exited: {s}");
        }
        assert!(start.elapsed() < std::time::Duration::from_secs(20));
        std::thread::sleep(std::time::Duration::from_millis(100));
    };
    assert_eq!(models["data"].as_array().unwrap().len(), 2);
    let o = run_in(&["down", "--lock", lock_s], home.path());
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(up.wait().unwrap().success());
}

#[test]
fn import_llama_swap_writes_a_valid_lockfile() {
    let home = tempfile::tempdir().unwrap();
    let hw = hw_file(&home, FakeHw::rtx_4090());
    let lock = home.path().join("models.lock");
    let config = workspace().join("crates/loadmaster-lock/tests/fixtures/llama-swap-homelab.yaml");
    let o = run_in(
        &[
            "import",
            "llama-swap",
            config.to_str().unwrap(),
            "--lock",
            lock.to_str().unwrap(),
            "--hw-file",
            hw.to_str().unwrap(),
        ],
        home.path(),
    );
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    let out = stdout(&o);
    assert!(
        out.contains("imported qwen3-coder") && out.contains("footprint unknown"),
        "{out}"
    );
    assert!(
        out.contains("not planned:"),
        "model files are not on this machine: {out}"
    );
    let text = std::fs::read_to_string(&lock).unwrap();
    assert!(
        text.contains("idle_unload_s = 900"),
        "globalTTL carried over"
    );
    assert!(text.contains("aliases = [\"coder\", \"gpt-4o\"]"));
}

#[test]
fn pretend_plans_on_hardware_you_do_not_have() {
    let o = run(&["plan", QWEN3_30B, "--ctx", "32768", "--pretend", "rtx 4090"]);
    assert_eq!(
        o.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&o.stderr)
    );
    assert!(
        stdout(&o).contains("✓ Fits on RTX 4090 (pretend)"),
        "{}",
        stdout(&o)
    );

    let o = run(&["gpu", "--pretend", "m4 pro 48gb", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&o.stdout).unwrap();
    assert_eq!(v["gpus"][0]["memory"], "unified");
    assert_eq!(v["host"]["total_bytes"], 48u64 << 30);

    let o = run(&["gpu", "--pretend", "list"]);
    assert!(stdout(&o).contains("NVIDIA GeForce RTX 5090"));

    let o = run(&["plan", QWEN3_8B, "--pretend", "voodoo 5"]);
    assert_eq!(o.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&o.stderr).contains("unknown hardware"));

    let o = run(&["up", "--pretend", "4090"]);
    assert_eq!(o.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&o.stderr).contains("--pretend is for planning"));
}

#[test]
fn gpu_selection_splits_by_default_and_pins_one() {
    let split = run(&[
        "plan",
        QWEN3_30B,
        "--ctx",
        "32768",
        "--pretend",
        "2x rtx 3060",
    ]);
    let out = stdout(&split);
    assert_eq!(split.status.code(), Some(0), "{out}");
    assert!(
        out.contains("Fits on 2× RTX 3060") && out.contains("--tensor-split"),
        "{out}"
    );
    let one = run(&[
        "plan",
        QWEN3_30B,
        "--ctx",
        "32768",
        "--pretend",
        "2x rtx 3060",
        "--gpu",
        "1",
    ]);
    let out = stdout(&one);
    assert!(
        out.contains("--device CUDA1") && !out.contains("--tensor-split"),
        "{out}"
    );
    let both = run(&[
        "plan",
        QWEN3_30B,
        "--ctx",
        "32768",
        "--pretend",
        "2x rtx 3060",
        "--gpu",
        "0,1",
    ]);
    assert!(stdout(&both).contains("--device CUDA0,CUDA1"));
}
