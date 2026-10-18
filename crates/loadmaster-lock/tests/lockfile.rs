use std::path::Path;

use loadmaster_lock::import::llama_swap;
use loadmaster_lock::{Change, Host, Lockfile, Model, Verify, diff};
use proptest::prelude::*;

fn sample() -> Lockfile {
    let mut l = Lockfile::new(Host {
        gpu: "NVIDIA GeForce RTX 4090".into(),
        vram_gb: 24.0,
        headroom_gb: 0.78,
    });
    l.models.push(Model {
        name: "qwen3-coder".into(),
        source: "hf://unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf".into(),
        sha256: Some("ab".repeat(32)),
        runtime: "llama.cpp".into(),
        runtime_bin: "llama-server".into(),
        ctx: 65536,
        footprint_gb: Some(21.4),
        flags: ["-ngl", "49", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--jinja"].map(String::from).to_vec(),
        mmproj: None,
        env: vec![],
        keep_warm: true,
        aliases: vec!["gpt-4o".into(), "coder".into()],
        idle_unload_s: None,
        verify: Some(Verify {
            probe_version: "0.3.0".into(),
            verdict: "partial".into(),
            failures: vec!["parallel_in_content".into()],
            runtime_version: Some("llama.cpp b6700".into()),
            checked_at: "2026-12-01T10:00:00Z".into(),
        }),
    });
    l
}

fn parse(s: &str) -> Result<Lockfile, loadmaster_lock::Error> {
    Lockfile::parse(s, Path::new("models.lock"))
}

#[test]
fn canonical_form_is_stable() {
    let text = sample().to_toml();
    insta::assert_snapshot!(text);
    let back = parse(&text).unwrap();
    assert_eq!(back, sample());
    assert_eq!(back.to_toml(), text, "byte-identical round trip");
}

#[test]
fn the_spec_example_parses() {
    let spec = r#"
version = 1

[host]
gpu = "NVIDIA GeForce RTX 4090"
vram_gb = 24.0
headroom_gb = 0.8

[proxy]
listen = "127.0.0.1:8080"
idle_unload_s = 600

[[model]]
name = "qwen3-coder"
source = "hf://unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
sha256 = "..."
runtime = "llama.cpp"
runtime_bin = "llama-server"
ctx = 65536
footprint_gb = 21.4
flags = ["-ngl", "99", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "--jinja"]
keep_warm = true
aliases = ["gpt-4o", "coder"]

[model.verify]
probe_version = "0.3.0"
verdict = "partial"
failures = ["parallel_calls_in_content"]
checked_at = "2026-12-01T10:00:00Z"
"#;
    let l = parse(spec).unwrap();
    assert_eq!(
        l.proxy.load_timeout_s,
        loadmaster_lock::DEFAULT_LOAD_TIMEOUT_S
    );
    assert_eq!(l.route("coder").unwrap().name, "qwen3-coder");
}

#[test]
fn unknown_keys_are_errors() {
    let text = sample()
        .to_toml()
        .replace("keep_warm = true", "keep_warm = true\nkeepwarm = true");
    let err = parse(&text).unwrap_err().to_string();
    assert!(err.contains("keepwarm"), "{err}");
}

#[test]
fn versions_are_checked() {
    let newer = sample().to_toml().replace("version = 1", "version = 2");
    assert!(
        parse(&newer)
            .unwrap_err()
            .to_string()
            .contains("upgrade loadmaster")
    );
}

#[test]
fn validation_reports_every_problem() {
    let mut l = sample();
    let mut dup = l.models[0].clone();
    dup.aliases = vec!["qwen3-coder".into()];
    dup.runtime = "ollama".into();
    dup.flags.push("--port".into());
    l.models.push(dup);
    l.proxy.listen = "localhost".into();
    let problems = l.problems();
    for want in [
        "duplicate name",
        "runtime `ollama`",
        "must not contain `--port`",
        "collides",
        "proxy.listen",
    ] {
        assert!(
            problems.iter().any(|p| p.contains(want)),
            "{want}: {problems:?}"
        );
    }
}

#[test]
fn diff_names_fields() {
    let a = sample();
    let mut b = sample();
    b.models[0].ctx = 32768;
    b.models.push(Model {
        name: "embed".into(),
        aliases: vec![],
        ..a.models[0].clone()
    });
    let d = diff(&a, &b);
    assert!(d.contains(&Change::Field {
        model: "qwen3-coder".into(),
        field: "ctx",
        from: "65536".into(),
        to: "32768".into()
    }));
    assert!(d.contains(&Change::Added("embed".into())));
}

fn word() -> impl Strategy<Value = String> {
    "[a-zA-Z0-9_./:\"\\\\ -]{1,12}"
}

proptest! {
    /// Any valid lockfile we write parses back to itself and re-serialises identically.
    #[test]
    fn round_trip(names in prop::collection::btree_set("[a-z][a-z0-9-]{0,8}", 1..4), flags in prop::collection::vec(word(), 0..6), ctx in 1u32..1_000_000, fp in 0.01f64..200.0, keep in any::<bool>()) {
        let mut l = sample();
        l.models = names.iter().map(|n| Model {
            name: n.clone(),
            flags: flags.iter().filter(|f| !["--port", "--host", "-m", "--model"].contains(&f.as_str())).cloned().collect(),
            ctx,
            footprint_gb: Some((fp * 100.0).round() / 100.0),
            keep_warm: keep,
            aliases: vec![],
            ..sample().models[0].clone()
        }).collect();
        let text = l.to_toml();
        let back = parse(&text).unwrap();
        prop_assert_eq!(&back, &l);
        prop_assert_eq!(back.to_toml(), text);
    }
}

fn import(name: &str) -> llama_swap::Imported {
    let text = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name),
    )
    .unwrap();
    llama_swap::parse(&text, |k| (k == "HOME").then(|| "/home/me".to_owned())).unwrap()
}

/// The imported entries as a valid lockfile (footprints left for planning).
fn as_lockfile(i: &llama_swap::Imported) -> Lockfile {
    let mut l = sample();
    l.models = i.models.iter().map(|m| m.model.clone()).collect();
    assert!(l.problems().is_empty(), "{:?}", l.problems());
    parse(&l.to_toml()).unwrap()
}

#[test]
fn imports_readme_minimal() {
    let i = import("llama-swap-readme.yaml");
    assert_eq!(i.models.len(), 1);
    let m = &i.models[0].model;
    assert_eq!(
        (m.name.as_str(), m.source.as_str(), m.runtime.as_str()),
        ("model1", "/path/to/model.gguf", "llama.cpp")
    );
    assert!(m.flags.is_empty() && !i.models[0].ctx_given);
    as_lockfile(&i);
}

#[test]
fn imports_example_config() {
    let i = import("llama-swap-example.yaml");
    let names: Vec<&str> = i.models.iter().map(|m| m.model.name.as_str()).collect();
    assert_eq!(names, ["gpt-oss-120b"]);
    let m = &i.models[0].model;
    assert_eq!(m.runtime_bin, "/path/to/llama-server/llama-server-ec9e0301");
    assert_eq!(m.ctx, 16384, "per-model macro overrides the global one");
    assert_eq!(m.flags, ["--temperature", "0.7"]);
    assert_eq!(m.env, ["CUDA_VISIBLE_DEVICES=0,1,2"]);
    assert_eq!(m.aliases, ["gpt-4o-mini"]);
    assert_eq!(m.idle_unload_s, Some(60));
    let skipped: Vec<&str> = i.skipped.iter().map(|s| s.id.as_str()).collect();
    assert_eq!(skipped, ["unlisted-model", "docker-llama"]);
    assert!(
        i.warnings
            .iter()
            .any(|w| w.contains("gpt-oss-120b: `filters`"))
    );
    assert!(i.warnings.iter().any(|w| w.contains("`apiKeys`")));
    as_lockfile(&i);
}

#[test]
fn imports_homelab_config() {
    let i = import("llama-swap-homelab.yaml");
    let get = |n: &str| &i.models.iter().find(|m| m.model.name == n).unwrap().model;
    let coder = get("qwen3-coder");
    assert_eq!(
        coder.source,
        "/srv/models/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
    );
    assert_eq!(coder.ctx, 65536);
    assert_eq!(coder.idle_unload_s, Some(0));
    assert_eq!(
        coder.flags,
        [
            "-fa",
            "on",
            "--jinja",
            "-np",
            "1",
            "-ngl",
            "99",
            "--n-cpu-moe",
            "12",
            "-ctk",
            "q8_0",
            "-ctv",
            "q8_0",
            "--temp",
            "0.7",
            "--top-p",
            "0.8"
        ]
    );
    let vision = get("gemma3-vision");
    assert_eq!(
        vision.mmproj.as_deref(),
        Some("/srv/models/mmproj-gemma-3-12b-it-F16.gguf")
    );
    assert_eq!(vision.ctx, 32768, "--ctx-size=32768 form");
    let vllm = get("qwen3-awq");
    assert_eq!(
        (vllm.runtime.as_str(), vllm.source.as_str(), vllm.ctx),
        ("vllm", "Qwen/Qwen3-8B-AWQ", 32768)
    );
    assert_eq!(vllm.flags, ["--gpu-memory-utilization", "0.85"]);
    assert_eq!(i.global_ttl_s, Some(900));
    assert!(i.skipped.is_empty());
    assert!(i.warnings.iter().any(|w| w.contains("`groups`")));
    as_lockfile(&i);
}

#[test]
fn entry_from_plan_strips_machine_specific_args() {
    use loadmaster_hw::FakeHw;
    use loadmaster_plan::{PlanRequest, plan};
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/gguf/gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf");
    let info = loadmaster_model::ModelInfo::open(&root).unwrap();
    let hw = FakeHw::rtx_3060_12g();
    let p = plan(
        &info,
        &hw.0,
        &hw.0.gpus[0],
        &PlanRequest {
            ctx: Some(16384),
            ..Default::default()
        },
    )
    .unwrap();
    let m = Model::from_plan("gemma3", "gemma.gguf".into(), None, &p);
    assert_eq!(m.runtime_bin, "llama-server");
    assert_eq!(m.ctx, 16384);
    assert!(
        m.mmproj
            .as_deref()
            .unwrap()
            .ends_with("mmproj-gemma-3-12b-it-F16.gguf")
    );
    for banned in ["-m", "-c", "--mmproj", "--port"] {
        assert!(
            !m.flags.iter().any(|f| f == banned),
            "{banned} in {:?}",
            m.flags
        );
    }
    assert!(m.flags.contains(&"--jinja".to_owned()));
    assert!(m.footprint_gb.unwrap() > 1.0);
}
