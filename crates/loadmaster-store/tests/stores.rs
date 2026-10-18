use std::os::unix::fs::symlink;
use std::path::{Path, PathBuf};

use loadmaster_model::Source;
use loadmaster_store::{Roots, resolve, scan};

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(rel)
}

fn copy(from: &str, to: &Path) {
    std::fs::create_dir_all(to.parent().unwrap()).unwrap();
    std::fs::copy(fixture(from), to).unwrap();
}

/// A fake machine: HF cache, Ollama, LM Studio and a loose directory.
fn machine() -> tempfile::TempDir {
    let t = tempfile::tempdir().unwrap();
    let r = t.path();

    // HF cache: a GGUF repo with two refs on one blob, plus a broken symlink.
    let repo = r.join("hf/models--unsloth--Qwen3-8B-GGUF");
    copy(
        "gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf",
        &repo.join("blobs/aaaa"),
    );
    for rev in ["rev1", "rev2"] {
        let snap = repo.join("snapshots").join(rev);
        std::fs::create_dir_all(&snap).unwrap();
        symlink("../../blobs/aaaa", snap.join("Qwen3-8B-Q4_K_M.gguf")).unwrap();
        symlink("../../blobs/missing", snap.join("Qwen3-8B-Q8_0.gguf")).unwrap();
    }
    std::fs::create_dir_all(repo.join("refs")).unwrap();
    std::fs::write(repo.join("refs/main"), "rev2").unwrap();
    std::fs::write(repo.join("refs/old"), "rev1\n").unwrap();
    // HF cache: a safetensors repo (plain files are fine too).
    let st = r.join("hf/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/abc");
    for f in std::fs::read_dir(fixture("hf/Meta-Llama-3.1-8B-Instruct")).unwrap() {
        let f = f.unwrap().path();
        copy(
            &format!(
                "hf/Meta-Llama-3.1-8B-Instruct/{}",
                f.file_name().unwrap().to_string_lossy()
            ),
            &st.join(f.file_name().unwrap()),
        );
    }

    // Ollama: gemma3 with a projector layer, and a manifest whose blob is missing.
    let o = r.join("ollama");
    copy(
        "gguf/gemma-3-12b/gemma-3-12b-it-Q4_K_M.gguf",
        &o.join("blobs/sha256-1111"),
    );
    copy(
        "gguf/gemma-3-12b/mmproj-gemma-3-12b-it-F16.gguf",
        &o.join("blobs/sha256-2222"),
    );
    let m = o.join("manifests/registry.ollama.ai/library/gemma3");
    std::fs::create_dir_all(&m).unwrap();
    std::fs::write(
        m.join("12b"),
        r#"{"schemaVersion":2,"layers":[
            {"mediaType":"application/vnd.ollama.image.model","digest":"sha256:1111","size":1},
            {"mediaType":"application/vnd.ollama.image.projector","digest":"sha256:2222","size":1},
            {"mediaType":"application/vnd.ollama.image.template","digest":"sha256:3333","size":1}]}"#,
    )
    .unwrap();
    let p = o.join("manifests/registry.ollama.ai/someone/partial");
    std::fs::create_dir_all(&p).unwrap();
    std::fs::write(
        p.join("latest"),
        r#"{"schemaVersion":2,"layers":[{"mediaType":"application/vnd.ollama.image.model","digest":"sha256:9999"}]}"#,
    )
    .unwrap();

    // LM Studio.
    copy(
        "gguf/qwen3-30b-a3b/Qwen3-30B-A3B-Q4_K_M.gguf",
        &r.join("lmstudio/lmstudio-community/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"),
    );

    // Loose directory, nested.
    copy(
        "gguf/deepseek-v2-lite/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
        &r.join("loose/deepseek/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf"),
    );
    t
}

fn roots(r: &Path) -> Roots {
    Roots {
        hf_hub: Some(r.join("hf")),
        ollama: Some(r.join("ollama")),
        lmstudio: vec![r.join("lmstudio"), r.join("does-not-exist")],
        extra: vec![r.join("loose")],
    }
}

#[test]
fn finds_every_store() {
    let t = machine();
    let s = scan(&roots(t.path()));
    let mut names: Vec<&str> = s.entries.iter().map(|e| e.name.as_str()).collect();
    names.sort();
    assert_eq!(
        names,
        [
            "deepseek/DeepSeek-V2-Lite-Chat-Q4_K_M.gguf",
            "gemma3:12b",
            "lmstudio-community/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf",
            "meta-llama/Llama-3.1-8B-Instruct",
            "unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        ],
        "one entry for the blob shared by two revisions"
    );
    assert!(
        s.entries.iter().all(|e| e.info.is_ok()),
        "every fixture parses"
    );
    assert!(
        s.entries
            .windows(2)
            .all(|w| w[0].size_bytes >= w[1].size_bytes),
        "sorted by size"
    );
}

#[test]
fn reports_broken_links_and_partial_pulls() {
    let t = machine();
    let s = scan(&roots(t.path()));
    let problems: Vec<String> = s
        .issues
        .iter()
        .map(|i| {
            format!(
                "{}: {}",
                i.path.file_name().unwrap().to_string_lossy(),
                i.problem
            )
        })
        .collect();
    assert!(
        problems
            .iter()
            .any(|p| p.starts_with("Qwen3-8B-Q8_0.gguf: broken symlink")),
        "{problems:?}"
    );
    assert!(
        problems
            .iter()
            .any(|p| p.contains("blob sha256:9999 is missing")),
        "{problems:?}"
    );
}

#[test]
fn ollama_entry_carries_projector_and_source() {
    let t = machine();
    let s = scan(&roots(t.path()));
    let e = resolve(&s.entries, "gemma3:12b").unwrap();
    let info = e.info.as_ref().unwrap();
    assert!(info.mmproj.is_some(), "projector layer attached");
    assert!(
        matches!(&info.source, Source::Ollama { name, digest } if name == "gemma3:12b" && digest == "sha256:1111")
    );
}

#[test]
fn hf_entry_records_repo_and_revision() {
    let t = machine();
    let s = scan(&roots(t.path()));
    let e = resolve(&s.entries, "hf://meta-llama/Llama-3.1-8B-Instruct").unwrap();
    assert!(
        matches!(&e.source, Source::HfCache { repo, revision } if repo == "meta-llama/Llama-3.1-8B-Instruct" && revision == "abc")
    );
}

#[test]
fn resolve_by_stem_and_reports_misses() {
    let t = machine();
    let s = scan(&roots(t.path()));
    assert_eq!(
        resolve(&s.entries, "qwen3-8b-q4_k_m").unwrap().name,
        "unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"
    );
    let err = resolve(&s.entries, "nope").unwrap_err();
    assert!(err.contains("loadmaster scan"), "{err}");
}

#[test]
fn scanning_writes_nothing() {
    fn tree(dir: &Path, out: &mut Vec<(PathBuf, u64)>) {
        for e in std::fs::read_dir(dir).unwrap() {
            let p = e.unwrap().path();
            let md = std::fs::symlink_metadata(&p).unwrap();
            out.push((p.clone(), md.len()));
            if md.is_dir() {
                tree(&p, out);
            }
        }
    }
    let t = machine();
    let mut before = Vec::new();
    tree(t.path(), &mut before);
    let _ = scan(&roots(t.path()));
    let mut after = Vec::new();
    tree(t.path(), &mut after);
    before.sort();
    after.sort();
    assert_eq!(before, after);
}

#[test]
fn missing_roots_are_empty_not_errors() {
    let s = scan(&Roots {
        hf_hub: Some("/nonexistent/hf".into()),
        ollama: Some("/nonexistent/ollama".into()),
        lmstudio: vec!["/nonexistent/lms".into()],
        extra: vec![],
    });
    assert!(s.entries.is_empty() && s.issues.is_empty());
}

#[test]
fn skips_encoder_models_with_a_note() {
    let t = machine();
    let bert = t
        .path()
        .join("hf/models--BAAI--bge-small-en-v1.5/snapshots/r1");
    std::fs::create_dir_all(&bert).unwrap();
    std::fs::write(
        bert.join("config.json"),
        r#"{"architectures":["BertModel"],"model_type":"bert"}"#,
    )
    .unwrap();
    std::fs::write(bert.join("model.safetensors"), b"").unwrap();
    let s = scan(&roots(t.path()));
    assert!(s.entries.iter().all(|e| !e.name.contains("bge")));
    assert!(s.issues.iter().any(|i| {
        i.problem
            .contains("BertModel is not a text-generation model")
    }));
}
