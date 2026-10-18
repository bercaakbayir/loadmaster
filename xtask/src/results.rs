//! `cargo xtask results <select | run | render>`: the published conformance table
//! (spec task 3.8).
//!
//! - `select` picks the models from the Hugging Face API by fixed rules and writes
//!   `results/matrix.toml` (commit it; it records when and how it was chosen).
//! - `run` (on the GPU machine) launches each matrix cell through a one-model lockfile,
//!   probes it, and writes `results/runs/<cell>.json`. Cells whose model is not downloaded
//!   are reported and skipped; LM Studio cells probe a server you started.
//! - `render` writes `RESULTS.md` from the run files.

use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const QUANTS: [&str; 2] = ["Q4_K_M", "Q8_0"];
const MAX_Q4_BYTES: u64 = 22 << 30;
const VARIANT_WORDS: [&str; 8] = [
    "uncensored",
    "abliterated",
    "heretic",
    "distill",
    "merge",
    "rp-",
    "nsfw",
    "aggressive",
];

#[derive(Debug, Serialize, Deserialize)]
pub struct Matrix {
    pub selected_on: String,
    pub rules: String,
    #[serde(rename = "model")]
    pub models: Vec<Entry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entry {
    /// Short name used in the table.
    pub name: String,
    pub gguf_repo: String,
    pub base_model: Option<String>,
    pub downloads: u64,
    /// Quant -> file path inside the repo.
    pub files: BTreeMap<String, String>,
}

fn curl_json(url: &str) -> anyhow::Result<Value> {
    let out = Command::new("curl")
        .args(["-sSfL", url])
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("{url}: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(serde_json::from_slice(&out.stdout)?)
}

pub fn select(root: &Path, today: &str) -> anyhow::Result<()> {
    let list = curl_json(
        "https://huggingface.co/api/models?filter=gguf&sort=downloads&direction=-1&limit=200&expand[]=downloads&expand[]=cardData&expand[]=pipeline_tag",
    )?;
    let mut seen_base = std::collections::HashSet::new();
    let mut models = Vec::new();
    for m in list.as_array().into_iter().flatten() {
        if models.len() == 20 {
            break;
        }
        let id = m["id"].as_str().unwrap_or_default().to_owned();
        let chat = |t: &str| ["text-generation", "image-text-to-text"].contains(&t);
        let lower = id.to_ascii_lowercase();
        if VARIANT_WORDS.iter().any(|w| lower.contains(w)) {
            continue;
        }
        let base = match &m["cardData"]["base_model"] {
            Value::String(s) => Some(s.clone()),
            Value::Array(a) => a.first().and_then(Value::as_str).map(str::to_owned),
            _ => None,
        };
        // An untagged repo counts only if its base model is a chat model.
        let tagged = match m["pipeline_tag"].as_str() {
            Some(t) => chat(t),
            None => base.as_ref().is_some_and(|b| {
                curl_json(&format!("https://huggingface.co/api/models/{b}"))
                    .ok()
                    .and_then(|v| v["pipeline_tag"].as_str().map(chat))
                    .unwrap_or(false)
            }),
        };
        if !tagged {
            continue;
        }
        let key = base
            .clone()
            .unwrap_or_else(|| id.clone())
            .to_ascii_lowercase();
        if seen_base.contains(&key) {
            continue;
        }
        let tree = match curl_json(&format!(
            "https://huggingface.co/api/models/{id}/tree/main?recursive=true"
        )) {
            Ok(t) => t,
            Err(_) => continue,
        };
        let mut files = BTreeMap::new();
        let mut q4_size = None;
        for f in tree.as_array().into_iter().flatten() {
            let path = f["path"].as_str().unwrap_or_default();
            let size = f["lfs"]["size"]
                .as_u64()
                .or(f["size"].as_u64())
                .unwrap_or(0);
            let name = path.rsplit('/').next().unwrap_or(path);
            if !name.ends_with(".gguf") || name.to_ascii_lowercase().contains("mmproj") {
                continue;
            }
            // Single-file quants only (split models are planned, but this table keeps runs simple).
            if name.contains("-of-0") {
                continue;
            }
            for q in QUANTS {
                if name.to_ascii_uppercase().contains(q) && !files.contains_key(q) {
                    files.insert(q.to_owned(), path.to_owned());
                    if q == "Q4_K_M" {
                        q4_size = Some(size);
                    }
                }
            }
        }
        if files.len() != QUANTS.len() || q4_size.is_none_or(|s| s > MAX_Q4_BYTES) {
            continue;
        }
        seen_base.insert(key);
        models.push(Entry {
            name: id
                .rsplit('/')
                .next()
                .unwrap_or(&id)
                .trim_end_matches("-GGUF")
                .trim_end_matches("-gguf")
                .to_owned(),
            gguf_repo: id,
            base_model: base,
            downloads: m["downloads"].as_u64().unwrap_or(0),
            files,
        });
    }
    let matrix = Matrix {
        selected_on: today.to_owned(),
        rules: "Hugging Face API, GGUF repos by downloads; text or image-text chat models (by pipeline tag, or the base model's when the repo has none); no abliterated/uncensored/merge variants; one repo per base model (most downloaded); single-file Q4_K_M and Q8_0 present; Q4_K_M at most 22 GiB.".into(),
        models,
    };
    let path = root.join("results/matrix.toml");
    std::fs::create_dir_all(path.parent().unwrap())?;
    std::fs::write(&path, toml::to_string_pretty(&matrix)?)?;
    println!("wrote {} ({} models)", path.display(), matrix.models.len());
    Ok(())
}

/// One probe result, as written by `run`.
#[derive(Debug, Serialize, Deserialize)]
pub struct Cell {
    pub model: String,
    pub runtime: String,
    pub quant: String,
    /// `pass` | `partial` | `fail` | `not run: <why>`.
    pub verdict: String,
    #[serde(default)]
    pub failures: Vec<String>,
    #[serde(default)]
    pub runtime_version: Option<String>,
}

fn cell_id(model: &str, runtime: &str, quant: &str) -> String {
    format!("{model}--{runtime}--{quant}").replace('/', "_")
}

pub fn run(root: &Path, lmstudio_url: Option<&str>) -> anyhow::Result<()> {
    let matrix: Matrix = toml::from_str(
        &std::fs::read_to_string(root.join("results/matrix.toml"))
            .context("run `cargo xtask results select` first")?,
    )?;
    let out_dir = root.join("results/runs");
    std::fs::create_dir_all(&out_dir)?;
    let bin = root.join("target/release/loadmaster");
    if !bin.exists() {
        bail!("build first: cargo build --release -p loadmaster");
    }
    let work = std::env::temp_dir().join("loadmaster-results");
    std::fs::create_dir_all(&work)?;
    for m in &matrix.models {
        for (quant, file) in &m.files {
            for runtime in ["llama.cpp", "vllm", "lmstudio"] {
                let id = cell_id(&m.name, runtime, quant);
                let started = Instant::now();
                let cell = match runtime {
                    "lmstudio" => match lmstudio_url {
                        Some(url) => probe(&bin, url, &format!("{}/{}", m.gguf_repo, file))
                            .map(|r| (r, None)),
                        None => Err(anyhow::anyhow!(
                            "not run: pass --lmstudio URL with the model loaded"
                        )),
                    },
                    // vLLM serves HF repos; GGUF in vLLM is experimental, so the vLLM column
                    // probes the base model (its own dtype) once, under the Q4_K_M row.
                    "vllm" if quant != "Q4_K_M" => continue,
                    "vllm" => match &m.base_model {
                        Some(base) => serve_and_probe(&bin, &work, &format!("hf://{base}"), "vllm"),
                        None => Err(anyhow::anyhow!("not run: no base model recorded")),
                    },
                    _ => serve_and_probe(
                        &bin,
                        &work,
                        &format!("hf://{}/{file}", m.gguf_repo),
                        "llama.cpp",
                    ),
                };
                let cell = match cell {
                    Ok((report, version)) => Cell {
                        model: m.name.clone(),
                        runtime: runtime.into(),
                        quant: quant.clone(),
                        verdict: report["verdict"].as_str().unwrap_or("?").to_owned(),
                        failures: report["failures"]
                            .as_array()
                            .into_iter()
                            .flatten()
                            .filter_map(|f| f.as_str().map(str::to_owned))
                            .collect(),
                        runtime_version: version
                            .or_else(|| report["detected"]["version"].as_str().map(str::to_owned)),
                    },
                    Err(e) => Cell {
                        model: m.name.clone(),
                        runtime: runtime.into(),
                        quant: quant.clone(),
                        verdict: if e.to_string().starts_with("not run") {
                            e.to_string()
                        } else {
                            format!("not run: {e}")
                        },
                        failures: vec![],
                        runtime_version: None,
                    },
                };
                println!(
                    "{id}: {} ({:.0}s)",
                    cell.verdict,
                    started.elapsed().as_secs_f64()
                );
                std::fs::write(
                    out_dir.join(format!("{id}.json")),
                    serde_json::to_vec_pretty(&cell)?,
                )?;
            }
        }
    }
    render(root)
}

fn probe(bin: &Path, url: &str, model: &str) -> anyhow::Result<Value> {
    let out = Command::new(bin)
        .args(["probe", url, "--model", model, "--json"])
        .output()?;
    serde_json::from_slice(&out.stdout)
        .with_context(|| format!("probe output: {}", String::from_utf8_lossy(&out.stderr)))
}

/// One-model lockfile, `up`, probe through the proxy, `down`.
fn serve_and_probe(
    bin: &Path,
    work: &Path,
    source: &str,
    runtime: &str,
) -> anyhow::Result<(Value, Option<String>)> {
    let lock = work.join("models.lock");
    let name = source.trim_start_matches("hf://");
    let init = Command::new(bin)
        .args([
            "init",
            name,
            "--ctx",
            "8192",
            "--force",
            "--lock",
            lock.to_str().unwrap(),
            "--no-hash",
        ])
        .output()?;
    if !init.status.success() {
        bail!(
            "not run: {}",
            String::from_utf8_lossy(&init.stderr)
                .lines()
                .last()
                .unwrap_or("init failed")
        );
    }
    let lockfile = std::fs::read_to_string(&lock)?;
    if !lockfile.contains(&format!("runtime = \"{runtime}\"")) {
        bail!("not run: planned for another runtime");
    }
    let listen = lockfile
        .lines()
        .find_map(|l| l.strip_prefix("listen = "))
        .map(|s| s.trim_matches('"').to_owned())
        .unwrap_or_else(|| "127.0.0.1:8080".into());
    let served = lockfile
        .lines()
        .find_map(|l| l.strip_prefix("name = "))
        .map(|s| s.trim_matches('"').to_owned())
        .context("no model in lockfile")?;
    let mut up = Command::new(bin)
        .args(["up", "--verify", "--lock", lock.to_str().unwrap()])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    // `up --verify` loads the model and writes its verdict into the lockfile.
    let deadline = Instant::now() + Duration::from_secs(1800);
    let result = loop {
        std::thread::sleep(Duration::from_secs(2));
        let text = std::fs::read_to_string(&lock).unwrap_or_default();
        if text.contains("[model.verify]") {
            break probe(bin, &format!("http://{listen}"), &served).map(|r| {
                let version = text
                    .lines()
                    .find_map(|l| l.strip_prefix("runtime_version = "))
                    .map(|v| v.trim_matches('"').to_owned());
                (r, version)
            });
        }
        if let Ok(Some(_)) = up.try_wait() {
            break Err(anyhow::anyhow!(
                "not run: `up` exited before the model was ready"
            ));
        }
        if Instant::now() > deadline {
            break Err(anyhow::anyhow!("not run: model not ready after 30 minutes"));
        }
    };
    let _ = Command::new(bin)
        .args(["down", "--lock", lock.to_str().unwrap()])
        .status();
    let _ = up.wait();
    result
}

pub fn render(root: &Path) -> anyhow::Result<()> {
    let matrix: Option<Matrix> = std::fs::read_to_string(root.join("results/matrix.toml"))
        .ok()
        .and_then(|t| toml::from_str(&t).ok());
    let mut cells: Vec<Cell> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(root.join("results/runs")) {
        for e in rd.flatten() {
            if let Ok(c) = serde_json::from_slice(&std::fs::read(e.path())?) {
                cells.push(c);
            }
        }
    }
    let md = render_markdown(matrix.as_ref(), &cells);
    std::fs::write(root.join("RESULTS.md"), md)?;
    println!("wrote RESULTS.md ({} cells)", cells.len());
    Ok(())
}

pub fn render_markdown(matrix: Option<&Matrix>, cells: &[Cell]) -> String {
    let mut o = String::from("# Tool-calling and structured-output conformance\n\n");
    o += "Generated by `cargo xtask results run` (then `render`). Each cell is `loadmaster probe`'s verdict for the model on that runtime and quant: **pass** (all fixtures), **partial** (tool.single, tool.result_roundtrip and schema.strict pass), **fail**. Failure classes are listed; see `docs/fixtures.md`.\n\n";
    match matrix {
        Some(m) => writeln!(o, "Models selected on {}: {}\n", m.selected_on, m.rules).unwrap(),
        None => o += "No `results/matrix.toml` yet; run `cargo xtask results select`.\n\n",
    }
    if cells.is_empty() {
        o += "**No results yet.** The table is produced on a 24 GB NVIDIA GPU with `cargo build --release && cargo xtask results run`; nothing here is invented or estimated.\n";
        return o;
    }
    let fmt = |c: Option<&Cell>| match c {
        None => "-".to_owned(),
        Some(c) if c.verdict.starts_with("not run") => "not run".to_owned(),
        Some(c) if c.failures.is_empty() => c.verdict.clone(),
        Some(c) => format!("{} ({})", c.verdict, c.failures.join(", ")),
    };
    o += "| model | quant | llama.cpp | vLLM (base model) | LM Studio |\n|---|---|---|---|---|\n";
    let mut keys: Vec<(String, String)> = cells
        .iter()
        .map(|c| (c.model.clone(), c.quant.clone()))
        .collect();
    keys.sort();
    keys.dedup();
    for (model, quant) in keys {
        let get = |rt: &str| {
            cells
                .iter()
                .find(|c| c.model == model && c.quant == quant && c.runtime == rt)
        };
        writeln!(
            o,
            "| {model} | {quant} | {} | {} | {} |",
            fmt(get("llama.cpp")),
            fmt(get("vllm")),
            fmt(get("lmstudio"))
        )
        .unwrap();
    }
    let versions: std::collections::BTreeSet<String> = cells
        .iter()
        .filter_map(|c| {
            c.runtime_version
                .as_ref()
                .map(|v| format!("{}: {v}", c.runtime))
        })
        .collect();
    if !versions.is_empty() {
        o += "\nRuntime versions: ";
        o += &versions.into_iter().collect::<Vec<_>>().join("; ");
        o += "\n";
    }
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_cells_and_says_so_when_empty() {
        assert!(render_markdown(None, &[]).contains("No results yet"));
        let cells = vec![
            Cell {
                model: "M".into(),
                runtime: "llama.cpp".into(),
                quant: "Q4_K_M".into(),
                verdict: "partial".into(),
                failures: vec!["parallel_serialised".into()],
                runtime_version: Some("b6700".into()),
            },
            Cell {
                model: "M".into(),
                runtime: "vllm".into(),
                quant: "Q4_K_M".into(),
                verdict: "not run: x".into(),
                failures: vec![],
                runtime_version: None,
            },
        ];
        let md = render_markdown(None, &cells);
        assert!(
            md.contains("| M | Q4_K_M | partial (parallel_serialised) | not run | - |"),
            "{md}"
        );
        assert!(md.contains("llama.cpp: b6700"));
    }
}
