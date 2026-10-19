//! `cargo xtask intake <report.toml | issue-body.md> [--check] [--add-speed]`: turn a
//! `loadmaster report` (a file, or a GitHub issue body that contains one) into a validation
//! case. Writes the report to `validation/reports/<id>.toml` and appends a `[[case]]` to
//! `validation/cases.toml`; with `--add-speed`, also a `[[point]]` to `validation/speed.toml`.
//! `--check` validates without writing anything.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::{Context, bail};
use toml::Value;

/// The report inside an issue body: a ```toml block, or everything from the first
/// `[report]` (or the comment line before it) when pasted bare.
fn extract(text: &str) -> &str {
    if let Some(start) = text.find("```toml") {
        let rest = &text[start + 7..];
        return rest.split("```").next().unwrap_or(rest);
    }
    let start = text
        .find("# loadmaster report")
        .or_else(|| text.find("[report]"))
        .unwrap_or(0);
    &text[start..]
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> anyhow::Result<&'a str> {
    let mut cur = v;
    for k in path {
        cur = cur
            .get(k)
            .with_context(|| format!("missing `{}`", path.join(".")))?;
    }
    cur.as_str()
        .with_context(|| format!("`{}` is not a string", path.join(".")))
}

fn int_at(v: &Value, path: &[&str]) -> anyhow::Result<i64> {
    let mut cur = v;
    for k in path {
        cur = cur
            .get(k)
            .with_context(|| format!("missing `{}`", path.join(".")))?;
    }
    cur.as_integer()
        .with_context(|| format!("`{}` is not an integer", path.join(".")))
}

fn float_at(v: &Value, path: &[&str]) -> Option<f64> {
    let mut cur = v;
    for k in path {
        cur = cur.get(k)?;
    }
    cur.as_float()
        .or_else(|| cur.as_integer().map(|i| i as f64))
}

fn slug(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || c == '.' || c == '_' {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') {
            out.push('-');
        }
    }
    out.trim_matches('-').to_owned()
}

/// Where the validation run can fetch the model's header from.
fn header_url(source: Option<&str>, sha256: &str) -> Option<String> {
    let s = source?;
    if let Some(rest) = s.strip_prefix("hf://") {
        let (org, rest) = rest.split_once('/')?;
        let (repo, file) = rest.split_once('/')?;
        return Some(format!(
            "https://huggingface.co/{org}/{repo}/resolve/main/{file}"
        ));
    }
    if let Some(name) = s.strip_prefix("ollama://") {
        let name = name.split(':').next()?;
        let path = if name.contains('/') {
            name.to_owned()
        } else {
            format!("library/{name}")
        };
        return Some(format!(
            "https://registry.ollama.ai/v2/{path}/blobs/sha256:{sha256}"
        ));
    }
    None
}

#[derive(Debug)]
pub struct Intake {
    pub id: String,
    pub case: String,
    pub speed_point: Option<String>,
    pub report: String,
}

/// Validate a report and build its case; no file is touched.
pub fn build(text: &str) -> anyhow::Result<Intake> {
    let body = extract(text).trim();
    let v: Value = toml::from_str(body).context("the report is not valid TOML")?;
    let gpu = str_at(&v, &["hardware", "gpu"])?;
    if gpu.contains("(pretend)") {
        bail!("planned for pretend hardware, not a measurement");
    }
    let observed = str_at(&v, &["observed", "result"])?;
    let truth = match observed {
        "fits" => "fits",
        "oom" => "oom",
        other => bail!(
            "observed `{other}` is not ground truth (only `fits` and `oom` are){}",
            v.get("observed")
                .and_then(|o| o.get("reason"))
                .and_then(Value::as_str)
                .map_or(String::new(), |r| format!(": {r}"))
        ),
    };
    let Ok(sha256) = str_at(&v, &["model", "sha256"]) else {
        bail!("no model sha256: the case cannot name the exact file");
    };
    if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
        bail!("model.sha256 is not a sha256");
    }
    let name = str_at(&v, &["model", "name"])?;
    let quant = str_at(&v, &["model", "quant"])?;
    let ctx = int_at(&v, &["prediction", "ctx"])?;
    let kv = v
        .get("prediction")
        .and_then(|p| p.get("kv"))
        .and_then(Value::as_str)
        .unwrap_or("f16");
    let vendor = str_at(&v, &["hardware", "vendor"])?;
    let memory = str_at(&v, &["hardware", "memory"])?;
    if !["nvidia", "apple"].contains(&vendor) || !["dedicated", "unified"].contains(&memory) {
        bail!("unknown hardware kind {vendor}/{memory}");
    }
    let total = int_at(&v, &["hardware", "total_mib"])?;
    let usable = int_at(&v, &["hardware", "usable_mib"])?;
    let host_available = int_at(&v, &["hardware", "host_available_mib"])?;
    let date = str_at(&v, &["report", "date"])?;
    let source = v
        .get("model")
        .and_then(|m| m.get("source"))
        .and_then(Value::as_str);
    let short_gpu = gpu
        .trim_start_matches("NVIDIA ")
        .trim_start_matches("GeForce ")
        .trim_start_matches("Apple ");
    let id = slug(&format!(
        "{name}-{quant}-{short_gpu}-{}k-{kv}",
        (ctx + 1023) / 1024
    ));

    let mut case = String::new();
    writeln!(case, "\n[[case]]").unwrap();
    writeln!(case, "id = \"{id}\"").unwrap();
    writeln!(case, "model = \"{name} {quant}\"").unwrap();
    writeln!(case, "sha256 = \"{sha256}\"").unwrap();
    match header_url(source, sha256) {
        Some(u) => writeln!(case, "header_url = \"{u}\"").unwrap(),
        None => writeln!(
            case,
            "# header_url: add one so `cargo xtask fetch-headers` can fetch this model's header"
        )
        .unwrap(),
    }
    writeln!(case, "runtime = \"llama.cpp\"").unwrap();
    writeln!(case, "ctx = {ctx}").unwrap();
    writeln!(case, "kv = \"{kv}\"").unwrap();
    writeln!(
        case,
        "hw = {{ name = \"{gpu}\", vendor = \"{vendor}\", memory = \"{memory}\", total_mib = {total}, usable_mib = {usable}, host_available_mib = {host_available} }}"
    )
    .unwrap();
    writeln!(case, "truth = \"{truth}\"").unwrap();
    writeln!(
        case,
        "# from a loadmaster report, {date}: validation/reports/{id}.toml"
    )
    .unwrap();

    let speed_point = match (
        float_at(&v, &["observed", "speed", "tg_tok_s"]),
        float_at(&v, &["hardware", "bandwidth_gbs"]),
    ) {
        (Some(tg), Some(bw)) => {
            let weight_mib = float_at(&v, &["model", "weight_mib"]).context("model.weight_mib")?;
            let n_layer = int_at(&v, &["model", "n_layer"])?;
            let backend = if vendor == "apple" { "metal" } else { "cuda" };
            Some(format!(
                "\n[[point]]\ndevice = \"{gpu}\"\nbackend = \"{backend}\"\nbandwidth_gbs = {bw}\nn_layer = {n_layer}\nquant = \"{quant}\"\nmodel_gib = {:.2}\ntg_tok_s = {tg}\n# {name}, from a loadmaster report, {date}\n",
                weight_mib / 1024.0
            ))
        }
        _ => None,
    };
    Ok(Intake {
        id,
        case,
        speed_point,
        report: format!("{body}\n"),
    })
}

pub fn run(root: &Path, input: &Path, check: bool, add_speed: bool) -> anyhow::Result<()> {
    let text =
        std::fs::read_to_string(input).with_context(|| format!("reading {}", input.display()))?;
    let i = build(&text)?;
    let cases_path = root.join("validation/cases.toml");
    let cases = std::fs::read_to_string(&cases_path)?;
    if cases.contains(&format!("id = \"{}\"", i.id)) {
        bail!("case `{}` is already in validation/cases.toml", i.id);
    }
    println!("case `{}`:{}", i.id, i.case);
    if let Some(p) = &i.speed_point {
        println!("speed point:{p}");
    }
    if check {
        println!("valid (--check: nothing written)");
        return Ok(());
    }
    let reports = root.join("validation/reports");
    std::fs::create_dir_all(&reports)?;
    std::fs::write(reports.join(format!("{}.toml", i.id)), &i.report)?;
    std::fs::write(&cases_path, format!("{cases}{}", i.case))?;
    if add_speed && let Some(p) = &i.speed_point {
        let speed = root.join("validation/speed.toml");
        let s = std::fs::read_to_string(&speed)?;
        std::fs::write(&speed, format!("{s}{p}"))?;
        println!("added to validation/speed.toml; `cargo test -p loadmaster-plan` checks the band");
    }
    println!(
        "wrote validation/reports/{}.toml and the case; next: cargo xtask fetch-headers && cargo xtask validate",
        i.id
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const REPORT: &str = r#"# loadmaster report: review before sharing.
[report]
loadmaster = "0.3.0"
date = "2026-10-19"
os = "linux x86_64"
runtime = "llama.cpp"
command = "llama-server -m <model> -c 32768"
mode = "run"

[model]
name = "Qwen3 8B"
arch = "qwen3"
quant = "Q4_K_M"
n_layer = 36
weight_mib = 4790.5
sha256 = "074dde013c612ae9231da7d9030f64489f4e6325ac3de74e16d539eab8e71573"
source = "hf://Qwen/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf"

[hardware]
gpu = "NVIDIA GeForce RTX 3090"
vendor = "nvidia"
memory = "dedicated"
total_mib = 24576
usable_mib = 24076
bandwidth_gbs = 936.0
host_total_mib = 65536
host_available_mib = 49152

[prediction]
result = "fits"
ctx = 32768
kv = "f16"
gpu_mib = 10342.1

[observed]
result = "fits"
buffers = []

[observed.speed]
tg_tok_s = 98.4
pp_tok_s = 3210.0
prompt_tokens = 512
gen_tokens = 128
"#;

    #[test]
    fn a_report_becomes_a_case_and_a_speed_point() {
        let i = build(REPORT).unwrap();
        assert_eq!(i.id, "qwen3-8b-q4_k_m-rtx-3090-32k-f16");
        assert!(i.case.contains("truth = \"fits\""));
        assert!(i.case.contains(
            "header_url = \"https://huggingface.co/Qwen/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf\""
        ));
        // The case parses with the rest of cases.toml.
        let v: Value = toml::from_str(&i.case).unwrap();
        assert_eq!(v["case"][0]["hw"]["usable_mib"].as_integer(), Some(24076));
        let p: Value = toml::from_str(i.speed_point.as_deref().unwrap()).unwrap();
        assert_eq!(p["point"][0]["tg_tok_s"].as_float(), Some(98.4));
        assert_eq!(p["point"][0]["backend"].as_str(), Some("cuda"));
    }

    #[test]
    fn issue_bodies_and_rejections() {
        let body = format!("### Report\n\n```toml\n{REPORT}```\n\n### Notes\n\nfine");
        assert_eq!(build(&body).unwrap().id, "qwen3-8b-q4_k_m-rtx-3090-32k-f16");
        let pasted = format!("### Report\n\n{REPORT}");
        assert!(build(&pasted).is_ok());
        let timeout = REPORT.replace(
            "result = \"fits\"\nbuffers",
            "result = \"timeout\"\nbuffers",
        );
        assert!(
            build(&timeout)
                .unwrap_err()
                .to_string()
                .contains("ground truth")
        );
        let pretend = REPORT.replace("RTX 3090\"", "RTX 3090 (pretend)\"");
        assert!(build(&pretend).is_err());
        let no_sha = REPORT.replace("sha256 = \"074d", "# sha256 = \"074d");
        assert!(build(&no_sha).is_err());
    }

    #[test]
    fn ollama_sources_point_at_the_registry() {
        assert_eq!(
            header_url(Some("ollama://llama3.2:1b"), "abc").as_deref(),
            Some("https://registry.ollama.ai/v2/library/llama3.2/blobs/sha256:abc")
        );
        assert_eq!(header_url(None, "abc"), None);
    }
}
