//! `probe`, `probe eval`, `probe compare`.

use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use anyhow::{Context, bail};
use loadmaster_probe::run::{FixtureResult, Outcome, Report};
use loadmaster_probe::{Client, Options, compare, db, eval, fixture, known};

use crate::{ProbeArgs, ProbeCmd};

fn rt() -> anyhow::Result<tokio::runtime::Runtime> {
    Ok(tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?)
}

fn api_key(var: Option<&str>) -> anyhow::Result<Option<String>> {
    var.map(|v| std::env::var(v).with_context(|| format!("${v} is not set")))
        .transpose()
}

pub fn client(url: &str, key: Option<String>) -> Client {
    Client::new(url, Duration::from_secs(300), key)
}

pub fn run(a: ProbeArgs) -> anyhow::Result<ExitCode> {
    match a.sub {
        Some(ProbeCmd::Eval {
            prompts,
            models,
            url,
            lock,
            api_key_env,
        }) => {
            return run_eval(&prompts, &models, url, &lock, api_key_env.as_deref());
        }
        Some(ProbeCmd::Compare { a, b }) => return run_compare(a, b),
        None => {}
    }
    let url = a.url.context("pass the endpoint URL")?;
    let model = a.model.context("pass --model")?;
    if a.embeddings {
        let c = client(&url, api_key(a.api_key_env.as_deref())?);
        let report = rt()?.block_on(loadmaster_probe::embed::probe(
            &c,
            &model,
            a.repeat.unwrap_or(3),
        ));
        if a.json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            crate::ui::print(&render(&report, crate::ui::style()));
        }
        return Ok(ExitCode::from(report.exit_code()));
    }
    let mut fixtures = match &a.fixtures {
        Some(dir) => fixture::load_dir(dir)?,
        None => fixture::builtin(),
    };
    if !a.only.is_empty() {
        fixtures.retain(|f| a.only.contains(&f.id));
        if fixtures.is_empty() {
            bail!("--only matched no fixture");
        }
    }
    let mut issues = known::builtin();
    if let Some(p) = &a.known_issues {
        let text =
            std::fs::read_to_string(p).with_context(|| format!("reading {}", p.display()))?;
        issues.extend(known::parse(&text).map_err(|e| anyhow::anyhow!("{}: {e}", p.display()))?);
    }
    let capture = capture_dir(&model);
    let opts = Options {
        model,
        repeat: a.repeat,
        expect_template: a.expect_template,
        capture_dir: Some(capture.clone()),
        quant: None,
    };
    let c = client(&url, api_key(a.api_key_env.as_deref())?);
    let report = rt()?.block_on(loadmaster_probe::run::probe(&c, &fixtures, &opts, &issues));
    if a.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        crate::ui::print(&render(&report, crate::ui::style()));
        println!("captures: {}", capture.display());
    }
    Ok(ExitCode::from(report.exit_code()))
}

pub fn capture_dir(model: &str) -> std::path::PathBuf {
    let stamp = loadmaster_probe::time::now().replace([':', '-'], "");
    let safe: String = model
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    super::state_dir()
        .join("probe")
        .join(format!("{stamp}-{safe}"))
}

fn cell(r: &FixtureResult) -> (String, String) {
    match &r.outcome {
        Outcome::Pass => ("pass".into(), String::new()),
        Outcome::Fail(c) => ("FAIL".into(), c.clone()),
        Outcome::Skip(why) => ("skip".into(), why.clone()),
    }
}

pub fn render(r: &Report, style: crate::ui::Style) -> String {
    use std::fmt::Write;
    let mut o = String::new();
    writeln!(
        o,
        "loadmaster probe {} · {} · model {} · {}{}",
        r.probe_version,
        r.url,
        r.model,
        r.detected.runtime,
        r.detected
            .version
            .as_ref()
            .map_or(String::new(), |v| format!(" {v}"))
    )
    .unwrap();
    for x in &r.results {
        let (state, detail) = cell(x);
        // Core fixtures decide `partial`; mark them.
        let id = if x.core {
            format!("{}*", x.id)
        } else {
            x.id.clone()
        };
        // Pad before styling, and not at all when nothing follows.
        let padded = if detail.is_empty() {
            state
        } else {
            format!("{state:<5} {detail}")
        };
        let (word, rest) = padded.split_at(4);
        let word = match &x.outcome {
            Outcome::Pass => style.green(word),
            Outcome::Fail(_) => style.red(word),
            Outcome::Skip(_) => style.dim(word),
        };
        writeln!(o, "  {id:<24} {word}{rest}").unwrap();
    }
    let why = match r.verdict.as_str() {
        "pass" => "every fixture passed".to_owned(),
        "partial" => format!(
            "core fixtures pass ({}); failures: {}",
            r.results
                .iter()
                .filter(|x| x.core)
                .map(|x| x.id.as_str())
                .collect::<Vec<_>>()
                .join(", "),
            r.failures.join(", ")
        ),
        _ => format!("failures: {}", r.failures.join(", ")),
    };
    let verdict = match r.verdict.as_str() {
        "pass" => style.green(&r.verdict),
        "partial" => style.yellow(&r.verdict),
        _ => style.red(&r.verdict),
    };
    writeln!(o, "verdict: {verdict} ({why}); * = core fixture").unwrap();
    for k in &r.known_issues {
        writeln!(o, "known issue ({}): {} — {}", k.failure, k.url, k.note).unwrap();
    }
    o
}

fn run_eval(
    prompts: &Path,
    models: &[String],
    url: Option<String>,
    lock: &Path,
    key_var: Option<&str>,
) -> anyhow::Result<ExitCode> {
    let text = std::fs::read_to_string(prompts)
        .with_context(|| format!("reading {}", prompts.display()))?;
    let rows =
        eval::parse_jsonl(&text).map_err(|e| anyhow::anyhow!("{}: {e}", prompts.display()))?;
    let url = match url {
        Some(u) => u,
        None => format!(
            "http://{}",
            loadmaster_lock::Lockfile::load(lock)
                .context("pass --url, or run where models.lock is")?
                .proxy
                .listen
        ),
    };
    let c = client(&url, api_key(key_var)?);
    let mut database = db::Db::open(&super::state_dir().join("eval.db"))?;
    let digest = blake3::hash(text.as_bytes()).to_hex().to_string();
    let rt = rt()?;
    let detected = rt.block_on(loadmaster_probe::detect::detect(&c));
    for model in models {
        let results = rt.block_on(eval::run(&c, model, &rows));
        let id = database.insert(
            &loadmaster_probe::time::now(),
            &url,
            model,
            &detected.runtime,
            detected.version.as_deref(),
            &prompts.display().to_string(),
            &digest,
            &results,
        )?;
        let s = database.summary(id)?;
        println!(
            "run {id}: {model}  accuracy {}/{}  schema {}/{}  p50 {}ms  p95 {}ms{}",
            s.ok,
            s.n,
            s.schema_ok,
            s.schema_n,
            s.latency_p50_ms,
            s.latency_p95_ms,
            s.tokens_per_s_mean
                .map_or(String::new(), |t| format!("  {t:.1} tok/s"))
        );
    }
    println!("compare runs with `loadmaster probe compare <run> <run>`");
    Ok(ExitCode::SUCCESS)
}

fn run_compare(a: i64, b: i64) -> anyhow::Result<ExitCode> {
    let database = db::Db::open(&super::state_dir().join("eval.db"))?;
    let (ma, mb) = (
        database.run(a).with_context(|| format!("no run {a}"))?,
        database.run(b).with_context(|| format!("no run {b}"))?,
    );
    let (sa, sb) = (database.summary(a)?, database.summary(b)?);
    let (ra, rb) = (database.results(a)?, database.results(b)?);
    let changed = ra
        .iter()
        .zip(&rb)
        .filter(|(x, y)| x.response_blake3 != y.response_blake3)
        .count();
    print!("{}", compare::render(&ma, &sa, &mb, &sb, changed));
    Ok(ExitCode::SUCCESS)
}
