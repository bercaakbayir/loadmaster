//! Execute fixtures against an endpoint and reduce them to a verdict.

use std::path::PathBuf;

use serde::Serialize;
use serde_json::{Map, Value, json};

use crate::check::{eval, query};
use crate::client::{Client, Exchange};
use crate::detect::Detected;
use crate::fixture::{Fixture, Kind, parse_ref};
use crate::known::{KnownIssue, matches};

pub const PROBE_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Default)]
pub struct Options {
    pub model: String,
    /// Override every fixture's repeat count.
    pub repeat: Option<u32>,
    /// blake3 of the chat template the endpoint should use (from the model file).
    pub expect_template: Option<String>,
    /// Write each request and raw response here.
    pub capture_dir: Option<PathBuf>,
    /// Context for failure signatures, when known (e.g. from models.lock).
    pub quant: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "result", content = "detail", rename_all = "snake_case")]
pub enum Outcome {
    Pass,
    Fail(String),
    Skip(String),
}

#[derive(Debug, Clone, Serialize)]
pub struct FixtureResult {
    pub id: String,
    pub description: String,
    pub core: bool,
    pub outcome: Outcome,
    /// Each run's outcome; the fixture passes only if all pass.
    pub runs: Vec<Outcome>,
    pub latency_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct Report {
    pub probe_version: String,
    pub url: String,
    pub model: String,
    pub detected: Detected,
    pub quant: Option<String>,
    pub template_blake3: Option<String>,
    pub results: Vec<FixtureResult>,
    /// `pass`, `partial` or `fail`.
    pub verdict: String,
    /// Failure classes, sorted and deduplicated.
    pub failures: Vec<String>,
    pub known_issues: Vec<KnownIssue>,
}

impl Report {
    /// 0 pass, 1 partial, 2 fail.
    pub fn exit_code(&self) -> u8 {
        match self.verdict.as_str() {
            "pass" => 0,
            "partial" => 1,
            _ => 2,
        }
    }
}

/// Replace `{"$from": "N:<path>"}` objects with values from earlier exchanges.
fn substitute(v: &Value, xs: &[Exchange]) -> Result<Value, String> {
    match v {
        Value::Object(o) if o.len() == 1 && o.contains_key("$from") => {
            let r = parse_ref(o["$from"].as_str().ok_or("`$from` takes a string")?)?;
            query(xs, &r)
                .into_iter()
                .next()
                .filter(|v| !v.is_null())
                .ok_or_else(|| format!("{} not found in earlier responses", o["$from"]))
        }
        Value::Object(o) => {
            let mut m = Map::new();
            for (k, x) in o {
                m.insert(k.clone(), substitute(x, xs)?);
            }
            Ok(Value::Object(m))
        }
        Value::Array(a) => a
            .iter()
            .map(|x| substitute(x, xs))
            .collect::<Result<_, _>>()
            .map(Value::Array),
        other => Ok(other.clone()),
    }
}

fn request_body(step_body: &Value, stream: bool, model: &str) -> Value {
    let mut body = json!({ "model": model, "temperature": 0, "seed": 42, "max_tokens": 512 });
    if let Some(o) = step_body.as_object() {
        for (k, v) in o {
            body[k] = v.clone();
        }
    }
    body["stream"] = json!(stream);
    body
}

fn capture(dir: &Option<PathBuf>, fixture: &str, run: u32, step: usize, x: &Exchange) {
    let Some(dir) = dir else { return };
    let d = dir.join(fixture);
    if std::fs::create_dir_all(&d).is_err() {
        return;
    }
    let stem = format!("run{run}-step{step}");
    let _ = std::fs::write(
        d.join(format!("{stem}.request.json")),
        serde_json::to_vec_pretty(&x.request).unwrap_or_default(),
    );
    let _ = std::fs::write(d.join(format!("{stem}.response.raw")), &x.raw);
}

async fn run_once(c: &Client, f: &Fixture, opts: &Options, run: u32) -> (Outcome, u64) {
    let mut xs: Vec<Exchange> = Vec::new();
    let mut latency = 0;
    for (i, step) in f.steps.iter().enumerate() {
        let Ok(body) = substitute(&step.body, &xs) else {
            break; // an earlier step did not produce what this one needs
        };
        let x = c.chat(request_body(&body, step.stream, &opts.model)).await;
        latency += x.latency_ms;
        capture(&opts.capture_dir, &f.id, run, i, &x);
        xs.push(x);
    }
    if xs.is_empty() {
        return (Outcome::Fail("no_response".into()), latency);
    }
    if xs.iter().any(|x| x.status == 0) {
        return (Outcome::Fail("unreachable".into()), latency);
    }
    if eval(&f.pass, &xs) {
        return (Outcome::Pass, latency);
    }
    let class = f
        .fail
        .iter()
        .find(|r| eval(&r.when, &xs))
        .map_or_else(|| "unclassified".to_owned(), |r| r.class.clone());
    (Outcome::Fail(class), latency)
}

fn template_check(detected: &Detected, opts: &Options) -> Outcome {
    let Some(want) = &opts.expect_template else {
        return Outcome::Skip(
            "no expected template; pass --expect-template or run via `loadmaster up --verify`"
                .into(),
        );
    };
    match &detected.chat_template {
        None => Outcome::Skip(format!(
            "{} does not report its chat template",
            detected.runtime
        )),
        Some(t) if blake3::hash(t.as_bytes()).to_hex().as_str() == want => Outcome::Pass,
        Some(_) => Outcome::Fail("template_mismatch".into()),
    }
}

pub async fn run_fixture(
    c: &Client,
    f: &Fixture,
    opts: &Options,
    detected: &Detected,
) -> FixtureResult {
    let (outcome, runs, latency_ms) = match f.kind {
        Kind::Template => {
            let o = template_check(detected, opts);
            (o.clone(), vec![o], 0)
        }
        Kind::Chat => {
            let n = opts.repeat.unwrap_or(f.repeat).max(1);
            let mut runs = Vec::new();
            let mut latency = 0;
            for run in 0..n {
                let (o, l) = run_once(c, f, opts, run).await;
                latency += l;
                runs.push(o);
            }
            let first = runs[0].clone();
            let outcome = if runs.iter().all(|r| *r == first) {
                first
            } else {
                // Disagreeing runs at temperature 0: report the first failure seen.
                runs.iter().find(|r| matches!(r, Outcome::Fail(_))).map_or(
                    Outcome::Fail("nondeterministic".into()),
                    |r| match r {
                        Outcome::Fail(c) => Outcome::Fail(format!(
                            "{c} (in {} of {n} runs)",
                            runs.iter().filter(|x| *x == r).count()
                        )),
                        other => other.clone(),
                    },
                )
            };
            (outcome, runs, latency / u64::from(n))
        }
    };
    FixtureResult {
        id: f.id.clone(),
        description: f.description.clone(),
        core: f.core,
        outcome,
        runs,
        latency_ms,
    }
}

/// Verdict rules (spec §6): `pass` when nothing failed, `partial` when every core
/// fixture passed, `fail` otherwise. Skipped fixtures do not count either way.
pub fn verdict(results: &[FixtureResult]) -> &'static str {
    let failed = |r: &&FixtureResult| matches!(r.outcome, Outcome::Fail(_));
    if !results.iter().any(|r| failed(&r)) {
        "pass"
    } else if results
        .iter()
        .filter(|r| r.core)
        .all(|r| r.outcome == Outcome::Pass)
        && results.iter().any(|r| r.core)
    {
        "partial"
    } else {
        "fail"
    }
}

pub async fn probe(
    c: &Client,
    fixtures: &[Fixture],
    opts: &Options,
    known: &[KnownIssue],
) -> Report {
    let detected = crate::detect::detect(c).await;
    let mut results = Vec::new();
    for f in fixtures {
        results.push(run_fixture(c, f, opts, &detected).await);
    }
    let mut failures: Vec<String> = results
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Fail(c) => Some(c.split(' ').next().unwrap_or(c).to_owned()),
            _ => None,
        })
        .collect();
    failures.sort();
    failures.dedup();
    let known_issues = known
        .iter()
        .filter(|k| {
            failures
                .iter()
                .any(|f| matches(k, f, &detected, &opts.model))
        })
        .cloned()
        .collect();
    Report {
        probe_version: PROBE_VERSION.into(),
        url: c.base().to_owned(),
        model: opts.model.clone(),
        template_blake3: detected
            .chat_template
            .as_ref()
            .map(|t| blake3::hash(t.as_bytes()).to_hex().to_string()),
        quant: opts.quant.clone(),
        verdict: verdict(&results).into(),
        detected,
        results,
        failures,
        known_issues,
    }
}
