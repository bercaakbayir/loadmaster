//! Every built-in fixture against the fake runtime: all pass when it behaves, and each
//! misbehaviour yields its failure class on the fixture that should catch it.

use std::collections::HashSet;
use std::time::Duration;

use loadmaster_probe::fake::{self, Impersonate, Misbehavior};
use loadmaster_probe::run::{self, Outcome};
use loadmaster_probe::{Client, Options, fixture};

async fn start(misbehave: &[Misbehavior], impersonate: Impersonate) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let cfg = fake::Config {
        misbehave: misbehave.iter().copied().collect::<HashSet<_>>(),
        impersonate,
        ..fake::Config::default()
    };
    tokio::spawn(fake::serve(listener, cfg));
    format!("http://{addr}")
}

fn opts() -> Options {
    Options {
        model: "fake-model".into(),
        repeat: Some(1),
        ..Options::default()
    }
}

async fn probe(misbehave: &[Misbehavior]) -> loadmaster_probe::Report {
    let url = start(misbehave, Impersonate::LlamaCpp).await;
    let c = Client::new(&url, Duration::from_secs(10), None);
    run::probe(&c, &fixture::builtin(), &opts(), &[]).await
}

fn outcome<'a>(r: &'a loadmaster_probe::Report, id: &str) -> &'a Outcome {
    &r.results
        .iter()
        .find(|x| x.id == id)
        .unwrap_or_else(|| panic!("no fixture {id}"))
        .outcome
}

#[test]
fn builtin_fixtures_parse() {
    let ids: Vec<String> = fixture::builtin().into_iter().map(|f| f.id).collect();
    assert_eq!(
        ids,
        [
            "template.hash",
            "tool.single",
            "tool.parallel",
            "tool.nested",
            "tool.result_roundtrip",
            "schema.strict",
            "schema.union",
            "schema.enum_pattern",
            "stream.tool",
            "reasoning.leak"
        ]
    );
    let core: Vec<String> = fixture::builtin()
        .into_iter()
        .filter(|f| f.core)
        .map(|f| f.id)
        .collect();
    assert_eq!(
        core,
        ["tool.single", "tool.result_roundtrip", "schema.strict"]
    );
}

#[tokio::test]
async fn a_correct_runtime_passes_everything() {
    let r = probe(&[]).await;
    for x in &r.results {
        let ok = matches!(x.outcome, Outcome::Pass)
            || (x.id == "template.hash" && matches!(x.outcome, Outcome::Skip(_)));
        assert!(ok, "{}: {:?}", x.id, x.outcome);
    }
    assert_eq!(r.verdict, "pass");
    assert_eq!(r.exit_code(), 0);
    assert_eq!(r.detected.runtime, "llama.cpp");
    assert_eq!(r.detected.version.as_deref(), Some("b6700-abc1234"));
}

#[tokio::test]
async fn each_misbehaviour_is_classified() {
    use Misbehavior::*;
    let cases: &[(Misbehavior, &str, &str)] = &[
        (CallInContent, "tool.single", "call_in_content"),
        (CallInReasoning, "tool.single", "call_in_reasoning"),
        (CallInReasoning, "reasoning.leak", "call_in_reasoning"),
        (ArgsInvalidJson, "tool.single", "args_invalid_json"),
        (NoCall, "tool.single", "no_call"),
        (ParallelSerialised, "tool.parallel", "parallel_serialised"),
        (ParallelInContent, "tool.parallel", "parallel_in_content"),
        (SchemaFlattened, "tool.nested", "schema_flattened"),
        (ArgsInvalidJson, "tool.nested", "args_invalid_json"),
        (LoopsOnTool, "tool.result_roundtrip", "loops_on_tool"),
        (IgnoresResult, "tool.result_roundtrip", "ignores_result"),
        (NoCall, "tool.result_roundtrip", "no_call"),
        (
            SchemaDroppedSilent,
            "schema.strict",
            "schema_dropped_silent",
        ),
        (SchemaRejected, "schema.strict", "schema_rejected"),
        (SchemaRejected, "schema.union", "schema_rejected"),
        (SchemaDroppedSilent, "schema.union", "union_unconstrained"),
        (EnumIgnored, "schema.enum_pattern", "enum_ignored"),
        (PatternIgnored, "schema.enum_pattern", "pattern_ignored"),
        (StreamTruncated, "stream.tool", "stream_truncated"),
        (
            StreamArgsSplitInvalid,
            "stream.tool",
            "stream_args_split_invalid",
        ),
        (RuntimeParseError, "tool.single", "runtime_parse_error"),
        (RuntimeParseError, "tool.parallel", "runtime_parse_error"),
        (RuntimeParseError, "tool.nested", "runtime_parse_error"),
        (
            RuntimeParseError,
            "tool.result_roundtrip",
            "runtime_parse_error",
        ),
        (RuntimeParseError, "stream.tool", "runtime_parse_error"),
    ];
    for (m, fixture, class) in cases {
        let r = probe(&[*m]).await;
        assert_eq!(
            outcome(&r, fixture),
            &Outcome::Fail((*class).into()),
            "{m:?} on {fixture}"
        );
    }
}

/// llama-server answers a streamed request with 200, then an SSE `error` event: the
/// stream.tool rule must catch that even when the non-streamed step succeeded.
#[tokio::test]
async fn a_parser_error_mid_stream_is_classified() {
    let good = Client::new(
        &start(&[], Impersonate::LlamaCpp).await,
        Duration::from_secs(10),
        None,
    );
    let bad = Client::new(
        &start(&[Misbehavior::RuntimeParseError], Impersonate::LlamaCpp).await,
        Duration::from_secs(10),
        None,
    );
    let f = fixture::builtin()
        .into_iter()
        .find(|f| f.id == "stream.tool")
        .unwrap();
    let body = |stream: bool, b: &serde_json::Value| {
        let mut b = b.clone();
        b["model"] = "fake-model".into();
        b["stream"] = stream.into();
        b
    };
    let x0 = good.chat(body(false, &f.steps[0].body)).await;
    let x1 = bad.chat(body(true, &f.steps[1].body)).await;
    assert_eq!((x0.status, x1.status), (200, 200));
    assert!(!x1.stream.as_ref().unwrap().done);
    assert_eq!(x1.response["error"]["code"], 500);
    let xs = [x0, x1];
    let class = f
        .fail
        .iter()
        .find(|r| loadmaster_probe::check::eval(&r.when, &xs))
        .map(|r| r.class.as_str());
    assert_eq!(class, Some("runtime_parse_error"));
}

/// A non-streamed 500 for an unrelated reason stays out of runtime_parse_error.
#[tokio::test]
async fn only_parser_messages_are_parse_errors() {
    let f = fixture::builtin()
        .into_iter()
        .find(|f| f.id == "tool.single")
        .unwrap();
    let x = |status: u16, msg: &str| loadmaster_probe::client::Exchange {
        request: serde_json::Value::Null,
        status,
        raw: Vec::new(),
        response: serde_json::json!({ "error": { "code": status, "message": msg } }),
        stream: None,
        latency_ms: 0,
        ttft_ms: None,
    };
    let rule = f
        .fail
        .iter()
        .find(|r| r.class == "runtime_parse_error")
        .unwrap();
    let eval = |x| loadmaster_probe::check::eval(&rule.when, &[x]);
    assert!(eval(x(500, "Failed to parse tool call arguments as JSON")));
    assert!(!eval(x(500, "out of memory")));
    assert!(!eval(x(400, "unsupported response format")));
}

#[tokio::test]
async fn verdicts_and_exit_codes() {
    // Only a non-core fixture fails: partial.
    let r = probe(&[Misbehavior::ParallelSerialised]).await;
    assert_eq!((r.verdict.as_str(), r.exit_code()), ("partial", 1));
    assert_eq!(r.failures, ["parallel_serialised"]);
    // A core fixture fails: fail.
    let r = probe(&[Misbehavior::SchemaRejected]).await;
    assert_eq!((r.verdict.as_str(), r.exit_code()), ("fail", 2));
}

#[tokio::test]
async fn template_check_compares_hashes() {
    let url = start(&[], Impersonate::LlamaCpp).await;
    let c = Client::new(&url, Duration::from_secs(10), None);
    let fixtures: Vec<_> = fixture::builtin()
        .into_iter()
        .filter(|f| f.id == "template.hash")
        .collect();
    let good = blake3::hash(fake::Config::default().chat_template.as_bytes())
        .to_hex()
        .to_string();
    let r = run::probe(
        &c,
        &fixtures,
        &Options {
            expect_template: Some(good),
            ..opts()
        },
        &[],
    )
    .await;
    assert_eq!(r.results[0].outcome, Outcome::Pass);
    let r = run::probe(
        &c,
        &fixtures,
        &Options {
            expect_template: Some("00".repeat(32)),
            ..opts()
        },
        &[],
    )
    .await;
    assert_eq!(
        r.results[0].outcome,
        Outcome::Fail("template_mismatch".into())
    );
}

#[tokio::test]
async fn detects_every_runtime() {
    for (imp, name) in [
        (Impersonate::LlamaCpp, "llama.cpp"),
        (Impersonate::Vllm, "vllm"),
        (Impersonate::Ollama, "ollama"),
        (Impersonate::LmStudio, "lmstudio"),
        (Impersonate::SgLang, "sglang"),
        (Impersonate::TabbyApi, "tabbyapi"),
    ] {
        let url = start(&[], imp).await;
        let d = loadmaster_probe::detect::detect(&Client::new(&url, Duration::from_secs(5), None))
            .await;
        assert_eq!(d.runtime, name);
        assert_eq!(d.models, ["fake-model"]);
    }
}

#[tokio::test]
async fn repeats_must_agree() {
    let url = start(&[], Impersonate::LlamaCpp).await;
    let c = Client::new(&url, Duration::from_secs(10), None);
    let f: Vec<_> = fixture::builtin()
        .into_iter()
        .filter(|f| f.id == "tool.single")
        .collect();
    let r = run::probe(
        &c,
        &f,
        &Options {
            repeat: None,
            ..opts()
        },
        &[],
    )
    .await;
    assert_eq!(r.results[0].runs.len(), 3, "default repeat is 3");
    assert_eq!(r.results[0].outcome, Outcome::Pass);
}

#[test]
fn known_issues_match_by_class_runtime_and_model() {
    let list = loadmaster_probe::known::parse(
        r#"
[[issue]]
failure = "call_in_content"
runtime = "llama.cpp"
model = "qwen3"
url = "https://example.invalid/1"
note = "test"
"#,
    )
    .unwrap();
    let d = loadmaster_probe::detect::Detected {
        runtime: "llama.cpp".into(),
        version: None,
        chat_template: None,
        models: vec![],
    };
    assert!(loadmaster_probe::known::matches(
        &list[0],
        "call_in_content",
        &d,
        "Qwen3-8B"
    ));
    assert!(!loadmaster_probe::known::matches(
        &list[0],
        "call_in_content",
        &d,
        "llama-3"
    ));
    assert!(
        loadmaster_probe::known::builtin().is_empty(),
        "no uncurated entries ship"
    );
}

#[test]
fn a_new_fixture_is_just_a_file() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("x.yaml"),
        "id: custom.hello\ndescription: says hello\nsteps:\n  - body: { messages: [{ role: user, content: hi }] }\npass: { path: \"$.choices[0].message.content\", contains_any: [Hello] }\n",
    )
    .unwrap();
    let f = fixture::load_dir(dir.path()).unwrap();
    assert_eq!(f[0].id, "custom.hello");
    let err = fixture::parse(
        "bad.yaml",
        "id: x\nsteps: [{ body: {} }]\npass: { path: \"$.a\" }\n",
    )
    .unwrap_err();
    assert!(err.to_string().contains("no test"), "{err}");
}
