//! A scriptable fake OpenAI-compatible server for tests (and `loadmaster __fake-runtime`).
//!
//! By default it behaves like a correct runtime: tools in the request produce `tool_calls`
//! whose arguments are built from the tool's parameter schema, `response_format` produces
//! JSON built from the schema, a `tool` message produces a final answer, and streaming
//! splits everything across deltas. Each [`Misbehavior`] breaks one of those the way real
//! runtimes do, so every failure class the probe knows can be exercised.

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use axum::Router;
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde_json::{Value, json};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Misbehavior {
    /// The tool call is written as JSON text in `content`, `tool_calls` stays empty.
    CallInContent,
    /// The tool call appears in `reasoning_content` (or a `<think>` block), not `tool_calls`.
    CallInReasoning,
    /// `arguments` is not valid JSON.
    ArgsInvalidJson,
    /// Answers in prose instead of calling a tool.
    NoCall,
    /// Several requested tools: only the first is called.
    ParallelSerialised,
    /// Several requested tools: all calls written into `content`.
    ParallelInContent,
    /// Nested object/array arguments flattened to strings.
    SchemaFlattened,
    /// After a tool result, calls the same tool again instead of answering.
    LoopsOnTool,
    /// After a tool result, answers without using it.
    IgnoresResult,
    /// `response_format` ignored: HTTP 200 with prose.
    SchemaDroppedSilent,
    /// `response_format` with a schema answered with HTTP 400.
    SchemaRejected,
    /// `enum` constraints ignored.
    EnumIgnored,
    /// `pattern` constraints ignored.
    PatternIgnored,
    /// Stream ends mid tool call: no finish_reason, no `[DONE]`.
    StreamTruncated,
    /// Streamed argument fragments do not concatenate to valid JSON.
    StreamArgsSplitInvalid,
}

impl Misbehavior {
    pub const ALL: &[(&str, Misbehavior)] = &[
        ("call_in_content", Misbehavior::CallInContent),
        ("call_in_reasoning", Misbehavior::CallInReasoning),
        ("args_invalid_json", Misbehavior::ArgsInvalidJson),
        ("no_call", Misbehavior::NoCall),
        ("parallel_serialised", Misbehavior::ParallelSerialised),
        ("parallel_in_content", Misbehavior::ParallelInContent),
        ("schema_flattened", Misbehavior::SchemaFlattened),
        ("loops_on_tool", Misbehavior::LoopsOnTool),
        ("ignores_result", Misbehavior::IgnoresResult),
        ("schema_dropped_silent", Misbehavior::SchemaDroppedSilent),
        ("schema_rejected", Misbehavior::SchemaRejected),
        ("enum_ignored", Misbehavior::EnumIgnored),
        ("pattern_ignored", Misbehavior::PatternIgnored),
        ("stream_truncated", Misbehavior::StreamTruncated),
        (
            "stream_args_split_invalid",
            Misbehavior::StreamArgsSplitInvalid,
        ),
    ];

    pub fn parse(name: &str) -> Option<Self> {
        Self::ALL.iter().find(|(n, _)| *n == name).map(|(_, m)| *m)
    }
}

/// Which runtime's identifying endpoints and headers to imitate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Impersonate {
    LlamaCpp,
    Vllm,
    Ollama,
    LmStudio,
    SgLang,
    TabbyApi,
}

impl Impersonate {
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "llama.cpp" => Impersonate::LlamaCpp,
            "vllm" => Impersonate::Vllm,
            "ollama" => Impersonate::Ollama,
            "lmstudio" => Impersonate::LmStudio,
            "sglang" => Impersonate::SgLang,
            "tabbyapi" => Impersonate::TabbyApi,
            _ => return None,
        })
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub model: String,
    pub misbehave: HashSet<Misbehavior>,
    pub impersonate: Impersonate,
    /// Chat template text served on llama.cpp's `/props`.
    pub chat_template: String,
    /// Prompt and generation speed (tokens/s) reported in llama.cpp `/completion` timings.
    pub speed: (f64, f64),
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: "fake-model".into(),
            misbehave: HashSet::new(),
            impersonate: Impersonate::LlamaCpp,
            chat_template: "{% for m in messages %}{{ m.content }}{% endfor %}".into(),
            speed: (1000.0, 50.0),
        }
    }
}

struct Fake {
    cfg: Config,
    requests: AtomicU64,
}

pub fn router(cfg: Config) -> Router {
    let state = Arc::new(Fake {
        cfg,
        requests: AtomicU64::new(0),
    });
    let mut r = Router::new()
        .route(
            "/health",
            get(|| async { axum::Json(json!({ "status": "ok" })) }),
        )
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat));
    r = match state.cfg.impersonate {
        Impersonate::LlamaCpp => r
            .route("/props", get(props))
            .route("/completion", post(native_completion)),
        Impersonate::Vllm => r.route(
            "/version",
            get(|| async { axum::Json(json!({ "version": "0.11.0" })) }),
        ),
        Impersonate::Ollama => r.route(
            "/api/version",
            get(|| async { axum::Json(json!({ "version": "0.12.5" })) }),
        ),
        Impersonate::LmStudio => r.route("/api/v0/models", get(lmstudio_models)),
        Impersonate::SgLang => r.route("/get_model_info", get(sglang_info)),
        Impersonate::TabbyApi => r.route("/v1/model", get(tabby_model)),
    };
    r.with_state(state)
}

/// Serve on `listener` until the process ends.
pub async fn serve(listener: tokio::net::TcpListener, cfg: Config) -> std::io::Result<()> {
    axum::serve(listener, router(cfg)).await
}

type St = State<Arc<Fake>>;

async fn models(State(f): St) -> Response {
    let mut m = json!({ "id": f.cfg.model, "object": "model", "created": 0, "owned_by": "fake" });
    match f.cfg.impersonate {
        Impersonate::LlamaCpp => m["owned_by"] = json!("llamacpp"),
        Impersonate::Vllm => {
            m["owned_by"] = json!("vllm");
            m["max_model_len"] = json!(32768);
        }
        Impersonate::Ollama => m["owned_by"] = json!("library"),
        Impersonate::LmStudio => m["owned_by"] = json!("organization_owner"),
        Impersonate::SgLang => m["owned_by"] = json!("sglang"),
        Impersonate::TabbyApi => m["owned_by"] = json!("tabbyAPI"),
    }
    let mut resp = axum::Json(json!({ "object": "list", "data": [m] })).into_response();
    if f.cfg.impersonate == Impersonate::Ollama {
        resp.headers_mut()
            .insert(header::SERVER, HeaderValue::from_static("ollama"));
    }
    resp
}

async fn props(State(f): St) -> Response {
    axum::Json(json!({
        "chat_template": f.cfg.chat_template,
        "build_info": "b6700-abc1234",
        "model_path": format!("/models/{}.gguf", f.cfg.model),
    }))
    .into_response()
}

/// llama.cpp's native `/completion`: text plus `timings` at the configured speeds. A
/// string prompt counts one token per whitespace-separated word.
async fn native_completion(State(f): St, axum::Json(body): axum::Json<Value>) -> Response {
    let prompt_n = match &body["prompt"] {
        Value::Array(t) => t.len(),
        Value::String(s) => s.split_whitespace().count(),
        _ => 0,
    } as f64;
    let predicted_n = body["n_predict"].as_u64().unwrap_or(16) as f64;
    let (pp, tg) = f.cfg.speed;
    let (prompt_ms, predicted_ms) = (prompt_n / pp * 1000.0, predicted_n / tg * 1000.0);
    axum::Json(json!({
        "content": "fake ".repeat(predicted_n as usize),
        "tokens_predicted": predicted_n,
        "tokens_evaluated": prompt_n,
        "timings": {
            "cache_n": 0,
            "prompt_n": prompt_n,
            "prompt_ms": prompt_ms,
            "prompt_per_token_ms": if prompt_n > 0.0 { prompt_ms / prompt_n } else { 0.0 },
            "prompt_per_second": pp,
            "predicted_n": predicted_n,
            "predicted_ms": predicted_ms,
            "predicted_per_token_ms": predicted_ms / predicted_n.max(1.0),
            "predicted_per_second": tg,
        }
    }))
    .into_response()
}

async fn lmstudio_models(State(f): St) -> Response {
    axum::Json(json!({ "data": [{ "id": f.cfg.model, "type": "llm", "arch": "qwen3", "quantization": "Q4_K_M", "state": "loaded" }] })).into_response()
}

async fn sglang_info(State(f): St) -> Response {
    axum::Json(json!({ "model_path": f.cfg.model, "is_generation": true, "version": "0.5.3" }))
        .into_response()
}

async fn tabby_model(State(f): St) -> Response {
    axum::Json(json!({ "id": f.cfg.model, "object": "model", "owned_by": "tabbyAPI", "parameters": { "max_seq_len": 32768 } })).into_response()
}

/// A minimal instance of a JSON Schema: required properties, first enum value, first
/// `examples` entry (which is how fixtures satisfy `pattern`), first `anyOf` branch.
pub fn instance(schema: &Value) -> Value {
    if let Some(v) = schema.get("const") {
        return v.clone();
    }
    if let Some(v) = schema.get("examples").and_then(|e| e.get(0)) {
        return v.clone();
    }
    if let Some(v) = schema.get("default") {
        return v.clone();
    }
    if let Some(v) = schema.get("enum").and_then(|e| e.get(0)) {
        return v.clone();
    }
    for key in ["anyOf", "oneOf"] {
        if let Some(first) = schema.get(key).and_then(|b| b.get(0)) {
            return instance(first);
        }
    }
    let ty = match schema.get("type") {
        Some(Value::String(t)) => t.as_str(),
        Some(Value::Array(ts)) => ts
            .iter()
            .filter_map(Value::as_str)
            .find(|t| *t != "null")
            .unwrap_or("null"),
        _ if schema.get("properties").is_some() => "object",
        _ => "string",
    };
    match ty {
        "object" => {
            let props = schema.get("properties").and_then(Value::as_object);
            let mut out = serde_json::Map::new();
            for (k, v) in props.into_iter().flatten() {
                out.insert(k.clone(), instance(v));
            }
            Value::Object(out)
        }
        "array" => {
            let n = schema
                .get("minItems")
                .and_then(Value::as_u64)
                .unwrap_or(1)
                .max(1);
            let item = schema.get("items").map_or(json!("item"), instance);
            Value::Array(vec![item; n as usize])
        }
        "integer" => json!(schema.get("minimum").and_then(Value::as_i64).unwrap_or(1)),
        "number" => json!(schema.get("minimum").and_then(Value::as_f64).unwrap_or(1.5)),
        "boolean" => json!(true),
        "null" => Value::Null,
        _ => json!("example"),
    }
}

/// Break constraints the way runtimes that ignore them do.
fn violate(v: &mut Value, schema: &Value, bad: &HashSet<Misbehavior>) {
    if bad.contains(&Misbehavior::EnumIgnored) && schema.get("enum").is_some() {
        *v = json!("not-in-enum");
    }
    if bad.contains(&Misbehavior::PatternIgnored) && schema.get("pattern").is_some() {
        *v = json!("does not match");
    }
    if let (Some(obj), Some(props)) = (
        v.as_object_mut(),
        schema.get("properties").and_then(Value::as_object),
    ) {
        for (k, s) in props {
            if let Some(child) = obj.get_mut(k) {
                violate(child, s, bad);
            }
        }
    }
}

struct Call {
    name: String,
    arguments: String,
}

enum Reply {
    Text {
        content: String,
        reasoning: Option<String>,
    },
    Calls(Vec<Call>),
    Error(StatusCode, String),
}

fn decide(f: &Fake, req: &Value) -> Reply {
    let bad = &f.cfg.misbehave;
    let messages = req["messages"].as_array().cloned().unwrap_or_default();
    let last = messages.last().cloned().unwrap_or(Value::Null);

    // A tool result: answer with it, unless told to misbehave.
    if last["role"] == "tool" {
        let result = last["content"].as_str().unwrap_or("").to_owned();
        if bad.contains(&Misbehavior::LoopsOnTool) {
            if let Some(tool) = req["tools"].get(0) {
                return Reply::Calls(vec![call_for(tool, bad)]);
            }
        }
        let content = if bad.contains(&Misbehavior::IgnoresResult) {
            "I could not find that information.".to_owned()
        } else {
            format!("Based on the tool result: {result}")
        };
        return Reply::Text {
            content,
            reasoning: None,
        };
    }

    if let Some(tools) = req["tools"].as_array().filter(|t| !t.is_empty()) {
        if bad.contains(&Misbehavior::NoCall) {
            return Reply::Text {
                content: "I would need to look that up.".into(),
                reasoning: None,
            };
        }
        let mut calls: Vec<Call> = tools.iter().map(|t| call_for(t, bad)).collect();
        if bad.contains(&Misbehavior::ParallelSerialised) {
            calls.truncate(1);
        }
        let as_text = |calls: &[Call]| {
            calls
                .iter()
                .map(|c| {
                    format!(
                        "{{\"name\": \"{}\", \"arguments\": {}}}",
                        c.name, c.arguments
                    )
                })
                .collect::<Vec<_>>()
                .join("\n")
        };
        if bad.contains(&Misbehavior::ParallelInContent) && calls.len() > 1
            || bad.contains(&Misbehavior::CallInContent)
        {
            return Reply::Text {
                content: format!("<tool_call>\n{}\n</tool_call>", as_text(&calls)),
                reasoning: None,
            };
        }
        if bad.contains(&Misbehavior::CallInReasoning) {
            return Reply::Text {
                content: String::new(),
                reasoning: Some(format!(
                    "I should call the tool.\n<tool_call>{}</tool_call>",
                    as_text(&calls)
                )),
            };
        }
        return Reply::Calls(calls);
    }

    if let Some(rf) = req
        .get("response_format")
        .filter(|r| r["type"] == "json_schema")
    {
        if bad.contains(&Misbehavior::SchemaRejected) {
            return Reply::Error(
                StatusCode::BAD_REQUEST,
                "response_format json_schema is not supported".into(),
            );
        }
        if bad.contains(&Misbehavior::SchemaDroppedSilent) {
            return Reply::Text {
                content: "Sure! Here is the information you asked for.".into(),
                reasoning: None,
            };
        }
        let schema = &rf["json_schema"]["schema"];
        let mut v = instance(schema);
        violate(&mut v, schema, bad);
        return Reply::Text {
            content: v.to_string(),
            reasoning: None,
        };
    }

    Reply::Text {
        content: "Hello from the fake runtime.".into(),
        reasoning: None,
    }
}

fn call_for(tool: &Value, bad: &HashSet<Misbehavior>) -> Call {
    let f = &tool["function"];
    let schema = &f["parameters"];
    let mut args = instance(schema);
    if bad.contains(&Misbehavior::SchemaFlattened)
        && let Some(obj) = args.as_object_mut()
    {
        for v in obj.values_mut() {
            if v.is_object() || v.is_array() {
                *v = Value::String(v.to_string());
            }
        }
    }
    let mut arguments = args.to_string();
    if bad.contains(&Misbehavior::ArgsInvalidJson) {
        arguments = arguments
            .replacen('"', "'", 2)
            .trim_end_matches('}')
            .to_owned();
    }
    Call {
        name: f["name"].as_str().unwrap_or("tool").to_owned(),
        arguments,
    }
}

async fn chat(State(f): St, axum::Json(req): axum::Json<Value>) -> Response {
    let n = f.requests.fetch_add(1, Ordering::SeqCst);
    let id = format!("chatcmpl-fake{n}");
    let stream = req["stream"].as_bool().unwrap_or(false);
    match decide(&f, &req) {
        Reply::Error(status, msg) => (
            status,
            axum::Json(json!({ "error": { "message": msg, "type": "invalid_request_error" } })),
        )
            .into_response(),
        reply if stream => sse(&f, &id, reply),
        Reply::Text { content, reasoning } => {
            let mut msg = json!({ "role": "assistant", "content": content });
            if let Some(r) = reasoning {
                msg["reasoning_content"] = json!(r);
            }
            axum::Json(completion(&f, &id, msg, "stop")).into_response()
        }
        Reply::Calls(calls) => {
            let tool_calls: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(i, c)| json!({ "id": format!("call_{i}"), "type": "function", "function": { "name": c.name, "arguments": c.arguments } }))
                .collect();
            let msg = json!({ "role": "assistant", "content": null, "tool_calls": tool_calls });
            axum::Json(completion(&f, &id, msg, "tool_calls")).into_response()
        }
    }
}

fn completion(f: &Fake, id: &str, message: Value, finish: &str) -> Value {
    json!({
        "id": id, "object": "chat.completion", "created": 0, "model": f.cfg.model,
        "choices": [{ "index": 0, "message": message, "finish_reason": finish }],
        "usage": { "prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20 },
    })
}

fn sse(f: &Fake, id: &str, reply: Reply) -> Response {
    let bad = &f.cfg.misbehave;
    let chunk = |delta: Value, finish: Option<&str>| {
        let c = json!({ "id": id, "object": "chat.completion.chunk", "created": 0, "model": f.cfg.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish }] });
        format!("data: {c}\n\n")
    };
    let mut events = vec![chunk(json!({ "role": "assistant" }), None)];
    let mut truncated = false;
    match reply {
        Reply::Text { content, reasoning } => {
            if let Some(r) = reasoning {
                events.push(chunk(json!({ "reasoning_content": r }), None));
            }
            for piece in split_in(&content, 3) {
                events.push(chunk(json!({ "content": piece }), None));
            }
            events.push(chunk(json!({}), Some("stop")));
        }
        Reply::Calls(calls) => {
            for (i, c) in calls.iter().enumerate() {
                events.push(chunk(json!({ "tool_calls": [{ "index": i, "id": format!("call_{i}"), "type": "function", "function": { "name": c.name, "arguments": "" } }] }), None));
                let mut pieces = split_in(&c.arguments, 3);
                if bad.contains(&Misbehavior::StreamArgsSplitInvalid) && pieces.len() > 1 {
                    pieces.remove(1);
                }
                for (j, p) in pieces.iter().enumerate() {
                    if bad.contains(&Misbehavior::StreamTruncated) && j == pieces.len() - 1 {
                        truncated = true;
                        break;
                    }
                    events.push(chunk(
                        json!({ "tool_calls": [{ "index": i, "function": { "arguments": p } }] }),
                        None,
                    ));
                }
                if truncated {
                    break;
                }
            }
            if !truncated {
                events.push(chunk(json!({}), Some("tool_calls")));
            }
        }
        Reply::Error(..) => unreachable!("errors are not streamed"),
    }
    if !truncated {
        events.push("data: [DONE]\n\n".into());
    }
    let body = Body::from_stream(futures_util::stream::iter(
        events.into_iter().map(Ok::<_, std::io::Error>),
    ));
    Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .body(body)
        .expect("static response")
}

/// `s` in up to `n` pieces on char boundaries.
fn split_in(s: &str, n: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.is_empty() {
        return vec![];
    }
    let size = chars.len().div_ceil(n);
    chars.chunks(size).map(|c| c.iter().collect()).collect()
}
