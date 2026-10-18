//! `probe eval`: the user's prompts against one or more models, deterministic sampling.
//!
//! A prompts file is JSONL; each line is an object with `messages` (or `prompt`) and any
//! of `tools`, `expect_tool`, `expect_args` (a subset the arguments must contain),
//! `schema` (sent as `response_format` and checked), `expect_contains`, and an `id`.

use serde::Serialize;
use serde_json::{Value, json};

use crate::client::Client;

#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub id: String,
    pub messages: Value,
    pub tools: Option<Value>,
    pub expect_tool: Option<String>,
    pub expect_args: Option<Value>,
    pub schema: Option<Value>,
    pub expect_contains: Option<String>,
}

pub fn parse_jsonl(text: &str) -> Result<Vec<Row>, String> {
    let mut rows = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with("//") {
            continue;
        }
        let v: Value = serde_json::from_str(line).map_err(|e| format!("line {}: {e}", i + 1))?;
        let messages = match (v.get("messages"), v.get("prompt").and_then(Value::as_str)) {
            (Some(m), _) => m.clone(),
            (None, Some(p)) => json!([{ "role": "user", "content": p }]),
            (None, None) => return Err(format!("line {}: needs `messages` or `prompt`", i + 1)),
        };
        if let Some(s) = v.get("schema") {
            jsonschema::validator_for(s)
                .map_err(|e| format!("line {}: invalid schema: {e}", i + 1))?;
        }
        rows.push(Row {
            id: v
                .get("id")
                .and_then(Value::as_str)
                .map_or_else(|| format!("{}", i + 1), str::to_owned),
            messages,
            tools: v.get("tools").cloned(),
            expect_tool: v
                .get("expect_tool")
                .and_then(Value::as_str)
                .map(str::to_owned),
            expect_args: v.get("expect_args").cloned(),
            schema: v.get("schema").cloned(),
            expect_contains: v
                .get("expect_contains")
                .and_then(Value::as_str)
                .map(str::to_owned),
        });
    }
    if rows.is_empty() {
        return Err("no prompts".into());
    }
    Ok(rows)
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RowResult {
    pub idx: usize,
    pub id: String,
    /// Every expectation the row states held (a row with none: the request succeeded).
    pub ok: bool,
    pub schema_valid: Option<bool>,
    pub tool_ok: Option<bool>,
    pub latency_ms: u64,
    pub completion_tokens: Option<u64>,
    pub tokens_per_s: Option<f64>,
    pub error: Option<String>,
    /// blake3 of the response message, to spot output changes between runs.
    pub response_blake3: String,
}

/// `want` is contained in `have`: objects by key recursively, everything else by equality.
pub fn subset(want: &Value, have: &Value) -> bool {
    match (want, have) {
        (Value::Object(w), Value::Object(h)) => w
            .iter()
            .all(|(k, v)| h.get(k).is_some_and(|hv| subset(v, hv))),
        _ => want == have,
    }
}

pub fn score(
    row: &Row,
    status: u16,
    response: &Value,
) -> (bool, Option<bool>, Option<bool>, Option<String>) {
    if status != 200 {
        let msg = response["error"]["message"]
            .as_str()
            .unwrap_or("request failed");
        return (
            false,
            row.schema.as_ref().map(|_| false),
            row.expect_tool.as_ref().map(|_| false),
            Some(format!("HTTP {status}: {msg}")),
        );
    }
    let msg = &response["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("");
    let tool_ok = row.expect_tool.as_ref().map(|name| {
        let call = &msg["tool_calls"][0]["function"];
        call["name"].as_str() == Some(name.as_str())
            && row.expect_args.as_ref().is_none_or(|want| {
                call["arguments"]
                    .as_str()
                    .and_then(|a| serde_json::from_str::<Value>(a).ok())
                    .is_some_and(|have| subset(want, &have))
            })
    });
    let schema_valid = row.schema.as_ref().map(|s| {
        jsonschema::validator_for(s)
            .is_ok_and(|v| serde_json::from_str::<Value>(content).is_ok_and(|i| v.is_valid(&i)))
    });
    let contains_ok = row
        .expect_contains
        .as_ref()
        .map(|needle| content.contains(needle.as_str()));
    let ok = tool_ok.unwrap_or(true) && schema_valid.unwrap_or(true) && contains_ok.unwrap_or(true);
    (ok, schema_valid, tool_ok, None)
}

pub async fn run(c: &Client, model: &str, rows: &[Row]) -> Vec<RowResult> {
    let mut out = Vec::new();
    for (idx, row) in rows.iter().enumerate() {
        let mut body = json!({ "model": model, "messages": row.messages, "temperature": 0, "seed": 42, "max_tokens": 1024 });
        if let Some(t) = &row.tools {
            body["tools"] = t.clone();
        }
        if let Some(s) = &row.schema {
            body["response_format"] = json!({ "type": "json_schema", "json_schema": { "name": "eval", "strict": true, "schema": s } });
        }
        let x = c.chat(body).await;
        let (ok, schema_valid, tool_ok, error) = score(row, x.status, &x.response);
        let completion_tokens = x.response["usage"]["completion_tokens"].as_u64();
        let tokens_per_s = completion_tokens
            .filter(|_| x.latency_ms > 0)
            .map(|t| t as f64 / (x.latency_ms as f64 / 1000.0));
        out.push(RowResult {
            idx,
            id: row.id.clone(),
            ok,
            schema_valid,
            tool_ok,
            latency_ms: x.latency_ms,
            completion_tokens,
            tokens_per_s,
            error,
            response_blake3: blake3::hash(
                x.response["choices"][0]["message"].to_string().as_bytes(),
            )
            .to_hex()
            .to_string(),
        });
    }
    out
}
