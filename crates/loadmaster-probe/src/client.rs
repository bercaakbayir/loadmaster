//! OpenAI-compatible client with raw capture and streaming reassembly.

use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{Value, json};

/// One request and what came back.
#[derive(Debug, Clone, Serialize)]
pub struct Exchange {
    pub request: Value,
    pub status: u16,
    /// The response body exactly as received.
    #[serde(skip)]
    pub raw: Vec<u8>,
    /// The response in non-streamed shape (`choices[0].message`), reassembled from deltas
    /// for streamed requests; for errors, the error body.
    pub response: Value,
    pub stream: Option<StreamInfo>,
    pub latency_ms: u64,
    /// Time to the first streamed chunk.
    pub ttft_ms: Option<u64>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct StreamInfo {
    pub chunks: usize,
    /// `data: [DONE]` arrived.
    pub done: bool,
    /// Lines that were not valid SSE JSON.
    pub bad_lines: usize,
}

#[derive(Clone)]
pub struct Client {
    http: reqwest::Client,
    base: String,
    api_key: Option<String>,
}

impl Client {
    pub fn new(base: &str, timeout: Duration, api_key: Option<String>) -> Self {
        Client {
            http: reqwest::Client::builder()
                .timeout(timeout)
                .build()
                .expect("default client"),
            base: base
                .trim_end_matches('/')
                .trim_end_matches("/v1")
                .to_owned(),
            api_key,
        }
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub async fn get(&self, path: &str) -> Option<(u16, reqwest::header::HeaderMap, Value)> {
        let mut req = self.http.get(format!("{}{path}", self.base));
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let r = req.timeout(Duration::from_secs(5)).send().await.ok()?;
        let status = r.status().as_u16();
        let headers = r.headers().clone();
        let body = r.json().await.unwrap_or(Value::Null);
        Some((status, headers, body))
    }

    /// POST `/v1/chat/completions`.
    pub async fn chat(&self, body: Value) -> Exchange {
        let stream = body["stream"].as_bool().unwrap_or(false);
        let start = Instant::now();
        let mut req = self
            .http
            .post(format!("{}/v1/chat/completions", self.base))
            .json(&body);
        if let Some(k) = &self.api_key {
            req = req.bearer_auth(k);
        }
        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                return Exchange {
                    request: body,
                    status: 0,
                    raw: Vec::new(),
                    response: json!({ "error": { "message": format!("request failed: {e}") } }),
                    stream: None,
                    latency_ms: start.elapsed().as_millis() as u64,
                    ttft_ms: None,
                };
            }
        };
        let status = resp.status().as_u16();
        let sse = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|v| v.starts_with("text/event-stream"));
        let mut raw = Vec::new();
        let mut ttft = None;
        let mut bytes = resp.bytes_stream();
        while let Some(chunk) = bytes.next().await {
            match chunk {
                Ok(c) => {
                    if ttft.is_none() {
                        ttft = Some(start.elapsed().as_millis() as u64);
                    }
                    raw.extend_from_slice(&c);
                }
                Err(_) => break, // a cut connection is a truncated stream
            }
        }
        let latency_ms = start.elapsed().as_millis() as u64;
        let (response, info) = if stream && sse {
            let (r, i) = reassemble(&raw);
            (r, Some(i))
        } else {
            (
                serde_json::from_slice(&raw)
                    .unwrap_or_else(|_| json!({ "raw": String::from_utf8_lossy(&raw) })),
                None,
            )
        };
        Exchange {
            request: body,
            status,
            raw,
            response,
            stream: info,
            latency_ms,
            ttft_ms: if stream { ttft } else { None },
        }
    }
}

/// Fold SSE chunks into one non-streamed completion: content and reasoning concatenated,
/// tool calls merged by `index` (id and name from the first delta, arguments appended).
pub fn reassemble(raw: &[u8]) -> (Value, StreamInfo) {
    let text = String::from_utf8_lossy(raw);
    let mut info = StreamInfo::default();
    let mut content = String::new();
    let mut reasoning = String::new();
    let mut calls: Vec<Value> = Vec::new();
    let mut finish = Value::Null;
    let mut usage = Value::Null;
    let mut model = Value::Null;
    for line in text.lines() {
        let Some(data) = line.strip_prefix("data:").map(str::trim) else {
            continue;
        };
        if data == "[DONE]" {
            info.done = true;
            continue;
        }
        let Ok(chunk) = serde_json::from_str::<Value>(data) else {
            info.bad_lines += 1;
            continue;
        };
        info.chunks += 1;
        if model.is_null() {
            model = chunk["model"].clone();
        }
        if !chunk["usage"].is_null() {
            usage = chunk["usage"].clone();
        }
        let choice = &chunk["choices"][0];
        if !choice["finish_reason"].is_null() {
            finish = choice["finish_reason"].clone();
        }
        let delta = &choice["delta"];
        if let Some(c) = delta["content"].as_str() {
            content.push_str(c);
        }
        for key in ["reasoning_content", "reasoning"] {
            if let Some(r) = delta[key].as_str() {
                reasoning.push_str(r);
            }
        }
        for tc in delta["tool_calls"].as_array().into_iter().flatten() {
            let idx = tc["index"].as_u64().unwrap_or(calls.len() as u64) as usize;
            while calls.len() <= idx {
                calls.push(json!({ "id": null, "type": "function", "function": { "name": null, "arguments": "" } }));
            }
            let slot = &mut calls[idx];
            if !tc["id"].is_null() {
                slot["id"] = tc["id"].clone();
            }
            if let Some(n) = tc["function"]["name"].as_str() {
                slot["function"]["name"] = json!(n);
            }
            if let Some(a) = tc["function"]["arguments"].as_str() {
                let prev = slot["function"]["arguments"]
                    .as_str()
                    .unwrap_or("")
                    .to_owned();
                slot["function"]["arguments"] = json!(prev + a);
            }
        }
    }
    let mut message = json!({ "role": "assistant", "content": if content.is_empty() { Value::Null } else { json!(content) } });
    if !reasoning.is_empty() {
        message["reasoning_content"] = json!(reasoning);
    }
    if !calls.is_empty() {
        message["tool_calls"] = Value::Array(calls);
    }
    (
        json!({ "object": "chat.completion", "model": model, "choices": [{ "index": 0, "message": message, "finish_reason": finish }], "usage": usage }),
        info,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reassembles_split_tool_calls() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c0\",\"function\":{\"name\":\"get\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: [DONE]\n\n",
        );
        let (r, info) = reassemble(raw.as_bytes());
        assert!(info.done);
        assert_eq!(info.chunks, 5);
        let call = &r["choices"][0]["message"]["tool_calls"][0];
        assert_eq!(call["id"], "c0");
        assert_eq!(call["function"]["arguments"], "{\"a\":1}");
        assert_eq!(r["choices"][0]["finish_reason"], "tool_calls");
    }
}
