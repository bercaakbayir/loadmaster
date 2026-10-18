//! Which runtime is behind an OpenAI-compatible URL, from its identifying endpoints.

use serde::Serialize;
use serde_json::Value;

use crate::client::Client;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Detected {
    /// `llama.cpp`, `ollama`, `vllm`, `lmstudio`, `sglang`, `tabbyapi`, or `unknown`.
    pub runtime: String,
    pub version: Option<String>,
    /// The chat template the runtime reports using (llama.cpp `/props`).
    #[serde(skip)]
    pub chat_template: Option<String>,
    pub models: Vec<String>,
}

fn str_of(v: &Value, k: &str) -> Option<String> {
    v.get(k).and_then(Value::as_str).map(str::to_owned)
}

/// Probe in an order where no runtime matches another's check: llama.cpp `/props`,
/// SGLang `/get_model_info`, LM Studio `/api/v0/models`, Ollama `/api/version`, TabbyAPI
/// `/v1/model`, vLLM `/version`; then `owned_by` in `/v1/models`.
pub async fn detect(c: &Client) -> Detected {
    let models_body = c.get("/v1/models").await;
    let models: Vec<String> = models_body
        .as_ref()
        .and_then(|(_, _, b)| b["data"].as_array().cloned())
        .unwrap_or_default()
        .iter()
        .filter_map(|m| str_of(m, "id"))
        .collect();
    let found = |runtime: &str, version: Option<String>, chat_template: Option<String>| Detected {
        runtime: runtime.into(),
        version,
        chat_template,
        models: models.clone(),
    };
    let ok = |r: &Option<(u16, reqwest::header::HeaderMap, Value)>| -> Option<Value> {
        r.as_ref()
            .filter(|(s, _, b)| *s == 200 && b.is_object())
            .map(|(_, _, b)| b.clone())
    };

    if let Some(b) = ok(&c.get("/props").await)
        && (b.get("chat_template").is_some()
            || b.get("build_info").is_some()
            || b.get("default_generation_settings").is_some())
    {
        return found(
            "llama.cpp",
            str_of(&b, "build_info"),
            str_of(&b, "chat_template"),
        );
    }
    if let Some(b) = ok(&c.get("/get_model_info").await)
        && b.get("model_path").is_some()
    {
        return found("sglang", str_of(&b, "version"), None);
    }
    if let Some(b) = ok(&c.get("/api/v0/models").await)
        && b.get("data").is_some()
    {
        return found("lmstudio", None, None);
    }
    if let Some(b) = ok(&c.get("/api/version").await)
        && b.get("version").is_some()
    {
        return found("ollama", str_of(&b, "version"), None);
    }
    if let Some(b) = ok(&c.get("/v1/model").await)
        && (b["owned_by"] == "tabbyAPI" || b.get("parameters").is_some())
    {
        return found("tabbyapi", None, None);
    }
    if let Some(b) = ok(&c.get("/version").await)
        && let Some(v) = str_of(&b, "version")
    {
        return found("vllm", Some(v), None);
    }
    let owned = models_body
        .as_ref()
        .and_then(|(_, _, b)| b["data"][0]["owned_by"].as_str().map(str::to_owned))
        .unwrap_or_default();
    let runtime = match owned.as_str() {
        "llamacpp" => "llama.cpp",
        "vllm" => "vllm",
        "library" => "ollama",
        "sglang" => "sglang",
        "tabbyAPI" => "tabbyapi",
        _ => "unknown",
    };
    found(runtime, None, None)
}
