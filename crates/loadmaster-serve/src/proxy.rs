//! The single-port OpenAI-compatible proxy, plus admin and metrics endpoints.

use std::collections::HashMap;
use std::sync::Arc;

use axum::Router;
use axum::body::{Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{HeaderMap, HeaderValue, Method, StatusCode, Uri, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use futures_util::StreamExt;
use serde_json::{Value, json};
use tokio::sync::Notify;

use crate::supervisor::{Error, Supervisor};

/// Requests carry images as base64; allow large bodies.
const MAX_BODY: usize = 64 << 20;
/// Hop-by-hop headers are not forwarded (RFC 9110 §7.6.1), nor is the length we may change.
const HOP: &[&str] = &[
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
];

#[derive(Clone)]
pub struct Routes {
    /// Name or alias -> model name.
    pub by_name: HashMap<String, String>,
    /// Model names in lockfile order, for `/v1/models`.
    pub listed: Vec<String>,
}

#[derive(Clone)]
struct App {
    sup: Arc<Supervisor>,
    routes: Arc<Routes>,
    http: reqwest::Client,
    shutdown: Arc<Notify>,
}

pub fn router(sup: Arc<Supervisor>, routes: Routes, shutdown: Arc<Notify>) -> Router {
    let app = App {
        sup,
        routes: Arc::new(routes),
        http: reqwest::Client::builder().build().expect("default client"),
        shutdown,
    };
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(models))
        .route("/metrics", get(metrics))
        .route("/loadmaster/status", get(status))
        .route("/loadmaster/swap/{name}", post(swap))
        .route("/loadmaster/unload/{name}", post(unload))
        .route("/loadmaster/shutdown", post(shutdown_now))
        .fallback(forward)
        .layer(axum::extract::DefaultBodyLimit::max(MAX_BODY))
        .with_state(app)
}

fn error(status: StatusCode, message: impl Into<String>, code: &str) -> Response {
    let body =
        json!({ "error": { "message": message.into(), "type": "loadmaster_error", "code": code } });
    (status, axum::Json(body)).into_response()
}

fn sup_error(e: Error) -> Response {
    match &e {
        Error::Unknown(_) => error(StatusCode::NOT_FOUND, e.to_string(), "model_not_found"),
        Error::Timeout(_) => {
            let mut r = error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!("{e}; retry shortly"),
                "model_loading",
            );
            r.headers_mut()
                .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
            r
        }
        Error::CannotFit { .. } => error(
            StatusCode::SERVICE_UNAVAILABLE,
            e.to_string(),
            "insufficient_memory",
        ),
        Error::Failed { .. } => error(StatusCode::BAD_GATEWAY, e.to_string(), "model_failed"),
    }
}

async fn models(State(app): State<App>) -> Response {
    let status: HashMap<String, _> = app
        .sup
        .status()
        .into_iter()
        .map(|s| (s.name.clone(), s))
        .collect();
    let data: Vec<Value> = app
        .routes
        .listed
        .iter()
        .map(|n| {
            json!({
                "id": n,
                "object": "model",
                "owned_by": "loadmaster",
                "status": status.get(n).map(|s| s.state),
            })
        })
        .collect();
    axum::Json(json!({ "object": "list", "data": data })).into_response()
}

async fn metrics(State(app): State<App>) -> Response {
    (
        [(header::CONTENT_TYPE, "text/plain; version=0.0.4")],
        app.sup.metrics.render(),
    )
        .into_response()
}

async fn status(State(app): State<App>) -> Response {
    axum::Json(app.sup.status()).into_response()
}

async fn swap(State(app): State<App>, Path(name): Path<String>) -> Response {
    let Some(target) = app.routes.by_name.get(&name) else {
        return sup_error(Error::Unknown(name));
    };
    match app.sup.swap(target).await {
        Ok(()) => axum::Json(json!({ "ready": target })).into_response(),
        Err(e) => sup_error(e),
    }
}

async fn unload(State(app): State<App>, Path(name): Path<String>) -> Response {
    let Some(target) = app.routes.by_name.get(&name) else {
        return sup_error(Error::Unknown(name));
    };
    match app.sup.unload(target).await {
        Ok(()) => axum::Json(json!({ "unloaded": target })).into_response(),
        Err(e) => sup_error(e),
    }
}

async fn shutdown_now(State(app): State<App>) -> Response {
    app.shutdown.notify_one();
    axum::Json(json!({ "shutting_down": true })).into_response()
}

/// Route by the JSON body's `model`, load if needed, and stream the backend's response
/// back unchanged.
async fn forward(
    State(app): State<App>,
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    let mut json: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) if body.is_empty() => {
            return error(
                StatusCode::BAD_REQUEST,
                "request has no JSON body naming a `model`",
                "missing_model",
            );
        }
        Err(e) => {
            return error(
                StatusCode::BAD_REQUEST,
                format!("body is not JSON: {e}"),
                "invalid_json",
            );
        }
    };
    let Some(requested) = json.get("model").and_then(Value::as_str).map(str::to_owned) else {
        return error(
            StatusCode::BAD_REQUEST,
            "request has no `model` field",
            "missing_model",
        );
    };
    let Some(name) = app.routes.by_name.get(&requested).cloned() else {
        return error(
            StatusCode::NOT_FOUND,
            format!("no model or alias `{requested}` in models.lock"),
            "model_not_found",
        );
    };
    let lease = match app.sup.acquire(&name).await {
        Ok(l) => l,
        Err(e) => return sup_error(e),
    };
    // Backends see the lockfile name (vLLM only answers to its served name).
    json["model"] = Value::String(name);
    let body = serde_json::to_vec(&json).expect("re-serialising parsed JSON");

    let path = uri.path_and_query().map_or("/", |p| p.as_str());
    let url = format!("http://127.0.0.1:{}{path}", lease.port);
    let mut req = app.http.request(method, &url).body(body);
    for (k, v) in &headers {
        if !HOP.contains(&k.as_str()) {
            req = req.header(k, v);
        }
    }
    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            return error(
                StatusCode::BAD_GATEWAY,
                format!("backend unreachable: {e}"),
                "backend_error",
            );
        }
    };
    let mut out = Response::builder().status(resp.status().as_u16());
    for (k, v) in resp.headers() {
        if !HOP.contains(&k.as_str()) {
            out = out.header(k, v);
        }
    }
    // The lease lives as long as the body stream, so the model is not evicted mid-stream.
    let stream = resp.bytes_stream().map(move |chunk| {
        let _held = &lease;
        chunk
    });
    out.body(Body::from_stream(stream)).unwrap_or_else(|e| {
        error(
            StatusCode::BAD_GATEWAY,
            format!("bad backend response: {e}"),
            "backend_error",
        )
    })
}
