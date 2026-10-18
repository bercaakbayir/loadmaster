//! Supervisor: process manager, footprint-based eviction, OpenAI-compatible proxy,
//! health checks and `/metrics`.

pub mod evict;
pub mod metrics;
pub mod process;
pub mod proxy;
pub mod supervisor;

use std::sync::Arc;
use std::time::Duration;

pub use proxy::Routes;
pub use supervisor::{ModelSpec, Options, State, Status, Supervisor};
use tokio::sync::Notify;

/// Serve until `/loadmaster/shutdown`, Ctrl-C or SIGTERM, then stop every model.
pub async fn serve(
    listener: tokio::net::TcpListener,
    sup: Arc<Supervisor>,
    routes: Routes,
    preload: Vec<String>,
) -> std::io::Result<()> {
    let shutdown = Arc::new(Notify::new());
    let app = proxy::router(Arc::clone(&sup), routes, Arc::clone(&shutdown));

    let reaper = {
        let sup = Arc::clone(&sup);
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(1)).await;
                sup.reap_idle().await;
            }
        })
    };
    for name in preload {
        let sup = Arc::clone(&sup);
        tokio::spawn(async move {
            if let Err(e) = sup.swap(&name).await {
                tracing::warn!("preloading {name}: {e}");
            }
        });
    }

    let signal = {
        let shutdown = Arc::clone(&shutdown);
        async move {
            tokio::select! {
                _ = shutdown.notified() => {}
                _ = tokio::signal::ctrl_c() => {}
                _ = terminate() => {}
            }
        }
    };
    let result = axum::serve(listener, app)
        .with_graceful_shutdown(signal)
        .await;
    reaper.abort();
    sup.shutdown().await;
    result
}

#[cfg(unix)]
async fn terminate() {
    match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
        Ok(mut term) => {
            term.recv().await;
        }
        Err(_) => std::future::pending().await,
    }
}

#[cfg(not(unix))]
async fn terminate() {
    std::future::pending::<()>().await
}
