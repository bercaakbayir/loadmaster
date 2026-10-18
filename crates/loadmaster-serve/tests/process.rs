#![cfg(unix)]
//! Supervisor behaviour with real (tiny) processes; no model runtimes involved.

use std::time::Duration;

use loadmaster_serve::process::LaunchSpec;
use loadmaster_serve::supervisor::Error;
use loadmaster_serve::{ModelSpec, Options, State, Supervisor};

fn opts() -> Options {
    Options {
        budget: 100,
        load_timeout: Duration::from_secs(10),
        idle_unload: Duration::ZERO,
        max_restarts: 3,
        backoff: Duration::from_millis(10),
        stop_grace: Duration::from_secs(2),
    }
}

fn spec(dir: &tempfile::TempDir, name: &str, script: &str, bytes: u64) -> ModelSpec {
    ModelSpec {
        name: name.into(),
        launch: LaunchSpec {
            argv: vec!["sh".into(), "-c".into(), script.into()],
            env: vec![],
            log: dir.path().join(format!("{name}.log")),
            ready_path: "/health".into(),
        },
        bytes,
        keep_warm: false,
        idle_unload: None,
    }
}

#[tokio::test]
async fn crash_loop_stops_after_three_restarts() {
    let dir = tempfile::tempdir().unwrap();
    let sup = Supervisor::new(opts(), vec![spec(&dir, "crashy", "echo boom; exit 1", 10)]);
    let err = sup.acquire("crashy").await.err().unwrap();
    assert!(
        matches!(&err, Error::Failed { reason, .. } if reason.contains("gave up after 3 restarts")),
        "{err}"
    );
    let st = &sup.status()[0];
    assert_eq!((st.state, st.restarts), (State::Failed, 3));
    let log = std::fs::read_to_string(dir.path().join("crashy.log")).unwrap();
    assert_eq!(
        log.matches("--- loadmaster: starting").count(),
        4,
        "first start + 3 restarts"
    );
    assert_eq!(log.lines().filter(|l| *l == "boom").count(), 4);
    assert!(
        err.to_string().contains("crashy.log"),
        "error names the log: {err}"
    );
}

#[tokio::test]
async fn too_big_is_cannot_fit() {
    let dir = tempfile::tempdir().unwrap();
    let sup = Supervisor::new(opts(), vec![spec(&dir, "huge", "sleep 60", 101)]);
    assert!(matches!(
        sup.acquire("huge").await,
        Err(Error::CannotFit { need: 101, .. })
    ));
    assert_eq!(sup.status()[0].state, State::Stopped);
}

#[tokio::test]
async fn load_timeout_while_not_ready() {
    let dir = tempfile::tempdir().unwrap();
    let mut o = opts();
    o.load_timeout = Duration::from_millis(300);
    o.max_restarts = 0;
    // Never answers /health.
    let sup = Supervisor::new(o, vec![spec(&dir, "slow", "sleep 60", 10)]);
    assert!(matches!(sup.acquire("slow").await, Err(Error::Timeout(_))));
    sup.shutdown().await;
    assert!(matches!(
        sup.status()[0].state,
        State::Stopped | State::Failed
    ));
}

#[test]
fn metrics_render_prometheus_text() {
    let m = loadmaster_serve::metrics::Metrics::new(&["a".into()]);
    m.request("a");
    m.loaded("a", Duration::from_millis(1500), 42);
    let text = m.render();
    assert!(text.contains("# TYPE loadmaster_model_loads_total counter"));
    assert!(text.contains("loadmaster_model_requests_total{model=\"a\"} 1"));
    assert!(text.contains("loadmaster_model_resident_bytes{model=\"a\"} 42"));
    assert!(text.contains("loadmaster_model_load_seconds_sum{model=\"a\"} 1.500"));
}
