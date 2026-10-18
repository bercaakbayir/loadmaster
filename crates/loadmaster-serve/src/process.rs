//! Runtime processes: spawn with logs, wait for readiness, stop gracefully.

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use tokio::process::{Child, Command};

/// How to start one model's runtime. The supervisor appends `--host 127.0.0.1 --port N`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchSpec {
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub log: PathBuf,
    /// Polled until it answers 200.
    pub ready_path: String,
}

/// A free loopback port (bound and released; the runtime binds it right after).
pub fn free_port() -> std::io::Result<u16> {
    let l = std::net::TcpListener::bind(("127.0.0.1", 0))?;
    Ok(l.local_addr()?.port())
}

pub fn spawn(spec: &LaunchSpec, port: u16) -> std::io::Result<Child> {
    if let Some(dir) = spec.log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let log = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&spec.log)?;
    let mut argv = spec.argv.clone();
    argv.extend([
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        port.to_string(),
    ]);
    {
        use std::io::Write;
        let mut l = &log;
        writeln!(l, "--- loadmaster: starting {}", argv.join(" "))?;
    }
    let mut cmd = Command::new(&argv[0]);
    cmd.args(&argv[1..])
        .envs(spec.env.iter().cloned())
        .stdin(Stdio::null())
        .stdout(log.try_clone()?)
        .stderr(log)
        .kill_on_drop(true);
    cmd.spawn()
}

#[derive(Debug, PartialEq, Eq)]
pub enum NotReady {
    Exited(Option<i32>),
    Timeout,
}

/// Poll `http://127.0.0.1:port{path}` until 200, the child exits, or `timeout`.
pub async fn wait_ready(
    http: &reqwest::Client,
    child: &mut Child,
    port: u16,
    path: &str,
    timeout: Duration,
) -> Result<(), NotReady> {
    let url = format!("http://127.0.0.1:{port}{path}");
    let deadline = tokio::time::Instant::now() + timeout;
    loop {
        if let Ok(Some(status)) = child.try_wait() {
            return Err(NotReady::Exited(status.code()));
        }
        if let Ok(r) = http.get(&url).timeout(Duration::from_secs(2)).send().await
            && r.status().is_success()
        {
            return Ok(());
        }
        if tokio::time::Instant::now() >= deadline {
            return Err(NotReady::Timeout);
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

/// SIGTERM, then SIGKILL after `grace`. On Windows there is no graceful signal for a
/// console child, so it is terminated directly.
pub async fn stop(child: &mut Child, grace: Duration) {
    #[cfg(unix)]
    if let Some(pid) = child.id() {
        let _ = nix::sys::signal::kill(
            nix::unistd::Pid::from_raw(pid as i32),
            nix::sys::signal::Signal::SIGTERM,
        );
        if tokio::time::timeout(grace, child.wait()).await.is_ok() {
            return;
        }
    }
    #[cfg(not(unix))]
    let _ = grace;
    let _ = child.kill().await;
}

/// The default log location for a model.
pub fn log_path(logs_dir: &Path, name: &str) -> PathBuf {
    logs_dir.join(format!("{name}.log"))
}
