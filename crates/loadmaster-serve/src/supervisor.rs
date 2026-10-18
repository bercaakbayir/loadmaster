//! Model lifecycle: load on demand, evict by footprint, restart on crash, unload when idle.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::{Mutex, Notify, watch};

use crate::evict::{self, Decision, Resident};
use crate::metrics::Metrics;
use crate::process::{self, LaunchSpec, NotReady};

#[derive(Debug, Clone)]
pub struct Options {
    /// Bytes models may use together (GPU budget minus headroom).
    pub budget: u64,
    pub load_timeout: Duration,
    /// Default idle unload; zero means never.
    pub idle_unload: Duration,
    pub max_restarts: u32,
    /// First restart delay; doubles each time.
    pub backoff: Duration,
    pub stop_grace: Duration,
}

#[derive(Debug, Clone)]
pub struct ModelSpec {
    pub name: String,
    pub launch: LaunchSpec,
    /// Bytes this model adds while resident (footprint minus the shared headroom).
    pub bytes: u64,
    pub keep_warm: bool,
    pub idle_unload: Option<Duration>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum State {
    Stopped,
    Loading,
    Ready,
    Stopping,
    Failed,
}

struct Slot {
    state: State,
    port: Option<u16>,
    pid: Option<u32>,
    restarts: u32,
    error: Option<String>,
    last_used: Instant,
    stop: Option<watch::Sender<bool>>,
}

struct Model {
    spec: ModelSpec,
    slot: std::sync::Mutex<Slot>,
    /// Woken on every state change.
    changed: Notify,
    in_flight: AtomicU32,
    lru: AtomicU64,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum Error {
    #[error("no model named `{0}`")]
    Unknown(String),
    #[error("`{name}` failed to start: {reason}; see {log}")]
    Failed {
        name: String,
        reason: String,
        log: String,
    },
    #[error("`{0}` is still loading")]
    Timeout(String),
    #[error(
        "`{name}` needs {need} bytes but only {available} can be freed (models in use are not unloaded)"
    )]
    CannotFit {
        name: String,
        need: u64,
        available: u64,
    },
}

#[derive(Debug, Clone, Serialize)]
pub struct Status {
    pub name: String,
    pub state: State,
    pub pid: Option<u32>,
    pub port: Option<u16>,
    pub resident_bytes: u64,
    pub restarts: u32,
    pub in_flight: u32,
    pub keep_warm: bool,
    pub error: Option<String>,
    pub log: String,
}

/// Held while a request uses a model; the model is not evicted while any are alive.
pub struct Lease {
    model: Arc<Supervisor>,
    idx: usize,
    pub port: u16,
}

impl Drop for Lease {
    fn drop(&mut self) {
        let m = &self.model.models[self.idx];
        m.in_flight.fetch_sub(1, Ordering::SeqCst);
        m.slot.lock().unwrap().last_used = Instant::now();
    }
}

pub struct Supervisor {
    opts: Options,
    models: Vec<Model>,
    /// One load/evict decision at a time.
    load_lock: Mutex<()>,
    tick: AtomicU64,
    http: reqwest::Client,
    pub metrics: Metrics,
}

impl Supervisor {
    pub fn new(opts: Options, specs: Vec<ModelSpec>) -> Arc<Self> {
        let names: Vec<String> = specs.iter().map(|s| s.name.clone()).collect();
        Arc::new(Supervisor {
            opts,
            models: specs
                .into_iter()
                .map(|spec| Model {
                    spec,
                    slot: std::sync::Mutex::new(Slot {
                        state: State::Stopped,
                        port: None,
                        pid: None,
                        restarts: 0,
                        error: None,
                        last_used: Instant::now(),
                        stop: None,
                    }),
                    changed: Notify::new(),
                    in_flight: AtomicU32::new(0),
                    lru: AtomicU64::new(0),
                })
                .collect(),
            load_lock: Mutex::new(()),
            tick: AtomicU64::new(1),
            http: reqwest::Client::new(),
            metrics: Metrics::new(&names),
        })
    }

    fn index(&self, name: &str) -> Result<usize, Error> {
        self.models
            .iter()
            .position(|m| m.spec.name == name)
            .ok_or_else(|| Error::Unknown(name.to_owned()))
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.models.iter().map(|m| m.spec.name.as_str())
    }

    fn set(&self, idx: usize, f: impl FnOnce(&mut Slot)) {
        f(&mut self.models[idx].slot.lock().unwrap());
        self.models[idx].changed.notify_waiters();
    }

    /// A lease on a ready model, loading it (and evicting others) if needed. Waits at most
    /// `load_timeout`; the load continues in the background after a timeout.
    pub async fn acquire(self: &Arc<Self>, name: &str) -> Result<Lease, Error> {
        let idx = self.index(name)?;
        let m = &self.models[idx];
        self.metrics.request(name);
        let deadline = tokio::time::Instant::now() + self.opts.load_timeout;
        loop {
            let notified = m.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            // Decide under the slot lock; await only after releasing it.
            let start = {
                let slot = m.slot.lock().unwrap();
                match slot.state {
                    State::Ready => {
                        m.in_flight.fetch_add(1, Ordering::SeqCst);
                        m.lru
                            .store(self.tick.fetch_add(1, Ordering::SeqCst), Ordering::SeqCst);
                        return Ok(Lease {
                            model: Arc::clone(self),
                            idx,
                            port: slot.port.expect("ready models have a port"),
                        });
                    }
                    State::Failed => {
                        return Err(Error::Failed {
                            name: name.to_owned(),
                            reason: slot.error.clone().unwrap_or_default(),
                            log: m.spec.launch.log.display().to_string(),
                        });
                    }
                    State::Loading | State::Stopping => false,
                    State::Stopped => true,
                }
            };
            if start {
                // Decide evictions and launch under the load lock; CannotFit surfaces here.
                self.start(idx).await?;
                continue;
            }
            if tokio::time::timeout_at(deadline, notified).await.is_err() {
                return Err(Error::Timeout(name.to_owned()));
            }
        }
    }

    /// Load `name` now (evicting as needed) and wait until it is ready or failed. A
    /// `failed` model gets a fresh set of restarts.
    pub async fn swap(self: &Arc<Self>, name: &str) -> Result<(), Error> {
        let idx = self.index(name)?;
        let state = self.models[idx].slot.lock().unwrap().state;
        if state == State::Failed {
            self.unload(name).await?;
        }
        drop(self.acquire(name).await?);
        Ok(())
    }

    /// Evict what the scheduler says, then launch the model's instance task.
    async fn start(self: &Arc<Self>, idx: usize) -> Result<(), Error> {
        let _g = self.load_lock.lock().await;
        let state = self.models[idx].slot.lock().unwrap().state;
        if state != State::Stopped {
            return Ok(()); // someone else started it meanwhile
        }
        let residents: Vec<Resident> = self
            .models
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != idx)
            .filter(|(_, m)| {
                matches!(
                    m.slot.lock().unwrap().state,
                    State::Ready | State::Loading | State::Stopping
                )
            })
            .map(|(_, m)| Resident {
                name: m.spec.name.clone(),
                bytes: m.spec.bytes,
                last_used: m.lru.load(Ordering::SeqCst),
                keep_warm: m.spec.keep_warm,
                busy: m.in_flight.load(Ordering::SeqCst) > 0
                    || m.slot.lock().unwrap().state == State::Loading,
            })
            .collect();
        let spec = &self.models[idx].spec;
        match evict::schedule(&residents, spec.bytes, self.opts.budget) {
            Decision::CannotFit { need, available } => {
                return Err(Error::CannotFit {
                    name: spec.name.clone(),
                    need,
                    available,
                });
            }
            Decision::Load { evict } => {
                for name in evict {
                    let i = self.index(&name).expect("residents are known models");
                    tracing::info!("evicting {name} to make room for {}", spec.name);
                    self.metrics.eviction(&name);
                    self.stop_idx(i).await;
                }
            }
        }
        let (tx, rx) = watch::channel(false);
        self.set(idx, |s| {
            s.state = State::Loading;
            s.error = None;
            s.stop = Some(tx);
        });
        tokio::spawn(Arc::clone(self).instance(idx, rx));
        Ok(())
    }

    /// One model's process lifecycle: start, wait ready, watch, restart, stop.
    async fn instance(self: Arc<Self>, idx: usize, mut stop: watch::Receiver<bool>) {
        let m = &self.models[idx];
        let name = m.spec.name.clone();
        loop {
            let started = Instant::now();
            let port = match process::free_port() {
                Ok(p) => p,
                Err(e) => return self.fail(idx, format!("no free port: {e}")),
            };
            let mut child = match process::spawn(&m.spec.launch, port) {
                Ok(c) => c,
                Err(e) => {
                    return self.fail(
                        idx,
                        format!("cannot start `{}`: {e}", m.spec.launch.argv[0]),
                    );
                }
            };
            self.set(idx, |s| s.pid = child.id());
            let ready = tokio::select! {
                r = process::wait_ready(&self.http, &mut child, port, &m.spec.launch.ready_path, self.opts.load_timeout) => r,
                _ = stop.changed() => {
                    process::stop(&mut child, self.opts.stop_grace).await;
                    return self.stopped(idx);
                }
            };
            let exit = match ready {
                Ok(()) => {
                    self.metrics.loaded(&name, started.elapsed(), m.spec.bytes);
                    tracing::info!(
                        "{name} ready on port {port} after {:.1}s",
                        started.elapsed().as_secs_f64()
                    );
                    self.set(idx, |s| {
                        s.state = State::Ready;
                        s.port = Some(port);
                        s.last_used = Instant::now();
                    });
                    tokio::select! {
                        status = child.wait() => format!("exited unexpectedly ({})", status.map_or("unknown".into(), |s| s.to_string())),
                        _ = stop.changed() => {
                            self.set(idx, |s| s.state = State::Stopping);
                            process::stop(&mut child, self.opts.stop_grace).await;
                            return self.stopped(idx);
                        }
                    }
                }
                Err(NotReady::Exited(code)) => format!("exited before ready (code {code:?})"),
                Err(NotReady::Timeout) => {
                    process::stop(&mut child, self.opts.stop_grace).await;
                    format!("not ready after {}s", self.opts.load_timeout.as_secs())
                }
            };
            let restarts = {
                let mut s = m.slot.lock().unwrap();
                s.port = None;
                if s.restarts < self.opts.max_restarts {
                    s.restarts += 1;
                    Some(s.restarts)
                } else {
                    None
                }
            };
            let Some(restarts) = restarts else {
                tracing::warn!("{name} {exit}; giving up");
                return self.fail(
                    idx,
                    format!("{exit}; gave up after {} restarts", self.opts.max_restarts),
                );
            };
            tracing::warn!(
                "{name} {exit}; restart {restarts}/{}",
                self.opts.max_restarts
            );
            self.set(idx, |s| s.state = State::Loading);
            let delay = self.opts.backoff * 2u32.pow(restarts.saturating_sub(1).min(6));
            tokio::select! {
                _ = tokio::time::sleep(delay) => {}
                _ = stop.changed() => return self.stopped(idx),
            }
        }
    }

    fn fail(&self, idx: usize, reason: String) {
        self.metrics.unloaded(&self.models[idx].spec.name);
        self.set(idx, |s| {
            s.state = State::Failed;
            s.error = Some(reason);
            s.port = None;
            s.pid = None;
            s.stop = None;
        });
    }

    fn stopped(&self, idx: usize) {
        self.metrics.unloaded(&self.models[idx].spec.name);
        self.set(idx, |s| {
            s.state = State::Stopped;
            s.port = None;
            s.pid = None;
            s.stop = None;
        });
    }

    async fn stop_idx(&self, idx: usize) {
        let m = &self.models[idx];
        loop {
            let notified = m.changed.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            let done = {
                let mut s = m.slot.lock().unwrap();
                match s.state {
                    State::Stopped => true,
                    State::Failed => {
                        // Clear the failure so the next request may try again.
                        s.state = State::Stopped;
                        s.restarts = 0;
                        true
                    }
                    _ => {
                        if let Some(tx) = &s.stop {
                            let _ = tx.send(true);
                        }
                        false
                    }
                }
            };
            if done {
                return;
            }
            notified.await;
        }
    }

    /// Unload a model (also clears a `failed` state).
    pub async fn unload(&self, name: &str) -> Result<(), Error> {
        let idx = self.index(name)?;
        let _g = self.load_lock.lock().await;
        self.stop_idx(idx).await;
        Ok(())
    }

    pub async fn shutdown(&self) {
        let _g = self.load_lock.lock().await;
        for i in 0..self.models.len() {
            self.stop_idx(i).await;
        }
    }

    /// Unload idle models; run periodically.
    pub async fn reap_idle(&self) {
        for (i, m) in self.models.iter().enumerate() {
            let idle = m.spec.idle_unload.unwrap_or(self.opts.idle_unload);
            let expired = {
                let s = m.slot.lock().unwrap();
                s.state == State::Ready
                    && !m.spec.keep_warm
                    && !idle.is_zero()
                    && m.in_flight.load(Ordering::SeqCst) == 0
                    && s.last_used.elapsed() >= idle
            };
            if expired {
                tracing::info!("unloading {} after {}s idle", m.spec.name, idle.as_secs());
                let _g = self.load_lock.lock().await;
                self.stop_idx(i).await;
            }
        }
    }

    pub fn status(&self) -> Vec<Status> {
        self.models
            .iter()
            .map(|m| {
                let s = m.slot.lock().unwrap();
                let resident = matches!(s.state, State::Ready | State::Loading | State::Stopping);
                Status {
                    name: m.spec.name.clone(),
                    state: s.state,
                    pid: s.pid,
                    port: s.port,
                    resident_bytes: if resident { m.spec.bytes } else { 0 },
                    restarts: s.restarts,
                    in_flight: m.in_flight.load(Ordering::SeqCst),
                    keep_warm: m.spec.keep_warm,
                    error: s.error.clone(),
                    log: m.spec.launch.log.display().to_string(),
                }
            })
            .collect()
    }
}
