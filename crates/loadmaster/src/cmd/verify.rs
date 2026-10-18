//! `up --verify`: probe each model through the proxy once it is ready and record the
//! verdict in models.lock.

use std::path::PathBuf;
use std::sync::Arc;

use loadmaster_lock::{Lockfile, Verify};
use loadmaster_probe::{Options, fixture, known};
use loadmaster_serve::Supervisor;

/// One model to verify: its name and the chat-template hash of its model file.
pub struct Target {
    pub name: String,
    pub template_blake3: Option<String>,
    pub quant: String,
}

/// The runtime and version a verdict applies to, e.g. `llama.cpp b6700-abc1234`.
pub fn runtime_label(d: &loadmaster_probe::detect::Detected) -> String {
    match &d.version {
        Some(v) => format!("{} {v}", d.runtime),
        None => d.runtime.clone(),
    }
}

pub async fn run(sup: Arc<Supervisor>, targets: Vec<Target>, listen: String, lock_path: PathBuf) {
    let c = super::probe::client(&format!("http://{listen}"), None);
    let fixtures = fixture::builtin();
    let issues = known::builtin();
    for t in targets {
        if let Err(e) = sup.swap(&t.name).await {
            eprintln!("verify {}: {e}", t.name);
            continue;
        }
        let opts = Options {
            model: t.name.clone(),
            expect_template: t.template_blake3.clone(),
            capture_dir: Some(super::probe::capture_dir(&t.name)),
            quant: Some(t.quant.clone()),
            ..Options::default()
        };
        let report = loadmaster_probe::run::probe(&c, &fixtures, &opts, &issues).await;
        // The proxy hides the backend; detect the runtime on the backend port itself.
        let label = match sup
            .status()
            .into_iter()
            .find(|s| s.name == t.name)
            .and_then(|s| s.port)
        {
            Some(port) => {
                let backend = super::probe::client(&format!("http://127.0.0.1:{port}"), None);
                runtime_label(&loadmaster_probe::detect::detect(&backend).await)
            }
            None => runtime_label(&report.detected),
        };
        eprintln!(
            "verify {}: {} ({label}){}",
            t.name,
            report.verdict,
            if report.failures.is_empty() {
                String::new()
            } else {
                format!(": {}", report.failures.join(", "))
            }
        );
        let verify = Verify {
            probe_version: report.probe_version.clone(),
            verdict: report.verdict.clone(),
            failures: report.failures.clone(),
            runtime_version: Some(label),
            checked_at: loadmaster_probe::time::now(),
        };
        // Re-read so edits made meanwhile are kept; write only this entry's verdict.
        match Lockfile::load(&lock_path) {
            Ok(mut lock) => {
                if let Some(m) = lock.models.iter_mut().find(|m| m.name == t.name) {
                    m.verify = Some(verify);
                }
                if let Err(e) = lock.save(&lock_path, true) {
                    eprintln!("verify {}: writing {}: {e}", t.name, lock_path.display());
                }
            }
            Err(e) => eprintln!("verify {}: {e}", t.name),
        }
    }
    eprintln!("verify: done");
}
