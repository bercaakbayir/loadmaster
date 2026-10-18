//! `/metrics` in the Prometheus text exposition format.

use std::fmt::Write;
use std::sync::Mutex;
use std::time::Duration;

#[derive(Default, Clone)]
struct Counters {
    loads: u64,
    evictions: u64,
    requests: u64,
    load_seconds_sum: f64,
    resident_bytes: u64,
}

pub struct Metrics {
    by_model: Mutex<Vec<(String, Counters)>>,
}

impl Metrics {
    pub fn new(names: &[String]) -> Self {
        Metrics {
            by_model: Mutex::new(
                names
                    .iter()
                    .map(|n| (n.clone(), Counters::default()))
                    .collect(),
            ),
        }
    }

    fn with(&self, name: &str, f: impl FnOnce(&mut Counters)) {
        if let Some((_, c)) = self
            .by_model
            .lock()
            .unwrap()
            .iter_mut()
            .find(|(n, _)| n == name)
        {
            f(c);
        }
    }

    pub fn request(&self, name: &str) {
        self.with(name, |c| c.requests += 1);
    }

    pub fn loaded(&self, name: &str, took: Duration, bytes: u64) {
        self.with(name, |c| {
            c.loads += 1;
            c.load_seconds_sum += took.as_secs_f64();
            c.resident_bytes = bytes;
        });
    }

    pub fn unloaded(&self, name: &str) {
        self.with(name, |c| c.resident_bytes = 0);
    }

    pub fn eviction(&self, name: &str) {
        self.with(name, |c| c.evictions += 1);
    }

    pub fn render(&self) -> String {
        let models = self.by_model.lock().unwrap().clone();
        let mut o = String::new();
        let mut family =
            |name: &str, kind: &str, help: &str, value: &dyn Fn(&Counters) -> String| {
                writeln!(o, "# HELP {name} {help}").unwrap();
                writeln!(o, "# TYPE {name} {kind}").unwrap();
                for (m, c) in &models {
                    writeln!(o, "{name}{{model=\"{}\"}} {}", escape(m), value(c)).unwrap();
                }
            };
        family(
            "loadmaster_model_loads_total",
            "counter",
            "Times the model finished loading.",
            &|c| c.loads.to_string(),
        );
        family(
            "loadmaster_model_evictions_total",
            "counter",
            "Times the model was unloaded to make room.",
            &|c| c.evictions.to_string(),
        );
        family(
            "loadmaster_model_requests_total",
            "counter",
            "Requests routed to the model.",
            &|c| c.requests.to_string(),
        );
        family(
            "loadmaster_model_load_seconds_sum",
            "counter",
            "Total seconds spent loading the model.",
            &|c| format!("{:.3}", c.load_seconds_sum),
        );
        family(
            "loadmaster_model_load_seconds_count",
            "counter",
            "Number of completed loads (with _sum, mean load time).",
            &|c| c.loads.to_string(),
        );
        family(
            "loadmaster_model_resident_bytes",
            "gauge",
            "Planned footprint of the model while resident, else 0.",
            &|c| c.resident_bytes.to_string(),
        );
        o
    }
}

fn escape(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}
