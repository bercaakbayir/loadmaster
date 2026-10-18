//! What changed between two lockfiles, per model and field.

use crate::{Lockfile, Model};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    Added(String),
    Removed(String),
    Field {
        model: String,
        field: &'static str,
        from: String,
        to: String,
    },
    Host {
        field: &'static str,
        from: String,
        to: String,
    },
}

impl std::fmt::Display for Change {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Change::Added(m) => write!(f, "+ {m}"),
            Change::Removed(m) => write!(f, "- {m}"),
            Change::Field {
                model,
                field,
                from,
                to,
            } => write!(f, "~ {model}.{field}: {from} -> {to}"),
            Change::Host { field, from, to } => write!(f, "~ {field}: {from} -> {to}"),
        }
    }
}

fn fields(m: &Model) -> Vec<(&'static str, String)> {
    vec![
        ("source", m.source.clone()),
        ("sha256", format!("{:?}", m.sha256)),
        ("runtime", m.runtime.clone()),
        ("runtime_bin", m.runtime_bin.clone()),
        ("ctx", m.ctx.to_string()),
        ("footprint_gb", format!("{:?}", m.footprint_gb)),
        ("flags", m.flags.join(" ")),
        ("mmproj", format!("{:?}", m.mmproj)),
        ("env", m.env.join(" ")),
        ("keep_warm", m.keep_warm.to_string()),
        ("aliases", m.aliases.join(",")),
        ("idle_unload_s", format!("{:?}", m.idle_unload_s)),
        (
            "verify",
            format!("{:?}", m.verify.as_ref().map(|v| &v.verdict)),
        ),
    ]
}

pub fn diff(a: &Lockfile, b: &Lockfile) -> Vec<Change> {
    let mut out = Vec::new();
    let host = |l: &Lockfile| {
        vec![
            ("host.gpu", l.host.gpu.clone()),
            ("host.vram_gb", l.host.vram_gb.to_string()),
            ("host.headroom_gb", l.host.headroom_gb.to_string()),
            ("proxy.listen", l.proxy.listen.clone()),
            ("proxy.idle_unload_s", l.proxy.idle_unload_s.to_string()),
            ("proxy.load_timeout_s", l.proxy.load_timeout_s.to_string()),
        ]
    };
    for ((field, from), (_, to)) in host(a).into_iter().zip(host(b)) {
        if from != to {
            out.push(Change::Host { field, from, to });
        }
    }
    for m in &a.models {
        match b.model(&m.name) {
            None => out.push(Change::Removed(m.name.clone())),
            Some(n) => {
                for ((field, from), (_, to)) in fields(m).into_iter().zip(fields(n)) {
                    if from != to {
                        out.push(Change::Field {
                            model: m.name.clone(),
                            field,
                            from,
                            to,
                        });
                    }
                }
            }
        }
    }
    for n in &b.models {
        if a.model(&n.name).is_none() {
            out.push(Change::Added(n.name.clone()));
        }
    }
    out
}
