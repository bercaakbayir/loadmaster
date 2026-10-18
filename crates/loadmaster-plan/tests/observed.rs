//! Replays `validation/observed.toml` (buffer sizes llama.cpp reported for real runs)
//! through the footprint arithmetic. Needs the model files, so it is ignored by default:
//! `cargo test -p loadmaster-plan --test observed -- --ignored --nocapture`.

use std::path::{Path, PathBuf};

use loadmaster_model::ModelInfo;
use loadmaster_plan::footprint::{self, Config, Headroom, KvType, Placement};
use serde::Deserialize;

#[derive(Deserialize)]
struct File {
    model: Vec<Model>,
}

#[derive(Deserialize)]
struct Model {
    name: String,
    paths: Vec<String>,
    run: Vec<Run>,
}

#[derive(Deserialize)]
struct Run {
    args: Args,
    gpu: Device,
    cpu: Device,
}

#[derive(Deserialize)]
struct Args {
    ctx: u32,
    ubatch: u32,
    flash_attn: bool,
    ngl: u32,
    kv: String,
}

#[derive(Deserialize)]
struct Device {
    model: f64,
    kv: f64,
    compute: f64,
}

const MIB: f64 = (1 << 20) as f64;

fn expand(p: &str) -> PathBuf {
    match p.strip_prefix("~/") {
        Some(rest) => Path::new(&std::env::var("HOME").unwrap_or_default()).join(rest),
        None => PathBuf::from(p),
    }
}

fn mib(bytes: u64) -> f64 {
    bytes as f64 / MIB
}

#[test]
#[ignore = "needs the model files listed in validation/observed.toml"]
fn planner_matches_observed_buffers() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../validation/observed.toml");
    let file: File = toml::from_str(&std::fs::read_to_string(root).unwrap()).unwrap();
    let mut failures = Vec::new();
    for m in &file.model {
        let Some(path) = m.paths.iter().map(|p| expand(p)).find(|p| p.exists()) else {
            eprintln!("skip {}: not on disk", m.name);
            continue;
        };
        let info = ModelInfo::from_gguf(&path).unwrap();
        eprintln!("{}", m.name);
        eprintln!(
            "  {:>6} {:>5} {:>3} {:>4} {:>4} | {:>17} {:>17} {:>15} {:>15} | {:>17}",
            "ctx",
            "ub",
            "fa",
            "ngl",
            "kv",
            "gpu model",
            "cpu model",
            "gpu kv",
            "cpu kv",
            "gpu compute"
        );
        for r in &m.run {
            let a = &r.args;
            let kv = match a.kv.as_str() {
                "f16" => KvType::F16,
                "q8_0" => KvType::Q8_0,
                "q4_0" => KvType::Q4_0,
                k => panic!("unknown kv type {k}"),
            };
            let cfg = Config {
                ctx: a.ctx,
                n_seq: 1,
                n_ubatch: a.ubatch,
                kv,
                flash_attn: a.flash_attn,
                headroom: Headroom {
                    bytes: 0,
                    basis: String::new(),
                },
            };
            let placement = Placement {
                n_gpu_layers: a.ngl,
                n_cpu_moe: 0,
            };
            let fp = footprint::footprint(&info, placement, &cfg).unwrap();
            let get = |terms: &[footprint::Term], key: &str| {
                mib(terms.iter().find(|t| t.key == key).map_or(0, |t| t.bytes))
            };
            let planned = [
                get(&fp.gpu, "weights"),
                get(&fp.host, "weights_cpu"),
                get(&fp.gpu, "kv"),
                get(&fp.host, "kv_cpu"),
                get(&fp.gpu, "compute"),
            ];
            let seen = [r.gpu.model, r.cpu.model, r.gpu.kv, r.cpu.kv, r.gpu.compute];
            let cell = |i: usize| {
                format!(
                    "{:7.2}/{:<7.2}{}",
                    planned[i],
                    seen[i],
                    if (planned[i] - seen[i]).abs() <= 1.0 {
                        " "
                    } else {
                        "!"
                    }
                )
            };
            eprintln!(
                "  {:>6} {:>5} {:>3} {:>4} {:>4} | {} {} {} {} | {}",
                a.ctx,
                a.ubatch,
                if a.flash_attn { "on" } else { "off" },
                a.ngl,
                a.kv,
                cell(0),
                cell(1),
                cell(2),
                cell(3),
                cell(4)
            );
            // Weights and KV must match within 1 MiB; compute must not be underestimated.
            for (i, what) in ["gpu model", "cpu model", "gpu kv", "cpu kv"]
                .iter()
                .enumerate()
            {
                if (planned[i] - seen[i]).abs() > 1.0 {
                    failures.push(format!(
                        "{} ctx {} ngl {}: {what} planned {:.2} observed {:.2}",
                        m.name, a.ctx, a.ngl, planned[i], seen[i]
                    ));
                }
            }
            if planned[4] + 0.5 < seen[4] {
                failures.push(format!(
                    "{} ctx {} ub {}: gpu compute underestimated, planned {:.2} observed {:.2}",
                    m.name, a.ctx, a.ubatch, planned[4], seen[4]
                ));
            }
        }
    }
    assert!(failures.is_empty(), "\n{}", failures.join("\n"));
}
