//! `cargo xtask validate [--markdown]`: replay `validation/cases.toml` through the planner
//! and compare with ground truth.

use std::path::Path;

use anyhow::{Context, bail};
use loadmaster_hw::{Gpu, Host, HwSnapshot, MIB, MemoryKind, Vendor};
use loadmaster_model::ModelInfo;
use loadmaster_plan::{Error, KvType, PlanRequest, Runtime};
use serde::Deserialize;

#[derive(Deserialize)]
struct File {
    case: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    sha256: String,
    runtime: String,
    ctx: u32,
    kv: Option<String>,
    hw: Hw,
    truth: String,
    llmfit: Option<String>,
    llama_fit: Option<String>,
}

#[derive(Deserialize)]
struct Hw {
    name: String,
    vendor: String,
    memory: String,
    total_mib: u64,
    usable_mib: u64,
    host_available_mib: u64,
}

impl Hw {
    fn snapshot(&self) -> anyhow::Result<HwSnapshot> {
        let vendor = match self.vendor.as_str() {
            "nvidia" => Vendor::Nvidia,
            "apple" => Vendor::Apple,
            v => bail!("unknown vendor {v}"),
        };
        let memory = match self.memory.as_str() {
            "dedicated" => MemoryKind::Dedicated,
            "unified" => MemoryKind::Unified,
            m => bail!("unknown memory kind {m}"),
        };
        Ok(HwSnapshot {
            gpus: vec![Gpu {
                index: 0,
                name: self.name.clone(),
                vendor,
                memory,
                total_bytes: self.total_mib * MIB,
                usable_bytes: self.usable_mib * MIB,
                usable_basis: "validation case".into(),
                bandwidth_bytes_s: None,
                bandwidth_basis: String::new(),
            }],
            host: Host {
                total_bytes: self.total_mib * MIB,
                available_bytes: self.host_available_mib * MIB,
                available_basis: "validation case".into(),
            },
        })
    }
}

pub fn run(root: &Path, markdown: bool) -> anyhow::Result<()> {
    let text = std::fs::read_to_string(root.join("validation/cases.toml"))?;
    let file: File = toml::from_str(&text).context("validation/cases.toml")?;
    let mut rows = Vec::new();
    let (mut agree, mut total, mut missing) = (0, 0, 0);
    for c in &file.case {
        let header = root
            .join("validation/headers")
            .join(format!("{}.gguf", c.sha256));
        let Ok(model) = ModelInfo::open(&header) else {
            missing += 1;
            rows.push((c, "no header".to_owned(), None));
            continue;
        };
        let hw = c.hw.snapshot()?;
        let kv = match c.kv.as_deref() {
            None => None,
            Some("f16") => Some(KvType::F16),
            Some("q8_0") => Some(KvType::Q8_0),
            Some("q4_0") => Some(KvType::Q4_0),
            Some(k) => bail!("{}: unknown kv {k}", c.id),
        };
        let runtime = match c.runtime.as_str() {
            "llama.cpp" => Runtime::LlamaCpp,
            "vllm" => Runtime::Vllm,
            "exl3" => Runtime::Exl3,
            r => bail!("{}: unknown runtime {r}", c.id),
        };
        let req = PlanRequest {
            ctx: Some(c.ctx),
            kv,
            runtime,
            ..PlanRequest::default()
        };
        let predicted = match loadmaster_plan::plan(&model, &hw, &hw.gpus[0], &req) {
            Ok(_) => "fits",
            Err(Error::CannotFit(_)) => "oom",
            Err(e) => bail!("{}: {e}", c.id),
        };
        total += 1;
        let ok = predicted == c.truth;
        agree += usize::from(ok);
        rows.push((c, predicted.to_owned(), Some(ok)));
    }

    let mark = |ok: Option<bool>| match ok {
        Some(true) => "yes",
        Some(false) => "NO",
        None => "-",
    };
    let other = |p: &Option<String>, truth: &str| match p.as_deref() {
        None => "-".to_owned(),
        Some(p) => format!(
            "{p}{}",
            if p == truth || (p == "no" && truth == "oom") {
                ""
            } else {
                " (wrong)"
            }
        ),
    };
    if markdown {
        println!("| case | GPU | ctx | truth | loadmaster | llmfit | llama.cpp --fit |");
        println!("| --- | --- | ---: | --- | --- | --- | --- |");
        for (c, p, ok) in &rows {
            println!(
                "| {} | {} | {} | {} | {}{} | {} | {} |",
                c.id,
                c.hw.name,
                c.ctx,
                c.truth,
                p,
                if *ok == Some(false) { " (wrong)" } else { "" },
                other(&c.llmfit, &c.truth),
                other(&c.llama_fit, &c.truth),
            );
        }
        println!("\nloadmaster agrees with ground truth on {agree} of {total} cases.");
    } else {
        for (c, p, ok) in &rows {
            println!(
                "{:<40} {:<14} ctx {:>6}  truth {:<4}  predicted {:<9} agree {}",
                c.id,
                c.hw.name,
                c.ctx,
                c.truth,
                p,
                mark(*ok)
            );
        }
        println!(
            "\n{agree}/{total} agree{}",
            if missing > 0 {
                format!("; {missing} cases have no header, run `cargo xtask fetch-headers`")
            } else {
                String::new()
            }
        );
    }
    Ok(())
}
