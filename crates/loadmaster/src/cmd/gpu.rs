use std::process::ExitCode;

use loadmaster_hw::MemoryKind;
use loadmaster_hw::catalog::{CARDS, CHIPS};
use loadmaster_plan::gib;

pub fn run(json: bool, hw: &crate::HwArgs) -> anyhow::Result<ExitCode> {
    if hw.pretend.as_deref() == Some("list") {
        print_catalog();
        return Ok(ExitCode::SUCCESS);
    }
    let snap = super::snapshot(hw)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&snap)?);
        return Ok(ExitCode::SUCCESS);
    }
    if snap.gpus.is_empty() {
        println!("no GPU detected (NVIDIA needs a driver with NVML; Apple needs Apple Silicon)");
    }
    for g in &snap.gpus {
        let kind = match g.memory {
            MemoryKind::Dedicated => "dedicated",
            MemoryKind::Unified => "unified",
        };
        println!(
            "gpu {}  {}  {} {}, {} usable ({})",
            g.index,
            g.name,
            gib(g.total_bytes),
            kind,
            gib(g.usable_bytes),
            g.usable_basis
        );
    }
    println!(
        "host   {} RAM, {} available",
        gib(snap.host.total_bytes),
        gib(snap.host.available_bytes)
    );
    Ok(ExitCode::SUCCESS)
}

fn print_catalog() {
    println!("--pretend accepts any of these cards (VRAM as nvidia-smi reports it):");
    for c in CARDS {
        println!("  {:<36} {}", c.name, gib(c.vram_mib << 20));
    }
    let chips: Vec<&str> = CHIPS.iter().map(|(_, name)| *name).collect();
    println!(
        "a Mac with its RAM, e.g. `m4 pro 48gb`: {}",
        chips.join(", ")
    );
    println!(
        "or a size: `24GB` (dedicated), `48GB-unified` (Apple-like); add `+64GB` for host RAM"
    );
}
