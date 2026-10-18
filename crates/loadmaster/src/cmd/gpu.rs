use std::path::Path;
use std::process::ExitCode;

use loadmaster_hw::MemoryKind;
use loadmaster_plan::gib;

pub fn run(json: bool, hw_file: Option<&Path>) -> anyhow::Result<ExitCode> {
    let snap = super::snapshot(hw_file)?;
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
