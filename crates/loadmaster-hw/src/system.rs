//! Live detection. Missing drivers or tools mean "no GPU of that kind", never an error.

use std::process::Command;

use crate::{Error, GIB, Gpu, Host, HwProvider, HwSnapshot, MIB, MemoryKind, Vendor};

pub struct SystemHw;

impl HwProvider for SystemHw {
    fn snapshot(&self) -> Result<HwSnapshot, Error> {
        let host = host()?;
        let mut gpus = nvidia();
        if let Some(apple) = apple(&host) {
            gpus.push(apple);
        }
        Ok(HwSnapshot { gpus, host })
    }
}

/// Metal's default `recommendedMaxWorkingSetSize` on Apple Silicon is 2/3 of RAM up to
/// 32 GiB and 3/4 above; `iogpu.wired_limit_mb`, when set, replaces it.
pub(crate) fn apple_working_set(total: u64, wired_limit_mb: u64) -> (u64, String) {
    if wired_limit_mb > 0 {
        return (
            wired_limit_mb * MIB,
            format!("iogpu.wired_limit_mb = {wired_limit_mb}"),
        );
    }
    if total <= 32 * GIB {
        (
            total * 2 / 3,
            "Metal working-set default, 2/3 of RAM (estimated)".into(),
        )
    } else {
        (
            total * 3 / 4,
            "Metal working-set default, 3/4 of RAM (estimated)".into(),
        )
    }
}

fn sysctl(name: &str) -> Option<String> {
    let out = Command::new("sysctl").args(["-n", name]).output().ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_owned())
}

fn apple(host: &Host) -> Option<Gpu> {
    if !(cfg!(target_os = "macos") && cfg!(target_arch = "aarch64")) {
        return None;
    }
    let name = sysctl("machdep.cpu.brand_string").unwrap_or_else(|| "Apple Silicon".into());
    let wired = sysctl("iogpu.wired_limit_mb")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let (usable_bytes, usable_basis) = apple_working_set(host.total_bytes, wired);
    Some(Gpu {
        index: 0,
        name,
        vendor: Vendor::Apple,
        memory: MemoryKind::Unified,
        total_bytes: host.total_bytes,
        usable_bytes,
        usable_basis,
    })
}

#[cfg(target_os = "macos")]
fn host() -> Result<Host, Error> {
    let err = |reason: &str| Error::Host {
        source_name: "sysctl/vm_stat",
        reason: reason.into(),
    };
    let total_bytes: u64 = sysctl("hw.memsize")
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| err("hw.memsize unavailable"))?;
    // `kern.memorystatus_level` is the kernel's own "memory free" percentage (what
    // `memory_pressure` prints and what jetsam acts on): it counts memory the OS can
    // reclaim by compressing or paging, which vm_stat's free + inactive does not.
    let (available_bytes, available_basis) = match sysctl("kern.memorystatus_level")
        .and_then(|s| s.parse::<u64>().ok())
        .filter(|&l| l <= 100)
    {
        Some(level) => (
            total_bytes / 100 * level,
            format!("kern.memorystatus_level {level}% of RAM"),
        ),
        None => (
            vm_stat_available().unwrap_or(0),
            "vm_stat free + inactive + speculative + purgeable".into(),
        ),
    };
    Ok(Host {
        total_bytes,
        available_bytes,
        available_basis,
    })
}

/// Free + inactive + speculative + purgeable pages, as Activity Monitor counts reclaimable.
#[cfg(target_os = "macos")]
fn vm_stat_available() -> Option<u64> {
    let out = Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let page: u64 = text
        .lines()
        .next()?
        .split("page size of ")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let pages = |label: &str| -> u64 {
        text.lines()
            .find(|l| l.starts_with(label))
            .and_then(|l| l.rsplit(':').next())
            .and_then(|v| v.trim().trim_end_matches('.').parse().ok())
            .unwrap_or(0)
    };
    Some(
        (pages("Pages free")
            + pages("Pages inactive")
            + pages("Pages speculative")
            + pages("Pages purgeable"))
            * page,
    )
}

#[cfg(not(target_os = "macos"))]
fn host() -> Result<Host, Error> {
    let err = |reason: String| Error::Host {
        source_name: "/proc/meminfo",
        reason,
    };
    let text = std::fs::read_to_string("/proc/meminfo").map_err(|e| err(e.to_string()))?;
    let kib = |label: &str| -> Option<u64> {
        text.lines()
            .find(|l| l.starts_with(label))?
            .split_whitespace()
            .nth(1)?
            .parse()
            .ok()
    };
    Ok(Host {
        total_bytes: kib("MemTotal:").ok_or_else(|| err("no MemTotal".into()))? * 1024,
        available_bytes: kib("MemAvailable:").unwrap_or(0) * 1024,
        available_basis: "/proc/meminfo MemAvailable".into(),
    })
}

#[cfg(feature = "cuda")]
fn nvidia() -> Vec<Gpu> {
    use nvml_wrapper::Nvml;
    let nvml = match Nvml::init() {
        Ok(n) => n,
        Err(e) => {
            tracing::debug!("NVML unavailable: {e}");
            return vec![];
        }
    };
    let count = nvml.device_count().unwrap_or(0);
    (0..count)
        .filter_map(|i| {
            let dev = nvml.device_by_index(i).ok()?;
            let mem = dev.memory_info().ok()?;
            Some(Gpu {
                index: i,
                name: dev.name().unwrap_or_else(|_| format!("NVIDIA GPU {i}")),
                vendor: Vendor::Nvidia,
                memory: MemoryKind::Dedicated,
                total_bytes: mem.total,
                usable_bytes: mem.free,
                usable_basis: "NVML free".into(),
            })
        })
        .collect()
}

#[cfg(not(feature = "cuda"))]
fn nvidia() -> Vec<Gpu> {
    vec![]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_working_set_matches_metal_defaults() {
        assert_eq!(apple_working_set(8 * GIB, 0).0, 8 * GIB * 2 / 3);
        assert_eq!(apple_working_set(64 * GIB, 0).0, 48 * GIB);
        assert_eq!(apple_working_set(64 * GIB, 57344).0, 56 * GIB);
    }
}
