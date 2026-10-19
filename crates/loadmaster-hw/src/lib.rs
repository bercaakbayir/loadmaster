//! GPU and host memory detection.
//!
//! Everything downstream takes a [`HwSnapshot`] value, never a live provider, so planning
//! is deterministic and testable with [`FakeHw`].

pub mod catalog;
mod system;

use serde::{Deserialize, Serialize};

pub use catalog::{PretendError, pretend};
pub use system::SystemHw;

pub const GIB: u64 = 1 << 30;
pub const MIB: u64 = 1 << 20;

/// `24 GiB`, `7.5 GiB`.
fn gib_str(bytes: u64) -> String {
    let g = bytes as f64 / GIB as f64;
    if (g - g.round()).abs() < 0.05 {
        format!("{g:.0} GiB")
    } else {
        format!("{g:.1} GiB")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Vendor {
    Nvidia,
    Apple,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MemoryKind {
    /// Discrete VRAM.
    Dedicated,
    /// Shared with the host (Apple Silicon).
    Unified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Gpu {
    pub index: u32,
    pub name: String,
    pub vendor: Vendor,
    pub memory: MemoryKind,
    pub total_bytes: u64,
    /// Bytes a model may use before headroom: NVML `free`, or the Metal working-set limit.
    pub usable_bytes: u64,
    /// Where `usable_bytes` came from, for `--explain`.
    pub usable_basis: String,
    /// Peak memory bandwidth in bytes per second, when known (for speed estimates).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bandwidth_bytes_s: Option<u64>,
    /// Where `bandwidth_bytes_s` came from.
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub bandwidth_basis: String,
}

impl Gpu {
    /// Memory held by other processes (dedicated GPUs only).
    pub fn in_use_bytes(&self) -> u64 {
        match self.memory {
            MemoryKind::Dedicated => self.total_bytes.saturating_sub(self.usable_bytes),
            MemoryKind::Unified => 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Host {
    pub total_bytes: u64,
    pub available_bytes: u64,
    /// Where `available_bytes` came from.
    #[serde(default)]
    pub available_basis: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HwSnapshot {
    pub gpus: Vec<Gpu>,
    pub host: Host,
}

impl HwSnapshot {
    /// The GPU with the most usable memory; ties go to the lowest index.
    pub fn best_gpu(&self) -> Option<&Gpu> {
        self.gpus
            .iter()
            .max_by_key(|g| (g.usable_bytes, std::cmp::Reverse(g.index)))
    }

    /// Select a GPU by index or by a case-insensitive substring of its name.
    pub fn find_gpu(&self, query: &str) -> Option<&Gpu> {
        if let Some(g) = query
            .parse::<u32>()
            .ok()
            .and_then(|i| self.gpus.iter().find(|g| g.index == i))
        {
            return Some(g);
        }
        let q = query.to_ascii_lowercase();
        self.gpus
            .iter()
            .find(|g| g.name.to_ascii_lowercase().contains(&q))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("reading host memory from {source_name}: {reason}")]
    Host {
        source_name: &'static str,
        reason: String,
    },
}

pub trait HwProvider {
    fn snapshot(&self) -> Result<HwSnapshot, Error>;
}

/// A fixed snapshot, for tests and for `--hw-file` replays.
#[derive(Debug, Clone)]
pub struct FakeHw(pub HwSnapshot);

impl HwProvider for FakeHw {
    fn snapshot(&self) -> Result<HwSnapshot, Error> {
        Ok(self.0.clone())
    }
}

impl FakeHw {
    fn nvidia(name: &str, total_bytes: u64, usable_bytes: u64, host_gib: u64) -> Self {
        FakeHw(HwSnapshot {
            gpus: vec![Gpu {
                index: 0,
                name: name.into(),
                vendor: Vendor::Nvidia,
                memory: MemoryKind::Dedicated,
                total_bytes,
                usable_bytes,
                usable_basis: "NVML free".into(),
                bandwidth_bytes_s: None,
                bandwidth_basis: String::new(),
            }],
            host: Host {
                total_bytes: host_gib * GIB,
                available_bytes: host_gib * GIB * 3 / 4,
                available_basis: "fake: 3/4 of RAM".into(),
            },
        })
    }

    fn apple(name: &str, ram_gib: u64) -> Self {
        let total = ram_gib * GIB;
        let (usable, basis) = system::apple_working_set(total, 0);
        FakeHw(HwSnapshot {
            gpus: vec![Gpu {
                index: 0,
                name: name.into(),
                vendor: Vendor::Apple,
                memory: MemoryKind::Unified,
                total_bytes: total,
                usable_bytes: usable,
                usable_basis: basis,
                bandwidth_bytes_s: None,
                bandwidth_basis: String::new(),
            }],
            host: Host {
                total_bytes: total,
                available_bytes: total / 2,
                available_basis: "fake: 1/2 of RAM".into(),
            },
        })
    }

    /// RTX 4090 with a desktop session holding ~500 MiB.
    pub fn rtx_4090() -> Self {
        Self::nvidia(
            "NVIDIA GeForce RTX 4090",
            25_757_220_864,
            25_232_932_864,
            64,
        )
    }

    /// RTX 3060 12 GB, headless.
    pub fn rtx_3060_12g() -> Self {
        Self::nvidia(
            "NVIDIA GeForce RTX 3060",
            12_884_901_888,
            12_610_174_976,
            32,
        )
    }

    pub fn m2_8g() -> Self {
        Self::apple("Apple M2", 8)
    }

    pub fn m3_max_64g() -> Self {
        Self::apple("Apple M3 Max", 64)
    }

    pub fn cpu_only() -> Self {
        FakeHw(HwSnapshot {
            gpus: vec![],
            host: Host {
                total_bytes: 32 * GIB,
                available_bytes: 24 * GIB,
                available_basis: "fake".into(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_round_trips_through_json() {
        let snap = FakeHw::rtx_4090().snapshot().unwrap();
        let json = serde_json::to_string(&snap).unwrap();
        assert_eq!(serde_json::from_str::<HwSnapshot>(&json).unwrap(), snap);
    }

    #[test]
    fn snapshots_without_bandwidth_still_parse() {
        // Written by loadmaster before bandwidth existed.
        let old = r#"{"gpus":[{"index":0,"name":"X","vendor":"nvidia","memory":"dedicated",
            "total_bytes":1,"usable_bytes":1,"usable_basis":"b"}],
            "host":{"total_bytes":1,"available_bytes":1}}"#;
        let snap: HwSnapshot = serde_json::from_str(old).unwrap();
        assert_eq!(snap.gpus[0].bandwidth_bytes_s, None);
    }

    #[test]
    fn find_gpu_by_index_and_name() {
        let snap = FakeHw::rtx_4090().0;
        assert_eq!(snap.find_gpu("0").unwrap().index, 0);
        assert!(snap.find_gpu("4090").is_some());
        assert!(snap.find_gpu("a100").is_none());
    }

    #[test]
    fn system_snapshot_never_panics() {
        // Runs on whatever CI has: no GPU, no driver, any OS.
        let _ = SystemHw.snapshot();
    }
}
