//! Pretend hardware for `--pretend`: plan against a GPU you do not have, by catalog name
//! (`rtx 4090`, `m4 max 128gb`) or raw size (`24GB`, `48GB-unified`), with an optional host
//! RAM suffix (`rtx 3090+128GB`). Figures are assumptions, labelled as such in every basis.

use crate::{GIB, Gpu, Host, HwSnapshot, MIB, MemoryKind, Vendor, system};

/// Held by the driver and a desktop session on a dedicated card (as `FakeHw::rtx_4090`).
const DRIVER_RESERVE: u64 = 500 * MIB;
/// Host RAM assumed next to a dedicated card when no `+<size>` is given.
const DEFAULT_HOST: u64 = 32 * GIB;
/// RAM a freshly booted macOS holds.
const MACOS_HOLDS: u64 = 4 * GIB;

/// A dedicated card: marketing name, VRAM as `nvidia-smi` reports it, and the spellings
/// that select it (normalised: lowercase, no spaces, dashes, `nvidia` or `geforce`).
pub struct Card {
    pub name: &'static str,
    pub vram_mib: u64,
    aliases: &'static [&'static str],
}

pub static CARDS: &[Card] = &[
    card("NVIDIA GeForce RTX 3060 12GB", 12288, &["rtx3060", "3060"]),
    card(
        "NVIDIA GeForce RTX 3080 10GB",
        10240,
        &["rtx3080", "3080", "rtx308010gb", "308010gb"],
    ),
    card(
        "NVIDIA GeForce RTX 3080 12GB",
        12288,
        &["rtx308012gb", "308012gb"],
    ),
    card(
        "NVIDIA GeForce RTX 3090",
        24576,
        &["rtx3090", "3090", "rtx3090ti", "3090ti"],
    ),
    card(
        "NVIDIA GeForce RTX 4060 Ti 16GB",
        16384,
        &["rtx4060ti16gb", "4060ti16gb", "rtx4060ti", "4060ti"],
    ),
    card(
        "NVIDIA GeForce RTX 4070",
        12288,
        &[
            "rtx4070",
            "4070",
            "rtx4070super",
            "4070super",
            "rtx4070ti",
            "4070ti",
        ],
    ),
    card(
        "NVIDIA GeForce RTX 4070 Ti Super",
        16384,
        &["rtx4070tisuper", "4070tisuper"],
    ),
    card(
        "NVIDIA GeForce RTX 4080",
        16384,
        &["rtx4080", "4080", "rtx4080super", "4080super"],
    ),
    card("NVIDIA GeForce RTX 4090", 24564, &["rtx4090", "4090"]),
    card(
        "NVIDIA GeForce RTX 5070 Ti",
        16384,
        &["rtx5070ti", "5070ti"],
    ),
    card("NVIDIA GeForce RTX 5080", 16384, &["rtx5080", "5080"]),
    card("NVIDIA GeForce RTX 5090", 32768, &["rtx5090", "5090"]),
    card("NVIDIA RTX A4000", 16376, &["rtxa4000", "a4000"]),
    card("NVIDIA RTX A5000", 24564, &["rtxa5000", "a5000"]),
    card("NVIDIA RTX A6000", 49140, &["rtxa6000", "a6000"]),
    card("NVIDIA RTX 6000 Ada", 49140, &["rtx6000ada", "6000ada"]),
    card("NVIDIA L40S", 46068, &["l40s"]),
    card("NVIDIA A100 40GB", 40960, &["a10040gb", "a100pcie40gb"]),
    card("NVIDIA A100 80GB", 81920, &["a10080gb", "a100sxm80gb"]),
    card(
        "NVIDIA H100 80GB",
        81559,
        &["h100", "h10080gb", "h100sxm", "h100pcie"],
    ),
];

const fn card(name: &'static str, vram_mib: u64, aliases: &'static [&'static str]) -> Card {
    Card {
        name,
        vram_mib,
        aliases,
    }
}

/// Apple Silicon chips; the RAM tier is always given (`m4 pro 48gb`).
pub static CHIPS: &[(&str, &str)] = &[
    ("m1", "Apple M1"),
    ("m1pro", "Apple M1 Pro"),
    ("m1max", "Apple M1 Max"),
    ("m1ultra", "Apple M1 Ultra"),
    ("m2", "Apple M2"),
    ("m2pro", "Apple M2 Pro"),
    ("m2max", "Apple M2 Max"),
    ("m2ultra", "Apple M2 Ultra"),
    ("m3", "Apple M3"),
    ("m3pro", "Apple M3 Pro"),
    ("m3max", "Apple M3 Max"),
    ("m3ultra", "Apple M3 Ultra"),
    ("m4", "Apple M4"),
    ("m4pro", "Apple M4 Pro"),
    ("m4max", "Apple M4 Max"),
];

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PretendError {
    #[error(
        "unknown hardware `{spec}`{hint}; `loadmaster gpu --pretend list` shows the catalog, or pass a size like `24GB` or `48GB-unified`"
    )]
    Unknown { spec: String, hint: String },
    #[error("`{0}`: give the Mac's RAM, e.g. `{0} 16gb`")]
    NeedsRam(String),
    #[error("`{0}`: not a size (expected e.g. `24GB`)")]
    BadSize(String),
}

fn normalise(s: &str) -> String {
    s.to_lowercase()
        .replace("nvidia", "")
        .replace("geforce", "")
        .replace("apple", "")
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '-' && *c != '_')
        .collect()
}

/// `24gb`, `24GiB`, `24g`: always GiB (vendors' memory "GB" are GiB). A unit is required,
/// so `4090` is a card, not a size. Already normalised.
fn size(s: &str) -> Option<u64> {
    let digits = s
        .strip_suffix("gib")
        .or_else(|| s.strip_suffix("gb"))
        .or_else(|| s.strip_suffix('g'))?;
    let n: f64 = digits.parse().ok()?;
    (n > 0.0).then_some((n * GIB as f64) as u64)
}

fn dedicated(name: String, total: u64, host: Option<u64>) -> HwSnapshot {
    let host_total = host.unwrap_or(DEFAULT_HOST);
    HwSnapshot {
        gpus: vec![Gpu {
            index: 0,
            name: format!("{name} (pretend)"),
            vendor: Vendor::Nvidia,
            memory: MemoryKind::Dedicated,
            total_bytes: total,
            usable_bytes: total.saturating_sub(DRIVER_RESERVE),
            usable_basis: "pretend: total minus 500 MiB for the driver and a desktop session"
                .into(),
        }],
        host: Host {
            total_bytes: host_total,
            available_bytes: host_total * 3 / 4,
            available_basis: format!(
                "pretend: {} host RAM{}, 3/4 of it free",
                crate::gib_str(host_total),
                if host.is_some() {
                    ""
                } else {
                    " (default; add +<size> to change)"
                }
            ),
        },
    }
}

fn unified(name: String, total: u64) -> HwSnapshot {
    let (usable, basis) = system::apple_working_set(total, 0);
    HwSnapshot {
        gpus: vec![Gpu {
            index: 0,
            name: format!("{name} (pretend)"),
            vendor: Vendor::Apple,
            memory: MemoryKind::Unified,
            total_bytes: total,
            usable_bytes: usable,
            usable_basis: format!("pretend: {basis}"),
        }],
        host: Host {
            total_bytes: total,
            available_bytes: total.saturating_sub(MACOS_HOLDS).max(total / 4),
            available_basis: "pretend: macOS assumed to hold 4 GiB".into(),
        },
    }
}

/// The hardware a `--pretend` spec describes.
pub fn pretend(spec: &str) -> Result<HwSnapshot, PretendError> {
    let (gpu, host) = match spec.split_once('+') {
        Some((g, h)) => {
            let h = normalise(h);
            (g, Some(size(&h).ok_or(PretendError::BadSize(h))?))
        }
        None => (spec, None),
    };
    let s = normalise(gpu);

    if let Some(c) = CARDS.iter().find(|c| c.aliases.contains(&s.as_str())) {
        return Ok(dedicated(c.name.into(), c.vram_mib * MIB, host));
    }

    // Raw sizes: `24GB`, `48GB-unified`.
    if let Some(raw) = s.strip_suffix("unified") {
        let total = size(raw).ok_or_else(|| PretendError::BadSize(gpu.trim().into()))?;
        return Ok(unified(
            format!("{} unified memory", crate::gib_str(total)),
            total,
        ));
    }
    if let Some(total) = size(&s) {
        return Ok(dedicated(
            format!("{} GPU", crate::gib_str(total)),
            total,
            host,
        ));
    }

    // Apple: chip then RAM, e.g. `m4pro48gb`; longest chip key first so `m4pro` beats `m4`.
    let mut chips: Vec<_> = CHIPS.iter().collect();
    chips.sort_by_key(|(k, _)| std::cmp::Reverse(k.len()));
    for (key, name) in chips {
        if let Some(rest) = s.strip_prefix(key) {
            if rest.is_empty() {
                return Err(PretendError::NeedsRam(gpu.trim().into()));
            }
            if let Some(total) = size(rest) {
                return Ok(unified(format!("{name} {}", crate::gib_str(total)), total));
            }
        }
    }

    let close: Vec<&str> = CARDS
        .iter()
        .filter(|c| !s.is_empty() && c.aliases.iter().any(|a| a.contains(s.as_str())))
        .map(|c| c.name)
        .collect();
    Err(PretendError::Unknown {
        spec: gpu.trim().into(),
        hint: if close.is_empty() {
            String::new()
        } else {
            format!(" (did you mean {}?)", close.join(", "))
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu(spec: &str) -> Gpu {
        pretend(spec).unwrap().gpus.remove(0)
    }

    #[test]
    fn catalog_names_in_any_spelling() {
        for s in ["rtx 4090", "4090", "NVIDIA GeForce RTX 4090", "RTX-4090"] {
            let g = gpu(s);
            assert_eq!(g.name, "NVIDIA GeForce RTX 4090 (pretend)", "{s}");
            assert_eq!(g.total_bytes, 24564 * MIB);
            assert_eq!(g.usable_bytes, 24564 * MIB - DRIVER_RESERVE);
            assert_eq!(g.memory, MemoryKind::Dedicated);
        }
    }

    #[test]
    fn every_alias_is_unique() {
        let mut all: Vec<&str> = CARDS
            .iter()
            .flat_map(|c| c.aliases.iter().copied())
            .collect();
        let n = all.len();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), n);
    }

    #[test]
    fn raw_sizes() {
        let g = gpu("24GB");
        assert_eq!((g.total_bytes, g.memory), (24 * GIB, MemoryKind::Dedicated));
        let snap = pretend("48GB-unified").unwrap();
        assert_eq!(snap.gpus[0].memory, MemoryKind::Unified);
        assert_eq!(snap.gpus[0].usable_bytes, 48 * GIB * 3 / 4);
        assert_eq!(snap.host.available_bytes, 44 * GIB);
    }

    #[test]
    fn apple_chips_need_ram() {
        let snap = pretend("m4 pro 48gb").unwrap();
        assert_eq!(snap.gpus[0].name, "Apple M4 Pro 48 GiB (pretend)");
        assert_eq!(
            snap.gpus[0].usable_bytes,
            system::apple_working_set(48 * GIB, 0).0
        );
        assert_eq!(gpu("M2 8GB").usable_bytes, 8 * GIB * 2 / 3);
        assert_eq!(
            pretend("m4 max"),
            Err(PretendError::NeedsRam("m4 max".into()))
        );
    }

    #[test]
    fn host_suffix() {
        let snap = pretend("rtx 3090+128GB").unwrap();
        assert_eq!(snap.host.total_bytes, 128 * GIB);
        assert_eq!(pretend("3090").unwrap().host.total_bytes, DEFAULT_HOST);
        assert!(matches!(
            pretend("3090+lots"),
            Err(PretendError::BadSize(_))
        ));
    }

    #[test]
    fn unknown_suggests_close_matches() {
        let e = pretend("a100").unwrap_err().to_string();
        assert!(
            e.contains("NVIDIA A100 40GB") && e.contains("NVIDIA A100 80GB"),
            "{e}"
        );
        assert!(matches!(
            pretend("voodoo"),
            Err(PretendError::Unknown { .. })
        ));
    }
}
