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

/// A dedicated card: marketing name, VRAM as `nvidia-smi` reports it, peak memory bandwidth
/// from the spec sheet (sources in `docs/decisions/0009-speed-estimates.md`), and the
/// spellings that select it (normalised: lowercase, no spaces, dashes, `nvidia` or `geforce`).
pub struct Card {
    pub name: &'static str,
    pub vram_mib: u64,
    /// GB/s (10^9 bytes per second), as vendors state it.
    pub bandwidth_gbs: f64,
    aliases: &'static [&'static str],
}

#[rustfmt::skip]
pub static CARDS: &[Card] = &[
    card("NVIDIA GeForce RTX 3060 12GB", 12288, 360.0, &["rtx3060", "3060"]),
    card("NVIDIA GeForce RTX 3080 10GB", 10240, 760.0, &["rtx3080", "3080", "rtx308010gb", "308010gb"]),
    card("NVIDIA GeForce RTX 3080 12GB", 12288, 912.0, &["rtx308012gb", "308012gb"]),
    card("NVIDIA GeForce RTX 3090", 24576, 936.0, &["rtx3090", "3090", "rtx3090ti", "3090ti"]),
    card("NVIDIA GeForce RTX 4060 Ti 16GB", 16384, 288.0, &["rtx4060ti16gb", "4060ti16gb", "rtx4060ti", "4060ti"]),
    card("NVIDIA GeForce RTX 4070", 12288, 504.0, &["rtx4070", "4070", "rtx4070super", "4070super", "rtx4070ti", "4070ti"]),
    card("NVIDIA GeForce RTX 4070 Ti Super", 16384, 672.0, &["rtx4070tisuper", "4070tisuper"]),
    card("NVIDIA GeForce RTX 4080", 16384, 716.8, &["rtx4080", "4080", "rtx4080super", "4080super"]),
    card("NVIDIA GeForce RTX 4090", 24564, 1008.0, &["rtx4090", "4090"]),
    card("NVIDIA GeForce RTX 5070 Ti", 16384, 896.0, &["rtx5070ti", "5070ti"]),
    card("NVIDIA GeForce RTX 5080", 16384, 960.0, &["rtx5080", "5080"]),
    card("NVIDIA GeForce RTX 5090", 32768, 1792.0, &["rtx5090", "5090"]),
    card("NVIDIA RTX A4000", 16376, 448.0, &["rtxa4000", "a4000"]),
    card("NVIDIA RTX A5000", 24564, 640.0, &["rtxa5000", "a5000"]),
    card("NVIDIA RTX A6000", 49140, 768.0, &["rtxa6000", "a6000"]),
    card("NVIDIA RTX 6000 Ada", 49140, 960.0, &["rtx6000ada", "6000ada", "rtx6000adageneration"]),
    card("NVIDIA L40S", 46068, 864.0, &["l40s"]),
    card("NVIDIA A100 40GB", 40960, 1555.0, &["a10040gb", "a100pcie40gb", "a100sxm440gb"]),
    card("NVIDIA A100 80GB PCIe", 81920, 1935.0, &["a100pcie80gb", "a10080gbpcie"]),
    card("NVIDIA A100 80GB SXM", 81920, 2039.0, &["a10080gb", "a100sxm80gb", "a100sxm480gb"]),
    card("NVIDIA H100 80GB PCIe", 81559, 2000.0, &["h100pcie", "h100pcie80gb"]),
    card("NVIDIA H100 80GB SXM", 81559, 3350.0, &["h100", "h10080gb", "h100sxm", "h100sxm5", "h100hbm3"]),
];

const fn card(
    name: &'static str,
    vram_mib: u64,
    bandwidth_gbs: f64,
    aliases: &'static [&'static str],
) -> Card {
    Card {
        name,
        vram_mib,
        bandwidth_gbs,
        aliases,
    }
}

/// An Apple Silicon chip; the RAM tier is always given with `--pretend` (`m4 pro 48gb`).
pub struct Chip {
    key: &'static str,
    pub name: &'static str,
    /// GB/s. Chips sold with two bandwidths by GPU core count list the lower one.
    pub bandwidth_gbs: f64,
    pub note: &'static str,
}

#[rustfmt::skip]
pub static CHIPS: &[Chip] = &[
    chip("m1", "Apple M1", 68.25, ""),
    chip("m1pro", "Apple M1 Pro", 200.0, ""),
    chip("m1max", "Apple M1 Max", 400.0, ""),
    chip("m1ultra", "Apple M1 Ultra", 800.0, ""),
    chip("m2", "Apple M2", 100.0, ""),
    chip("m2pro", "Apple M2 Pro", 200.0, ""),
    chip("m2max", "Apple M2 Max", 400.0, ""),
    chip("m2ultra", "Apple M2 Ultra", 800.0, ""),
    chip("m3", "Apple M3", 100.0, ""),
    chip("m3pro", "Apple M3 Pro", 150.0, ""),
    chip("m3max", "Apple M3 Max", 300.0, "300 GB/s with the 30-core GPU, 400 with the 40-core; the lower is assumed"),
    chip("m3ultra", "Apple M3 Ultra", 819.0, ""),
    chip("m4", "Apple M4", 120.0, ""),
    chip("m4pro", "Apple M4 Pro", 273.0, ""),
    chip("m4max", "Apple M4 Max", 410.0, "410 GB/s with the 32-core GPU, 546 with the 40-core; the lower is assumed"),
];

const fn chip(
    key: &'static str,
    name: &'static str,
    bandwidth_gbs: f64,
    note: &'static str,
) -> Chip {
    Chip {
        key,
        name,
        bandwidth_gbs,
        note,
    }
}

fn gbs_to_bytes(gbs: f64) -> u64 {
    (gbs * 1e9).round() as u64
}

/// Peak memory bandwidth for a detected GPU name (`NVIDIA GeForce RTX 4090`, `Apple M3 Max`)
/// from the catalog, with its basis; `None` for anything not listed.
pub fn bandwidth_for(name: &str) -> Option<(u64, String)> {
    let s = normalise(name);
    if let Some(c) = CARDS.iter().find(|c| c.aliases.contains(&s.as_str())) {
        return Some(card_bandwidth(c));
    }
    CHIPS.iter().find(|c| c.key == s).map(chip_bandwidth)
}

fn card_bandwidth(c: &Card) -> (u64, String) {
    (
        gbs_to_bytes(c.bandwidth_gbs),
        format!("{} GB/s, {} spec sheet", c.bandwidth_gbs, c.name),
    )
}

fn chip_bandwidth(c: &Chip) -> (u64, String) {
    (
        gbs_to_bytes(c.bandwidth_gbs),
        if c.note.is_empty() {
            format!("{} GB/s, Apple spec", c.bandwidth_gbs)
        } else {
            format!("{} GB/s, Apple spec ({})", c.bandwidth_gbs, c.note)
        },
    )
}

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
    #[error("a Mac has one GPU; pretend several dedicated cards instead (`2x rtx 3090`)")]
    MultiUnified,
    #[error("mixing GPU vendors is not supported; llama.cpp splits across one backend")]
    MixedVendors,
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

/// Bandwidth and its basis, for the pretend builders.
type Bandwidth = Option<(u64, String)>;

fn dedicated(name: String, total: u64, host: Option<u64>, bw: Bandwidth) -> HwSnapshot {
    let (bandwidth_bytes_s, bandwidth_basis) = split_bw(bw);
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
            bandwidth_bytes_s,
            bandwidth_basis,
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

fn split_bw(bw: Bandwidth) -> (Option<u64>, String) {
    match bw {
        Some((b, basis)) => (Some(b), format!("pretend: {basis}")),
        None => (None, String::new()),
    }
}

fn unified(name: String, total: u64, bw: Bandwidth) -> HwSnapshot {
    let (usable, basis) = system::apple_working_set(total, 0);
    let (bandwidth_bytes_s, bandwidth_basis) = split_bw(bw);
    HwSnapshot {
        gpus: vec![Gpu {
            index: 0,
            name: format!("{name} (pretend)"),
            vendor: Vendor::Apple,
            memory: MemoryKind::Unified,
            total_bytes: total,
            usable_bytes: usable,
            usable_basis: format!("pretend: {basis}"),
            bandwidth_bytes_s,
            bandwidth_basis,
        }],
        host: Host {
            total_bytes: total,
            available_bytes: total.saturating_sub(MACOS_HOLDS).max(total / 4),
            available_basis: "pretend: macOS assumed to hold 4 GiB".into(),
        },
    }
}

/// The hardware a `--pretend` spec describes: one or more GPUs joined by `+` (`2x rtx 3090`,
/// `rtx 4090 + rtx 3060`), and optionally host RAM as a bare size after a `+`.
pub fn pretend(spec: &str) -> Result<HwSnapshot, PretendError> {
    let mut host = None;
    let mut gpus: Vec<&str> = Vec::new();
    for (i, part) in spec.split('+').map(str::trim).enumerate() {
        let n = normalise(part);
        if i > 0
            && let Some(h) = size(&n)
        {
            if host.replace(h).is_some() {
                return Err(PretendError::BadSize(part.into()));
            }
            continue;
        }
        let (count, gpu) = repeat(part);
        if !(1..=16).contains(&count) {
            return Err(PretendError::BadSize(part.into()));
        }
        gpus.extend(std::iter::repeat_n(gpu, count as usize));
    }
    let mut snaps = gpus
        .iter()
        .map(|g| one(g, host))
        .collect::<Result<Vec<_>, _>>()?;
    let mut out = snaps.remove(0);
    for s in snaps {
        out.gpus.extend(s.gpus);
    }
    if out.gpus.len() > 1 {
        if out.gpus.iter().any(|g| g.memory == MemoryKind::Unified) {
            return Err(PretendError::MultiUnified);
        }
        if out.gpus.iter().any(|g| g.vendor != out.gpus[0].vendor) {
            return Err(PretendError::MixedVendors);
        }
        for (i, g) in out.gpus.iter_mut().enumerate() {
            g.index = i as u32;
        }
    }
    Ok(out)
}

/// `2x rtx 3090` / `2 x 3090` / `2× 3090` → (2, `rtx 3090`); anything else is one GPU.
fn repeat(part: &str) -> (u32, &str) {
    let t = part.trim();
    let digits = t.bytes().take_while(u8::is_ascii_digit).count();
    if digits > 0 {
        let rest = t[digits..].trim_start();
        for x in ["x", "X", "×"] {
            if let Some(gpu) = rest.strip_prefix(x)
                && let Ok(n) = t[..digits].parse()
                && !gpu.trim().is_empty()
            {
                return (n, gpu.trim());
            }
        }
    }
    (1, t)
}

/// One GPU from the catalog or a raw size, with `host` RAM.
fn one(gpu: &str, host: Option<u64>) -> Result<HwSnapshot, PretendError> {
    let s = normalise(gpu);

    if let Some(c) = CARDS.iter().find(|c| c.aliases.contains(&s.as_str())) {
        return Ok(dedicated(
            c.name.into(),
            c.vram_mib * MIB,
            host,
            Some(card_bandwidth(c)),
        ));
    }

    // Raw sizes: `24GB`, `48GB-unified`, optionally with bandwidth: `24GB@936`.
    let (raw, bw) = match s.split_once('@') {
        Some((r, b)) => {
            let gbs: f64 = b
                .trim_end_matches("gb/s")
                .trim_end_matches("gbs")
                .parse()
                .ok()
                .filter(|g| *g > 0.0)
                .ok_or_else(|| PretendError::BadSize(gpu.trim().into()))?;
            (
                r,
                Some((gbs_to_bytes(gbs), format!("{gbs} GB/s, given with @"))),
            )
        }
        None => (s.as_str(), None),
    };
    if let Some(r) = raw.strip_suffix("unified") {
        let total = size(r).ok_or_else(|| PretendError::BadSize(gpu.trim().into()))?;
        return Ok(unified(
            format!("{} unified memory", crate::gib_str(total)),
            total,
            bw,
        ));
    }
    if let Some(total) = size(raw) {
        return Ok(dedicated(
            format!("{} GPU", crate::gib_str(total)),
            total,
            host,
            bw,
        ));
    }

    // Apple: chip then RAM, e.g. `m4pro48gb`; longest chip key first so `m4pro` beats `m4`.
    let mut chips: Vec<&Chip> = CHIPS.iter().collect();
    chips.sort_by_key(|c| std::cmp::Reverse(c.key.len()));
    for c in chips {
        if let Some(rest) = s.strip_prefix(c.key) {
            if rest.is_empty() {
                return Err(PretendError::NeedsRam(gpu.trim().into()));
            }
            if let Some(total) = size(rest) {
                return Ok(unified(
                    format!("{} {}", c.name, crate::gib_str(total)),
                    total,
                    Some(chip_bandwidth(c)),
                ));
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
            Err(PretendError::Unknown { .. })
        ));
    }

    #[test]
    fn bandwidth_from_catalog_names_and_raw_sizes() {
        assert_eq!(gpu("rtx 4090").bandwidth_bytes_s, Some(1_008_000_000_000));
        assert_eq!(gpu("m4 max 128gb").bandwidth_bytes_s, Some(410_000_000_000));
        assert!(
            gpu("m4 max 128gb")
                .bandwidth_basis
                .contains("lower is assumed")
        );
        assert_eq!(gpu("24GB").bandwidth_bytes_s, None);
        assert_eq!(gpu("24GB@936").bandwidth_bytes_s, Some(936_000_000_000));
        assert_eq!(
            pretend("48GB-unified@400").unwrap().gpus[0].bandwidth_bytes_s,
            Some(400_000_000_000)
        );
        // Detected names, as NVML and sysctl report them.
        assert_eq!(
            bandwidth_for("NVIDIA GeForce RTX 3090").unwrap().0,
            936_000_000_000
        );
        assert_eq!(bandwidth_for("Apple M2").unwrap().0, 100_000_000_000);
        assert_eq!(bandwidth_for("Some Future GPU"), None);
        assert!(matches!(
            pretend("24GB@fast"),
            Err(PretendError::BadSize(_))
        ));
    }

    #[test]
    fn every_catalog_entry_has_its_bandwidth() {
        for c in CARDS {
            let g = gpu(c.aliases[0]);
            assert_eq!(
                g.bandwidth_bytes_s,
                Some(gbs_to_bytes(c.bandwidth_gbs)),
                "{}",
                c.name
            );
        }
        for c in CHIPS {
            let g = gpu(&format!("{} 16gb", c.key));
            assert_eq!(
                g.bandwidth_bytes_s,
                Some(gbs_to_bytes(c.bandwidth_gbs)),
                "{}",
                c.name
            );
        }
    }

    #[test]
    fn several_gpus() {
        let s = pretend("2x rtx 3090").unwrap();
        assert_eq!(s.gpus.len(), 2);
        assert_eq!((s.gpus[0].index, s.gpus[1].index), (0, 1));
        assert!(s.gpus.iter().all(|g| g.name.contains("RTX 3090")));
        let s = pretend("rtx 4090 + rtx 3060 + 128GB").unwrap();
        assert_eq!(s.gpus.len(), 2);
        assert!(s.gpus[1].name.contains("3060"));
        assert_eq!(s.host.total_bytes, 128 * GIB);
        assert_eq!(pretend("4 x a6000").unwrap().gpus.len(), 4);
        // A card name that starts with digits is not a count.
        assert_eq!(pretend("4090").unwrap().gpus.len(), 1);
        assert_eq!(pretend("2x m4 max 64gb"), Err(PretendError::MultiUnified));
        assert!(pretend("0x 3090").is_err());
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
