//! Human-readable output: the short summary and the `--explain` breakdown, grouped as
//! WEIGHT (what the load is) and BALANCE (where it goes).

use std::fmt::Write;

use crate::{NoFit, Plan, Term, emit, gib};

fn label(key: &str) -> &'static str {
    match key {
        "weights" => "weights",
        "kv" => "KV cache",
        "compute" => "compute buffer",
        "activation" => "activations",
        "mmproj" => "vision projector",
        "headroom" => "headroom",
        "weights_cpu" => "weights on CPU",
        "kv_cpu" => "KV on CPU",
        "compute_cpu" => "compute on CPU",
        _ => "other",
    }
}

fn model_line(m: &crate::ModelSummary) -> String {
    format!(
        "{} · {} · {} · {} layers{}",
        m.name.as_deref().unwrap_or(&m.path),
        m.arch,
        m.quant,
        m.n_layer,
        if m.moe { " · MoE" } else { "" }
    )
}

fn terms(out: &mut String, terms: &[Term]) {
    for t in terms {
        writeln!(
            out,
            "  {:<18}{:>11}   {}",
            label(t.key),
            gib(t.bytes),
            t.detail
        )
        .unwrap();
    }
}

fn placement_line(p: &Plan) -> String {
    let n = p.model.n_layer;
    let pl = &p.placement;
    let layers = pl.gpu_layer_count(n);
    let experts = if pl.n_cpu_moe > 0 {
        format!(
            ", routed experts of the first {} on CPU",
            pl.n_cpu_moe.min(n)
        )
    } else {
        String::new()
    };
    if !pl.output_on_gpu(n) {
        "everything on CPU".to_owned()
    } else if layers == n {
        format!("all {n} layers and the output head on GPU{experts}")
    } else {
        format!("{layers}/{n} layers and the output head on GPU, the rest on CPU{experts}")
    }
}

/// The KV cache type in the runtime's own words.
pub fn kv_label(p: &Plan) -> &'static str {
    match (p.runtime, p.kv) {
        (crate::Runtime::Vllm, crate::KvType::F16) => "auto",
        (crate::Runtime::Exl3, kv) => emit::exl3::cache_mode(kv),
        (_, kv) => kv.flag(),
    }
}

/// Default output: one verdict line, the command, warnings.
pub fn summary(p: &Plan) -> String {
    let mut out = String::new();
    writeln!(out, "{}  ·  ctx {}", model_line(&p.model), p.ctx).unwrap();
    writeln!(
        out,
        "fits: {} of {} usable on {}; {}, {} KV",
        gib(p.footprint.gpu_bytes),
        gib(p.budget),
        p.gpu.name,
        placement_line(p),
        kv_label(p)
    )
    .unwrap();
    if let Some(s) = &p.speed {
        writeln!(out, "speed: {} (estimate)", speed_line(s)).unwrap();
    }
    for w in &p.warnings {
        writeln!(out, "warning: {w}").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "{}", emit::command(p)).unwrap();
    alternatives(&mut out, p);
    out
}

/// `~95-150 tok/s generating at 4k context, ~70-115 at 32k`.
fn speed_line(s: &crate::speed::Speed) -> String {
    let k = |fill: u32| {
        if fill >= 1024 && fill % 1024 == 0 {
            format!("{}k", fill / 1024)
        } else {
            fill.to_string()
        }
    };
    s.at.iter()
        .enumerate()
        .map(|(i, a)| {
            let b = crate::speed::band_text(a.tok_s_x10);
            if i == 0 {
                format!("~{b} generating at {} context", k(a.fill))
            } else {
                format!("~{} at {}", b.trim_end_matches(" tok/s"), k(a.fill))
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

fn speed_section(out: &mut String, s: &crate::speed::Speed) {
    let band = match s.backend {
        "Metal" => crate::speed::METAL,
        _ => crate::speed::CUDA,
    };
    writeln!(out).unwrap();
    writeln!(out, "SPEED    generation, memory-bound (estimate)").unwrap();
    writeln!(
        out,
        "  bandwidth          {:.0} GB/s ({})",
        s.bandwidth_bytes_s as f64 / 1e9,
        s.bandwidth_basis
    )
    .unwrap();
    writeln!(
        out,
        "  {:<18} efficiency {:.2}-{:.2}, {:.0}-{:.0} us per layer per token (ADR 0009)",
        format!("{} fit", band.name),
        band.eff.0,
        band.eff.1,
        band.per_layer_us.0,
        band.per_layer_us.1
    )
    .unwrap();
    if let Some(h) = &s.host_basis {
        writeln!(out, "  {:<18} {h}", "CPU-side reads").unwrap();
    }
    for a in &s.at {
        writeln!(
            out,
            "  {:<18} read {} from GPU{} per token: {}",
            format!("at {} tokens", a.fill),
            gib(a.gpu_bytes),
            if a.cpu_bytes > 0 {
                format!(" + {} from RAM", gib(a.cpu_bytes))
            } else {
                String::new()
            },
            crate::speed::band_text(a.tok_s_x10)
        )
        .unwrap();
    }
}

/// The alternatives table, under the command.
fn alternatives(out: &mut String, p: &Plan) {
    if p.alternatives.is_empty() {
        return;
    }
    writeln!(out).unwrap();
    writeln!(out, "alternatives:").unwrap();
    for a in &p.alternatives {
        let delta = if a.delta_bytes >= 0 {
            format!("+{}", gib(a.delta_bytes.unsigned_abs()))
        } else {
            format!("-{}", gib(a.delta_bytes.unsigned_abs()))
        };
        let speed = match (a.tok_s_x10, a.slowdown_x100) {
            (Some(t), _) => format!("~{} (estimate)", crate::speed::band_text(t)),
            (None, Some((lo, hi))) => format!(
                "{:.1}-{:.1}x slower (estimate)",
                f64::from(lo) / 100.0,
                f64::from(hi) / 100.0
            ),
            (None, None) if a.fits => "full GPU speed".to_owned(),
            (None, None) => String::new(),
        };
        writeln!(
            out,
            "  {:<22} {:>11} ({:>10})  {:<38} {}",
            a.label,
            gib(a.gpu_bytes),
            delta,
            a.placement,
            speed
        )
        .unwrap();
        if let Some(note) = &a.note {
            writeln!(out, "  {:<22} {note}", "").unwrap();
        }
    }
}

/// `--explain`: every term with its inputs, every choice with its reason.
pub fn explain(p: &Plan) -> String {
    let mut out = String::new();
    writeln!(out, "WEIGHT   {}", model_line(&p.model)).unwrap();
    terms(&mut out, &p.footprint.gpu);
    writeln!(out, "  {}", "-".repeat(30)).unwrap();
    writeln!(
        out,
        "  {:<18}{:>11}   of {} usable on {} ({}{})",
        "GPU total",
        gib(p.footprint.gpu_bytes),
        gib(p.budget),
        p.gpu.name,
        p.gpu.usable_basis,
        match p.gpu.in_use_bytes() {
            0 => String::new(),
            b => format!("; {} held by other processes", gib(b)),
        }
    )
    .unwrap();
    terms(&mut out, &p.footprint.host);
    if p.gpu.memory == loadmaster_hw::MemoryKind::Unified {
        writeln!(
            out,
            "  {:<18}{:>11}   of {} RAM available now ({}; unified memory: GPU side minus headroom + host side)",
            "RAM needed",
            gib(p.host_need),
            gib(p.host_budget),
            p.host_basis
        )
        .unwrap();
    } else {
        writeln!(
            out,
            "  {:<18}{:>11}   of {} available host memory",
            "host total",
            gib(p.host_need),
            gib(p.host_budget)
        )
        .unwrap();
    }
    writeln!(out).unwrap();

    writeln!(out, "BALANCE  {}", placement_line(p)).unwrap();
    writeln!(out, "  ctx {}: {}", p.ctx, p.ctx_reason).unwrap();
    writeln!(out, "  KV cache {}: {}", kv_label(p), p.kv_reason).unwrap();
    if p.runtime == crate::Runtime::LlamaCpp {
        writeln!(
            out,
            "  flash attention {}: required for a quantised V cache and smaller compute buffers",
            if p.flash_attn { "on" } else { "off" }
        )
        .unwrap();
    }
    writeln!(out, "  tried:").unwrap();
    for t in &p.trials {
        writeln!(
            out,
            "    {} {:<50} GPU {:>11}  host {:>11}",
            if t.fits { "fits" } else { "over" },
            t.label,
            gib(t.gpu_bytes),
            gib(t.host_bytes)
        )
        .unwrap();
    }
    for w in &p.warnings {
        writeln!(out, "  warning: {w}").unwrap();
    }
    if let Some(s) = &p.speed {
        speed_section(&mut out, s);
    }
    writeln!(out).unwrap();
    writeln!(out, "COMMAND").unwrap();
    for line in emit::command(p).lines() {
        writeln!(out, "  {line}").unwrap();
    }
    alternatives(&mut out, p);
    out
}

pub fn no_fit(n: &NoFit, explain: bool) -> String {
    let mut out = String::new();
    writeln!(out, "{}  ·  ctx {}", model_line(&n.model), n.ctx).unwrap();
    writeln!(
        out,
        "does not fit: {} short on {} ({})",
        gib(n.short_by),
        match (n.limit, n.gpu.memory) {
            ("gpu", _) => n.gpu.name.as_str(),
            (_, loadmaster_hw::MemoryKind::Unified) => "free RAM",
            _ => "host memory",
        },
        match (n.limit, n.gpu.memory) {
            ("gpu", _) => format!("{} usable", gib(n.gpu.usable_bytes)),
            (_, loadmaster_hw::MemoryKind::Unified) => {
                format!(
                    "unified memory: GPU and CPU share {}",
                    gib(n.gpu.total_bytes)
                )
            }
            _ => "even with every layer on the CPU".to_owned(),
        }
    )
    .unwrap();
    writeln!(out, "cheapest fixes:").unwrap();
    for f in &n.fixes {
        writeln!(out, "  - {}", f.summary).unwrap();
    }
    if explain {
        writeln!(out).unwrap();
        writeln!(out, "WEIGHT   closest configuration").unwrap();
        terms(&mut out, &n.closest.gpu);
        terms(&mut out, &n.closest.host);
        writeln!(out).unwrap();
        writeln!(out, "BALANCE  tried:").unwrap();
        for t in &n.trials {
            writeln!(
                out,
                "    {} {:<50} GPU {:>11}  host {:>11}",
                if t.fits { "fits" } else { "over" },
                t.label,
                gib(t.gpu_bytes),
                gib(t.host_bytes)
            )
            .unwrap();
        }
    }
    out
}
