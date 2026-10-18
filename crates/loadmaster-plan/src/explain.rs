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
    for w in &p.warnings {
        writeln!(out, "warning: {w}").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "{}", emit::command(p)).unwrap();
    alternatives(&mut out, p);
    out
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
        let speed = match a.slowdown_x100 {
            Some((lo, hi)) => format!(
                "{:.1}-{:.1}x slower (estimate)",
                f64::from(lo) / 100.0,
                f64::from(hi) / 100.0
            ),
            None if a.fits => "full GPU speed".to_owned(),
            None => String::new(),
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
