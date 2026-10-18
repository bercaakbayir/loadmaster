//! Human-readable output: the short summary and the `--explain` breakdown, grouped as
//! WEIGHT (what the load is) and BALANCE (where it goes).

use std::fmt::Write;

use crate::{NoFit, Plan, Term, emit, gib};

fn label(key: &str) -> &'static str {
    match key {
        "weights" => "weights",
        "kv" => "KV cache",
        "compute" => "compute buffer",
        "mmproj" => "vision projector",
        "headroom" => "headroom",
        "weights_cpu" => "weights on CPU",
        "kv_cpu" => "KV on CPU",
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
    if pl.is_full(n) {
        format!("all {n} layers and the output head on GPU")
    } else if pl.output_on_gpu(n) {
        format!(
            "all {n} layers on GPU, routed experts of the first {} on CPU",
            pl.n_cpu_moe
        )
    } else {
        format!(
            "{}/{n} layers on GPU, the rest and the output head on CPU{}",
            pl.gpu_layer_count(n),
            if pl.n_cpu_moe > 0 {
                ", routed experts on CPU"
            } else {
                ""
            }
        )
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
        p.kv.flag()
    )
    .unwrap();
    for w in &p.warnings {
        writeln!(out, "warning: {w}").unwrap();
    }
    writeln!(out).unwrap();
    writeln!(out, "{}", emit::llama_cpp::command(p)).unwrap();
    out
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
    writeln!(
        out,
        "  {:<18}{:>11}   of {} available host memory",
        "host total",
        gib(p.footprint.host_bytes),
        gib(p.host_budget)
    )
    .unwrap();
    writeln!(out).unwrap();

    writeln!(out, "BALANCE  {}", placement_line(p)).unwrap();
    writeln!(out, "  ctx {}: {}", p.ctx, p.ctx_reason).unwrap();
    writeln!(out, "  KV cache {}: {}", p.kv.flag(), p.kv_reason).unwrap();
    writeln!(
        out,
        "  flash attention {}: required for a quantised V cache and smaller compute buffers",
        if p.flash_attn { "on" } else { "off" }
    )
    .unwrap();
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
    writeln!(out, "  {}", emit::llama_cpp::command(p)).unwrap();
    out
}

pub fn no_fit(n: &NoFit, explain: bool) -> String {
    let mut out = String::new();
    writeln!(out, "{}  ·  ctx {}", model_line(&n.model), n.ctx).unwrap();
    writeln!(
        out,
        "does not fit: {} short on {} ({})",
        gib(n.short_by),
        if n.limit == "gpu" {
            n.gpu.name.as_str()
        } else {
            "host memory"
        },
        if n.limit == "gpu" {
            format!("{} usable", gib(n.gpu.usable_bytes))
        } else {
            "even with every layer on the CPU".to_owned()
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
