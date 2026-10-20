//! Human-readable output: the short summary (verdict first, one fact per line) and the
//! `--explain` breakdown, grouped as WEIGHT (what the load is) and BALANCE (where it goes).

mod style;

use std::fmt::Write;

use loadmaster_hw::Gpu;

pub use style::Style;

use crate::{NoFit, Plan, Term, WarningKind, emit, gib};

/// Columns the wrapped launch command stays within.
const COMMAND_WIDTH: usize = 88;

/// GiB with one decimal, for the summary (`--explain` keeps two).
pub fn gib1(bytes: u64) -> String {
    format!("{:.1} GiB", gib_f(bytes))
}

fn gib_f(bytes: u64) -> f64 {
    bytes as f64 / (1u64 << 30) as f64
}

/// `2× RTX 3090`, `RTX 4090 + RTX 3060 12GB (pretend)`, `Apple M2`: GPU names without the
/// vendor prefix, repeats counted, `(pretend)` said once.
pub fn hw_label(gpus: &[&Gpu]) -> String {
    let pretend = gpus.iter().any(|g| g.name.ends_with(" (pretend)"));
    let mut parts: Vec<(&str, usize)> = Vec::new();
    for g in gpus {
        let n = g
            .name
            .trim_end_matches(" (pretend)")
            .trim_start_matches("NVIDIA GeForce ")
            .trim_start_matches("NVIDIA ");
        match parts.last_mut() {
            Some((p, c)) if *p == n => *c += 1,
            _ => parts.push((n, 1)),
        }
    }
    let label = parts
        .iter()
        .map(|(n, c)| {
            if *c > 1 {
                format!("{c}× {n}")
            } else {
                (*n).to_owned()
            }
        })
        .collect::<Vec<_>>()
        .join(" + ");
    if pretend {
        format!("{label} (pretend)")
    } else {
        label
    }
}

fn plan_gpus(p: &Plan) -> Vec<&Gpu> {
    if p.gpus.is_empty() {
        vec![&p.gpu]
    } else {
        p.gpus.iter().collect()
    }
}

/// `Llama 3.3 70B Instruct  ·  Q4_K_M  ·  80 layers · MoE  ·  ctx 32768`.
fn header(m: &crate::ModelSummary, ctx: u32, style: Style) -> String {
    format!(
        "{}  ·  {}  ·  {} layers{}  ·  ctx {ctx}",
        style.bold(m.name.as_deref().unwrap_or(&m.path)),
        m.quant,
        m.n_layer,
        if m.moe { " · MoE" } else { "" }
    )
}

fn fact(out: &mut String, label: &str, value: &str) {
    writeln!(out, "  {label:<10} {value}").unwrap();
}

/// The launch command, wrapped for a terminal; an exl3 config file follows it.
fn command_lines(p: &Plan, style: Style) -> Vec<String> {
    if !style.wrap {
        return emit::command(p).lines().map(str::to_owned).collect();
    }
    let mut lines = emit::shell_wrap(&p.argv, COMMAND_WIDTH);
    if let Some(f) = &p.config_file {
        lines.push(String::new());
        lines.push(format!("# {}", f.name));
        lines.extend(f.contents.trim_end().lines().map(str::to_owned));
    }
    lines
}

fn label(key: &str) -> &'static str {
    match key {
        "weights" => "weights",
        "kv" => "KV cache",
        "compute" => "compute buffer",
        "activation" => "activations",
        "mmproj" => "vision projector",
        "vision_compute" => "vision compute",
        "mmproj_cpu" => "projector on CPU",
        "vision_compute_cpu" => "vision compute CPU",
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

/// BERT-style encoders: no KV cache and no output head.
fn is_encoder(p: &Plan) -> bool {
    matches!(
        p.model.kind,
        loadmaster_model::ModelKind::Embedding { encoder: true, .. }
    )
}

fn placement_line(p: &Plan) -> String {
    let n = p.model.n_layer;
    let head = if is_encoder(p) {
        ""
    } else {
        " and the output head"
    };
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
        format!("all {n} layers{head} on GPU{experts}")
    } else {
        format!("{layers}/{n} layers{head} on GPU, the rest on CPU{experts}")
    }
}

/// The KV cache type in the runtime's own words.
pub fn kv_label(p: &Plan) -> &'static str {
    if is_encoder(p) {
        return "no";
    }
    match (p.runtime, p.kv) {
        (crate::Runtime::Vllm, crate::KvType::F16) => "auto",
        (crate::Runtime::Exl3, kv) => emit::exl3::cache_mode(kv),
        (_, kv) => kv.flag(),
    }
}

/// Default output: the verdict, one fact per line, short warnings, the command and the
/// nearby options.
pub fn summary(p: &Plan, style: Style) -> String {
    let mut out = String::new();
    writeln!(out, "{}", header(&p.model, p.ctx, style)).unwrap();
    writeln!(out).unwrap();
    let hw = hw_label(&plan_gpus(p));
    if p.warnings.iter().any(|w| w.is(WarningKind::TightRam)) {
        writeln!(
            out,
            "{}",
            style.yellow(&format!("! Fits on {hw}, tight on RAM"))
        )
        .unwrap();
    } else {
        writeln!(out, "{}", style.green(&format!("✓ Fits on {hw}"))).unwrap();
    }

    let usable = if p.gpus.is_empty() {
        p.budget
    } else {
        p.gpus.iter().map(|g| g.usable_bytes).sum()
    };
    let mut memory = format!("{:.1} of {}", gib_f(p.footprint.gpu_bytes), gib1(usable));
    if !p.gpus.is_empty() {
        let shares = p
            .footprint
            .per_device
            .iter()
            .zip(&p.gpus)
            .enumerate()
            .map(|(i, (d, g))| format!("GPU{i} {:.1}/{:.1}", gib_f(d.bytes), gib_f(g.usable_bytes)))
            .collect::<Vec<_>>()
            .join(" · ");
        memory = format!("{memory}   {shares}");
    }
    fact(&mut out, "memory", &memory);
    let unified = p.gpu.memory == loadmaster_hw::MemoryKind::Unified;
    if unified {
        fact(
            &mut out,
            "RAM",
            &format!(
                "{:.1} of {} free now",
                gib_f(p.host_need),
                gib1(p.host_budget)
            ),
        );
    } else if !p.placement.is_full(p.model.n_layer) {
        fact(
            &mut out,
            "CPU RAM",
            &format!(
                "{:.1} of {} available",
                gib_f(p.host_need),
                gib1(p.host_budget)
            ),
        );
    }
    fact(&mut out, "layers", &layers_line(p));
    if !is_encoder(p) {
        fact(&mut out, "KV cache", kv_label(p));
    }
    if let Some(v) = projector_line(p) {
        fact(&mut out, "projector", &v);
    }
    if let loadmaster_model::ModelKind::Embedding { pooling, .. } = p.model.kind {
        let v = match pooling.flag() {
            Some(f) => format!("{f} pooling"),
            None => "the model's own pooling".to_owned(),
        };
        fact(&mut out, "embeddings", &v);
    }
    if let Some(a) = p.speed.as_ref().and_then(|s| s.at.last()) {
        fact(
            &mut out,
            "speed",
            &format!(
                "~{} at {} context {}",
                crate::speed::band_text(a.tok_s_x10),
                tokens_k(a.fill),
                style.dim("(estimate)")
            ),
        );
    }

    if !p.warnings.is_empty() {
        writeln!(out).unwrap();
        for w in &p.warnings {
            writeln!(out, "{} {}", style.yellow("!"), w.short).unwrap();
        }
    }
    writeln!(out).unwrap();
    writeln!(out, "{}", style.bold("Run:")).unwrap();
    for line in command_lines(p, style) {
        writeln!(out, "  {line}").unwrap();
    }
    alternatives(&mut out, p, style, false);
    out
}

/// `all 80 on GPU`, `62 of 80 on GPU, 18 on CPU`, with any experts kept on the CPU.
fn layers_line(p: &Plan) -> String {
    let n = p.model.n_layer;
    let pl = &p.placement;
    let l = pl.gpu_layer_count(n);
    let experts = if pl.n_cpu_moe > 0 {
        format!(", experts of the first {} on CPU", pl.n_cpu_moe.min(n))
    } else {
        String::new()
    };
    if !pl.output_on_gpu(n) {
        "everything on CPU".to_owned()
    } else if l == n {
        format!("all {n} on GPU{experts}")
    } else {
        format!("{l} of {n} on GPU, {} on CPU{experts}", n - l)
    }
}

/// A vision projector's weights and encoder: where they go and what they take.
fn projector_line(p: &Plan) -> Option<String> {
    let sum = |terms: &[Term], keys: [&str; 2]| -> Option<u64> {
        let t: Vec<u64> = terms
            .iter()
            .filter(|t| keys.contains(&t.key))
            .map(|t| t.bytes)
            .collect();
        (!t.is_empty()).then(|| t.iter().sum())
    };
    if let Some(b) = sum(&p.footprint.gpu, ["mmproj", "vision_compute"]) {
        Some(format!("{} on GPU, one image", gib1(b)))
    } else {
        sum(&p.footprint.host, ["mmproj_cpu", "vision_compute_cpu"])
            .map(|b| format!("{} on CPU (--no-mmproj-offload)", gib1(b)))
    }
}

fn tokens_k(fill: u32) -> String {
    if fill >= 1024 && fill % 1024 == 0 {
        format!("{}k", fill / 1024)
    } else {
        fill.to_string()
    }
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

/// The nearby options under the command. The summary shows up to three that fit;
/// `--explain` shows every one with its notes.
fn alternatives(out: &mut String, p: &Plan, style: Style, full: bool) {
    let rows: Vec<&crate::Alternative> = if full {
        p.alternatives.iter().collect()
    } else {
        p.alternatives.iter().filter(|a| a.fits).take(3).collect()
    };
    if rows.is_empty() {
        return;
    }
    let size = |a: &crate::Alternative| {
        format!(
            "{}{:.1} GiB",
            if a.estimated { "~" } else { "" },
            gib_f(a.gpu_bytes)
        )
    };
    let speed = |a: &crate::Alternative| match (a.tok_s_x10, a.slowdown_x100) {
        (Some(t), _) => format!("~{}", crate::speed::band_text(t)),
        (None, Some((lo, hi))) => format!(
            "{:.1}-{:.1}x slower",
            f64::from(lo) / 100.0,
            f64::from(hi) / 100.0
        ),
        (None, None) if a.fits => "full GPU speed".to_owned(),
        (None, None) => String::new(),
    };
    let lw = rows
        .iter()
        .map(|a| a.label.chars().count())
        .max()
        .unwrap_or(0);
    let sw = rows.iter().map(|a| size(a).len()).max().unwrap_or(0);
    let pw = rows
        .iter()
        .map(|a| a.placement.chars().count())
        .max()
        .unwrap_or(0);
    writeln!(out).unwrap();
    writeln!(out, "{}", style.bold("Other options:")).unwrap();
    for a in rows {
        let line = format!(
            "  {:<lw$}  {:>sw$}  {:<pw$}  {}",
            a.label,
            size(a),
            a.placement,
            speed(a)
        );
        writeln!(out, "{}", line.trim_end()).unwrap();
        if full && let Some(note) = &a.note {
            writeln!(out, "  {:<lw$}  {}", "", style.dim(note)).unwrap();
        }
    }
    if !full && p.alternatives.iter().any(|a| a.estimated && a.fits) {
        writeln!(
            out,
            "  {}",
            style.dim("~ = size estimated; plan that file to confirm")
        )
        .unwrap();
    }
}

/// `--explain`: every term with its inputs, every choice with its reason.
pub fn explain(p: &Plan, style: Style) -> String {
    let mut out = String::new();
    writeln!(out, "WEIGHT   {}", model_line(&p.model)).unwrap();
    terms(&mut out, &p.footprint.gpu);
    writeln!(out, "  {}", "-".repeat(30)).unwrap();
    if p.gpus.is_empty() {
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
    } else {
        writeln!(
            out,
            "  {:<18}{:>11}   of {} usable across {} GPUs; each must fit its own:",
            "GPU total",
            gib(p.footprint.gpu_bytes),
            gib(p.gpus.iter().map(|g| g.usable_bytes).sum()),
            p.gpus.len()
        )
        .unwrap();
    }
    for ((d, g), i) in p.footprint.per_device.iter().zip(&p.gpus).zip(0..) {
        writeln!(
            out,
            "  {:<18}{:>11}   of {} usable on {} ({} layer slots{})",
            format!("  GPU{i}"),
            gib(d.bytes),
            gib(g.usable_bytes),
            g.name,
            d.layers,
            if i + 1 == p.gpus.len() {
                ", incl. output head"
            } else {
                ""
            }
        )
        .unwrap();
    }
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
    if is_encoder(p) {
        writeln!(
            out,
            "  KV cache: none, an encoder takes each input in one batch"
        )
        .unwrap();
    } else {
        writeln!(out, "  KV cache {}: {}", kv_label(p), p.kv_reason).unwrap();
    }
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
        writeln!(out, "  {} {w}", style.yellow("!")).unwrap();
    }
    if let Some(s) = &p.speed {
        speed_section(&mut out, s);
    }
    writeln!(out).unwrap();
    writeln!(out, "COMMAND").unwrap();
    for line in command_lines(p, style) {
        writeln!(out, "  {line}").unwrap();
    }
    alternatives(&mut out, p, style, true);
    out
}

pub fn no_fit(n: &NoFit, explain: bool, style: Style) -> String {
    let mut out = String::new();
    writeln!(out, "{}", header(&n.model, n.ctx, style)).unwrap();
    writeln!(out).unwrap();
    let gpus: Vec<&Gpu> = if n.gpus.is_empty() {
        vec![&n.gpu]
    } else {
        n.gpus.iter().collect()
    };
    let usable: u64 = gpus.iter().map(|g| g.usable_bytes).sum();
    let place = match (n.limit, n.gpu.memory) {
        ("gpu", _) => format!("{} ({} usable)", hw_label(&gpus), gib1(usable)),
        (_, loadmaster_hw::MemoryKind::Unified) => {
            format!("free RAM (GPU and CPU share {})", gib1(n.gpu.total_bytes))
        }
        _ => "host memory, even with every layer on the CPU".to_owned(),
    };
    writeln!(
        out,
        "{}",
        style.red(&format!(
            "✗ Doesn't fit: {} short on {place}",
            gib1(n.short_by)
        ))
    )
    .unwrap();
    if !n.fixes.is_empty() {
        writeln!(out).unwrap();
        writeln!(out, "{}", style.bold("Try:")).unwrap();
        for f in &n.fixes {
            writeln!(out, "  - {}", f.summary).unwrap();
        }
    }
    if explain {
        writeln!(out).unwrap();
        writeln!(
            out,
            "WEIGHT   {}, closest configuration",
            model_line(&n.model)
        )
        .unwrap();
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
