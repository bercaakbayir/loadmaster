//! `probe compare`: two eval runs side by side, with significance for the proportions.

use std::fmt::Write;

use crate::db::{RunMeta, Summary};

/// Two-sided p-value of a two-proportion z-test; 1.0 when undefined (no variance).
pub fn two_proportion_p(x1: u64, n1: u64, x2: u64, n2: u64) -> f64 {
    if n1 == 0 || n2 == 0 {
        return 1.0;
    }
    let (p1, p2) = (x1 as f64 / n1 as f64, x2 as f64 / n2 as f64);
    let p = (x1 + x2) as f64 / (n1 + n2) as f64;
    let se = (p * (1.0 - p) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    if se == 0.0 {
        return 1.0;
    }
    let z = ((p1 - p2) / se).abs();
    2.0 * (1.0 - normal_cdf(z))
}

/// Standard normal CDF via the Abramowitz-Stegun 7.1.26 erf approximation (error < 1.5e-7).
fn normal_cdf(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erf = 1.0 - poly * (-x * x).exp();
    let erf = if x >= 0.0 { erf } else { -erf };
    0.5 * (1.0 + erf)
}

fn pct(x: u64, n: u64) -> String {
    if n == 0 {
        "-".into()
    } else {
        format!("{:.1}%", 100.0 * x as f64 / n as f64)
    }
}

fn proportion_row(out: &mut String, label: &str, (x1, n1): (u64, u64), (x2, n2): (u64, u64)) {
    if n1 == 0 && n2 == 0 {
        return;
    }
    let delta = if n1 > 0 && n2 > 0 {
        format!(
            "{:+.1}pp",
            100.0 * (x2 as f64 / n2 as f64 - x1 as f64 / n1 as f64)
        )
    } else {
        "-".into()
    };
    let p = two_proportion_p(x1, n1, x2, n2);
    let sig = if p < 0.05 {
        format!("p={p:.3}")
    } else {
        format!("p={p:.2} n.s.")
    };
    writeln!(
        out,
        "{label:<16} {:>10} {:>10} {delta:>9}  {sig}",
        pct(x1, n1),
        pct(x2, n2)
    )
    .unwrap();
}

/// At most 20 lines.
pub fn render(
    a: &RunMeta,
    sa: &Summary,
    b: &RunMeta,
    sb: &Summary,
    changed_outputs: usize,
) -> String {
    let mut o = String::new();
    let who = |m: &RunMeta| {
        format!(
            "{} @ {}{}",
            m.model,
            m.runtime,
            m.runtime_version
                .as_ref()
                .map_or(String::new(), |v| format!(" {v}"))
        )
    };
    writeln!(o, "run {}: {}  ({})", a.id, who(a), a.created_at).unwrap();
    writeln!(o, "run {}: {}  ({})", b.id, who(b), b.created_at).unwrap();
    if a.prompts_blake3 != b.prompts_blake3 {
        writeln!(o, "warning: the runs used different prompt files").unwrap();
    }
    writeln!(
        o,
        "{:<16} {:>10} {:>10} {:>9}",
        "",
        format!("run {}", a.id),
        format!("run {}", b.id),
        "delta"
    )
    .unwrap();
    proportion_row(&mut o, "accuracy", (sa.ok, sa.n), (sb.ok, sb.n));
    proportion_row(
        &mut o,
        "schema valid",
        (sa.schema_ok, sa.schema_n),
        (sb.schema_ok, sb.schema_n),
    );
    let num = |v: Option<f64>| v.map_or("-".into(), |x| format!("{x:.1}"));
    let delta = match (sa.tokens_per_s_mean, sb.tokens_per_s_mean) {
        (Some(x), Some(y)) => format!("{:+.1}", y - x),
        _ => "-".into(),
    };
    writeln!(
        o,
        "{:<16} {:>10} {:>10} {delta:>9}",
        "tokens/s (mean)",
        num(sa.tokens_per_s_mean),
        num(sb.tokens_per_s_mean)
    )
    .unwrap();
    for (label, x, y) in [
        ("latency p50", sa.latency_p50_ms, sb.latency_p50_ms),
        ("latency p95", sa.latency_p95_ms, sb.latency_p95_ms),
    ] {
        writeln!(
            o,
            "{label:<16} {:>8}ms {:>8}ms {:>+7}ms",
            x,
            y,
            y as i64 - x as i64
        )
        .unwrap();
    }
    writeln!(
        o,
        "{changed_outputs} of {} responses differ between the runs",
        sa.n.min(sb.n)
    )
    .unwrap();
    writeln!(
        o,
        "n.s. = not significant (two-proportion z-test, p >= 0.05); timings are not tested"
    )
    .unwrap();
    o
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_test_matches_reference_values() {
        // 45/50 vs 30/50: z = 3.464, p = 0.000532 (checked with Python math.erf).
        let p = two_proportion_p(45, 50, 30, 50);
        assert!((p - 0.000_532).abs() < 0.000_001, "{p}");
        // 41/50 vs 43/50: clearly not significant.
        assert!(two_proportion_p(41, 50, 43, 50) > 0.5);
        assert_eq!(two_proportion_p(10, 10, 10, 10), 1.0);
    }
}
