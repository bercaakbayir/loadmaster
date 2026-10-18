//! Canonical TOML for `models.lock`: fixed key order, one blank line between sections,
//! inline arrays, floats with at most two decimals.

use std::fmt::Write;

use crate::{Lockfile, Model};

const HEADER: &str = "# models.lock, written by loadmaster. Commit it. Plan-derived fields (ctx, footprint_gb,\n# flags) are frozen; change them with `loadmaster plan --update <name>`.\n";

pub fn to_toml(l: &Lockfile) -> String {
    let mut o = String::from(HEADER);
    writeln!(o, "version = {}", l.version).unwrap();
    writeln!(o, "\n[host]").unwrap();
    kv_str(&mut o, "gpu", &l.host.gpu);
    kv_raw(&mut o, "vram_gb", &float(l.host.vram_gb));
    kv_raw(&mut o, "headroom_gb", &float(l.host.headroom_gb));
    writeln!(o, "\n[proxy]").unwrap();
    kv_str(&mut o, "listen", &l.proxy.listen);
    kv_raw(&mut o, "idle_unload_s", &l.proxy.idle_unload_s.to_string());
    kv_raw(
        &mut o,
        "load_timeout_s",
        &l.proxy.load_timeout_s.to_string(),
    );
    for m in &l.models {
        model(&mut o, m);
    }
    o
}

fn model(o: &mut String, m: &Model) {
    writeln!(o, "\n[[model]]").unwrap();
    kv_str(o, "name", &m.name);
    kv_str(o, "source", &m.source);
    if let Some(s) = &m.sha256 {
        kv_str(o, "sha256", s);
    }
    kv_str(o, "runtime", &m.runtime);
    kv_str(o, "runtime_bin", &m.runtime_bin);
    kv_raw(o, "ctx", &m.ctx.to_string());
    if let Some(f) = m.footprint_gb {
        kv_raw(o, "footprint_gb", &float(f));
    }
    kv_raw(o, "flags", &array(&m.flags));
    if let Some(p) = &m.mmproj {
        kv_str(o, "mmproj", p);
    }
    if !m.env.is_empty() {
        kv_raw(o, "env", &array(&m.env));
    }
    kv_raw(o, "keep_warm", &m.keep_warm.to_string());
    kv_raw(o, "aliases", &array(&m.aliases));
    if let Some(t) = m.idle_unload_s {
        kv_raw(o, "idle_unload_s", &t.to_string());
    }
    if let Some(v) = &m.verify {
        writeln!(o, "\n[model.verify]").unwrap();
        kv_str(o, "probe_version", &v.probe_version);
        kv_str(o, "verdict", &v.verdict);
        kv_raw(o, "failures", &array(&v.failures));
        if let Some(r) = &v.runtime_version {
            kv_str(o, "runtime_version", r);
        }
        kv_str(o, "checked_at", &v.checked_at);
    }
}

fn kv_str(o: &mut String, k: &str, v: &str) {
    writeln!(o, "{k} = {}", string(v)).unwrap();
}

fn kv_raw(o: &mut String, k: &str, v: &str) {
    writeln!(o, "{k} = {v}").unwrap();
}

/// A TOML basic string.
fn string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if c.is_control() => write!(out, "\\u{:04X}", c as u32).unwrap(),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn array(items: &[String]) -> String {
    let inner: Vec<String> = items.iter().map(|s| string(s)).collect();
    format!("[{}]", inner.join(", "))
}

/// Rounded to hundredths, shortest form, always with a decimal point.
fn float(x: f64) -> String {
    let r = (x * 100.0).round() / 100.0;
    let s = format!("{r}");
    if s.contains('.') || s.contains('e') || !r.is_finite() {
        s
    } else {
        format!("{s}.0")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floats() {
        assert_eq!(float(24.0), "24.0");
        assert_eq!(float(21.4), "21.4");
        assert_eq!(float(0.781), "0.78");
    }

    #[test]
    fn strings_escape() {
        assert_eq!(string("a\"b\\c\nd"), r#""a\"b\\c\nd""#);
    }
}
