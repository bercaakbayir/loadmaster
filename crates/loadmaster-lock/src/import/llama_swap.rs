//! `loadmaster import llama-swap <config.yaml>`: turn a llama-swap config into lockfile
//! entries. Footprints are filled afterwards by planning each model file (in the CLI).
//!
//! Format reference: llama-swap `docs/config.example.yaml`. Supported: `models.<id>.cmd`
//! (llama-server or `vllm serve`), `macros` (global and per-model, `${env.X}`,
//! `${MODEL_ID}`), `aliases`, `ttl`, `globalTTL`, `env`. Everything else is reported.

use std::path::PathBuf;

use yaml_rust2::{Yaml, YamlLoader};

use crate::Model;

#[derive(Debug, Clone, PartialEq)]
pub struct Imported {
    pub models: Vec<ImportedModel>,
    pub skipped: Vec<Skipped>,
    /// Settings with no loadmaster equivalent, by model or top-level key.
    pub warnings: Vec<String>,
    /// `globalTTL` in seconds, if set.
    pub global_ttl_s: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ImportedModel {
    pub model: Model,
    /// The model file named in `cmd`, to plan the footprint from.
    pub model_path: PathBuf,
    /// Whether `cmd` set the context; if not, the planner picks it.
    pub ctx_given: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Skipped {
    pub id: String,
    pub reason: String,
}

/// Top-level keys loadmaster has no equivalent for (reported once if present).
const UNSUPPORTED_TOP: &[&str] = &[
    "groups",
    "routing",
    "matrix",
    "hooks",
    "peers",
    "profiles",
    "selectors",
    "apiKeys",
    "security",
];
/// Per-model keys reported as not carried over.
const UNSUPPORTED_MODEL: &[&str] = &[
    "filters",
    "useModelName",
    "checkEndpoint",
    "unlisted",
    "concurrencyLimit",
    "cmdStop",
    "proxy",
    "timeouts",
    "capabilities",
];

pub fn parse(yaml: &str, env: impl Fn(&str) -> Option<String>) -> Result<Imported, String> {
    let docs = YamlLoader::load_from_str(yaml).map_err(|e| format!("not valid YAML: {e}"))?;
    let root = docs.into_iter().next().unwrap_or(Yaml::Null);
    let models = match &root["models"] {
        Yaml::Hash(h) => h,
        Yaml::BadValue | Yaml::Null => return Err("no `models` section".into()),
        _ => return Err("`models` must be a mapping of model IDs".into()),
    };
    let global_macros = macros(&root["macros"]);
    let mut out = Imported {
        models: Vec::new(),
        skipped: Vec::new(),
        warnings: Vec::new(),
        global_ttl_s: root["globalTTL"]
            .as_i64()
            .and_then(|t| u64::try_from(t).ok()),
    };
    for key in UNSUPPORTED_TOP {
        if !root[*key].is_badvalue() {
            out.warnings.push(format!(
                "`{key}` is not imported; loadmaster schedules by memory footprint instead"
            ));
        }
    }

    for (id, m) in models {
        let Some(id) = id.as_str().map(str::to_owned) else {
            continue;
        };
        let skip = |reason: String| Skipped {
            id: id.clone(),
            reason,
        };
        let Some(cmd) = m["cmd"].as_str() else {
            out.skipped.push(skip("no `cmd`".into()));
            continue;
        };
        let mut vars = global_macros.clone();
        for (k, v) in macros(&m["macros"]) {
            vars.retain(|(gk, _)| *gk != k);
            vars.push((k, v));
        }
        vars.push(("MODEL_ID".into(), id.clone()));
        let expanded = match expand(&strip_comments(cmd), &vars, &env) {
            Ok(s) => s,
            Err(e) => {
                out.skipped.push(skip(e));
                continue;
            }
        };
        let argv = match split(&expanded) {
            Ok(a) if !a.is_empty() => a,
            Ok(_) => {
                out.skipped.push(skip("`cmd` is empty".into()));
                continue;
            }
            Err(e) => {
                out.skipped.push(skip(format!("cannot split `cmd`: {e}")));
                continue;
            }
        };
        let launch = match parse_cmd(&argv) {
            Ok(l) => l,
            Err(reason) => {
                out.skipped.push(skip(reason));
                continue;
            }
        };
        for key in UNSUPPORTED_MODEL {
            if !m[*key].is_badvalue() {
                out.warnings
                    .push(format!("{id}: `{key}` is not carried over"));
            }
        }
        let env_list: Vec<String> = m["env"]
            .as_vec()
            .map(|v| {
                v.iter()
                    .filter_map(|e| e.as_str())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let aliases: Vec<String> = m["aliases"]
            .as_vec()
            .map(|v| {
                v.iter()
                    .filter_map(|a| a.as_str())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let ttl = m["ttl"].as_i64().and_then(|t| u64::try_from(t).ok());
        out.models.push(ImportedModel {
            model: Model {
                name: id.clone(),
                source: launch.model.clone(),
                sha256: None,
                runtime: launch.runtime.into(),
                runtime_bin: launch.bin,
                ctx: launch.ctx.unwrap_or(4096),
                footprint_gb: None,
                flags: launch.flags,
                mmproj: launch.mmproj,
                env: env_list,
                keep_warm: false,
                aliases,
                idle_unload_s: ttl,
                verify: None,
            },
            model_path: PathBuf::from(launch.model),
            ctx_given: launch.ctx.is_some(),
        });
    }
    Ok(out)
}

fn macros(y: &Yaml) -> Vec<(String, String)> {
    let Yaml::Hash(h) = y else { return Vec::new() };
    h.iter()
        .filter_map(|(k, v)| {
            let v = match v {
                Yaml::String(s) => s.trim().to_owned(),
                Yaml::Integer(i) => i.to_string(),
                Yaml::Real(r) => r.clone(),
                Yaml::Boolean(b) => b.to_string(),
                _ => return None,
            };
            Some((k.as_str()?.to_owned(), v))
        })
        .collect()
}

/// llama-swap drops `#` comment lines from `cmd` before running it.
fn strip_comments(cmd: &str) -> String {
    cmd.lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Expand `${name}` from `vars`, and `${env.NAME}` from the environment, until stable.
/// `${PORT}` is left for the supervisor, which assigns ports itself.
fn expand(
    s: &str,
    vars: &[(String, String)],
    env: &impl Fn(&str) -> Option<String>,
) -> Result<String, String> {
    let mut cur = s.to_owned();
    for _ in 0..16 {
        let mut next = String::with_capacity(cur.len());
        let mut rest = cur.as_str();
        let mut changed = false;
        while let Some(start) = rest.find("${") {
            next.push_str(&rest[..start]);
            let Some(len) = rest[start..].find('}') else {
                return Err(format!("unterminated `${{` in `{s}`"));
            };
            let name = &rest[start + 2..start + len];
            let value = if name == "PORT" {
                None
            } else if let Some(var) = name.strip_prefix("env.") {
                Some(env(var).ok_or_else(|| format!("environment variable `{var}` is not set"))?)
            } else {
                Some(
                    vars.iter()
                        .find(|(k, _)| k == name)
                        .map(|(_, v)| v.clone())
                        .ok_or_else(|| format!("unknown macro `${{{name}}}`"))?,
                )
            };
            match value {
                Some(v) => {
                    next.push_str(&v);
                    changed = true;
                }
                None => next.push_str(&rest[start..start + len + 1]),
            }
            rest = &rest[start + len + 1..];
        }
        next.push_str(rest);
        if !changed {
            return Ok(next);
        }
        cur = next;
    }
    Err("macros expand recursively without end".into())
}

/// POSIX-style word splitting: whitespace, single and double quotes, backslash escapes.
pub fn split(s: &str) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        match c {
            c if c.is_whitespace() => {
                if in_word {
                    out.push(std::mem::take(&mut cur));
                    in_word = false;
                }
            }
            '\'' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('\'') => break,
                        Some(c) => cur.push(c),
                        None => return Err("unterminated single quote".into()),
                    }
                }
            }
            '"' => {
                in_word = true;
                loop {
                    match chars.next() {
                        Some('"') => break,
                        Some('\\') => match chars.next() {
                            Some(c @ ('"' | '\\' | '$' | '`')) => cur.push(c),
                            Some(c) => {
                                cur.push('\\');
                                cur.push(c);
                            }
                            None => return Err("trailing backslash".into()),
                        },
                        Some(c) => cur.push(c),
                        None => return Err("unterminated double quote".into()),
                    }
                }
            }
            '\\' => {
                in_word = true;
                match chars.next() {
                    Some(c) => cur.push(c),
                    None => return Err("trailing backslash".into()),
                }
            }
            c => {
                in_word = true;
                cur.push(c);
            }
        }
    }
    if in_word {
        out.push(cur);
    }
    Ok(out)
}

struct Launch {
    runtime: &'static str,
    bin: String,
    model: String,
    ctx: Option<u32>,
    mmproj: Option<String>,
    flags: Vec<String>,
}

fn parse_cmd(argv: &[String]) -> Result<Launch, String> {
    let bin = argv[0].clone();
    let base = std::path::Path::new(&bin)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let (runtime, rest, mut model) = if base.starts_with("llama-server") {
        ("llama.cpp", &argv[1..], None)
    } else if base == "vllm" && argv.get(1).is_some_and(|a| a == "serve") {
        let model = argv.get(2).filter(|m| !m.starts_with('-')).cloned();
        ("vllm", &argv[2 + usize::from(model.is_some())..], model)
    } else {
        return Err(format!(
            "`{base}` is not llama-server or `vllm serve`; loadmaster runs those itself (containers and other servers are not imported)"
        ));
    };
    let mut ctx = None;
    let mut mmproj = None;
    let mut flags = Vec::new();
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].as_str();
        // `--flag=value` form.
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f, Some(v.to_owned())),
            _ => (a, None),
        };
        let mut value = || -> Result<String, String> {
            match &inline {
                Some(v) => Ok(v.clone()),
                None => {
                    i += 1;
                    rest.get(i)
                        .cloned()
                        .ok_or_else(|| format!("`{flag}` needs a value"))
                }
            }
        };
        match flag {
            "-m" | "--model" => model = Some(value()?),
            "-c" | "--ctx-size" | "--max-model-len" => {
                let v = value()?;
                ctx = Some(
                    v.parse()
                        .map_err(|_| format!("context `{v}` is not a number"))?,
                );
            }
            "-mm" | "--mmproj" => mmproj = Some(value()?),
            "--port" | "--host" => {
                value()?;
            }
            _ => flags.push(rest[i].clone()),
        }
        i += 1;
    }
    let model = model.ok_or("`cmd` names no model (`--model`/`-m`)")?;
    Ok(Launch {
        runtime,
        bin,
        model,
        ctx: ctx.filter(|&c| c > 0),
        mmproj,
        flags,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_like_a_shell() {
        assert_eq!(
            split(r#"a 'b c' "d \"e\"" f\ g"#).unwrap(),
            ["a", "b c", "d \"e\"", "f g"]
        );
        assert!(split("'open").is_err());
    }

    #[test]
    fn expands_nested_macros_and_keeps_port() {
        let vars = vec![
            ("ctx".to_owned(), "4096".to_owned()),
            ("args".to_owned(), "--ctx-size ${ctx}".to_owned()),
        ];
        let env = |k: &str| (k == "HOME").then(|| "/home/u".to_owned());
        assert_eq!(
            expand("x ${args} --port ${PORT} ${env.HOME}/m", &vars, &env).unwrap(),
            "x --ctx-size 4096 --port ${PORT} /home/u/m"
        );
        assert!(
            expand("${nope}", &vars, &env)
                .unwrap_err()
                .contains("unknown macro")
        );
    }
}
