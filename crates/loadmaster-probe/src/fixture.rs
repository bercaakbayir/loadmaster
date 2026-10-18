//! Declarative probe fixtures: YAML with request steps, a pass condition and ordered
//! failure classes. Adding a fixture is adding a file; see `docs/fixtures.md`.

use serde_json::{Map, Value};
use yaml_rust2::{Yaml, YamlLoader};

#[derive(Debug, Clone, PartialEq)]
pub struct Fixture {
    pub id: String,
    pub description: String,
    pub kind: Kind,
    /// How many times to run; a fixture passes only if every run passes.
    pub repeat: u32,
    /// Counts toward a `partial` verdict.
    pub core: bool,
    pub steps: Vec<Step>,
    pub pass: Cond,
    pub fail: Vec<FailRule>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// Chat requests checked with conditions.
    Chat,
    /// Compare the runtime's chat template with the expected one.
    Template,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Step {
    pub stream: bool,
    /// Merged over `{model, temperature: 0, seed: 42, max_tokens: 512}`. Objects of the
    /// form `{"$from": "N:<jsonpath>"}` are replaced by that value from exchange N.
    pub body: Value,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FailRule {
    pub class: String,
    pub when: Cond,
}

/// Where a condition looks: exchange `N` (default: the last one) and a JSONPath.
#[derive(Debug, Clone, PartialEq)]
pub struct Ref {
    pub exchange: Option<usize>,
    pub path: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Cond {
    All(Vec<Cond>),
    Any(Vec<Cond>),
    Not(Box<Cond>),
    Path(Ref, Test),
    /// HTTP status: exact `code`, or the hundreds `class` (2 for 2xx).
    Status {
        exchange: Option<usize>,
        code: Option<u16>,
        class: Option<u16>,
    },
    /// The stream ended with `[DONE]`.
    StreamDone {
        exchange: Option<usize>,
        done: bool,
    },
    /// Both references resolve to equal values (strings holding JSON are compared parsed).
    Same(Ref, Ref),
    /// Exchange N happened (earlier steps produced what later ones needed).
    Ran(usize),
    Always,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Test {
    Exists(bool),
    Count {
        min: Option<u64>,
        max: Option<u64>,
    },
    Equals(Value),
    ContainsAny(Vec<String>),
    /// Every match is a string holding valid JSON (or not).
    Json(bool),
    /// Every match (parsed from a string when `parse`) validates against the schema.
    Schema {
        schema: Value,
        parse: bool,
    },
}

#[derive(Debug, thiserror::Error)]
#[error("fixture {file}: {message}")]
pub struct Error {
    pub file: String,
    pub message: String,
}

pub fn yaml_to_json(y: &Yaml) -> Value {
    match y {
        Yaml::Real(r) => r.parse::<f64>().map_or(Value::Null, |f| {
            serde_json::Number::from_f64(f).map_or(Value::Null, Value::Number)
        }),
        Yaml::Integer(i) => Value::from(*i),
        Yaml::String(s) => Value::String(s.clone()),
        Yaml::Boolean(b) => Value::Bool(*b),
        Yaml::Array(a) => Value::Array(a.iter().map(yaml_to_json).collect()),
        Yaml::Hash(h) => {
            let mut m = Map::new();
            for (k, v) in h {
                let key = match k {
                    Yaml::String(s) => s.clone(),
                    Yaml::Integer(i) => i.to_string(),
                    Yaml::Boolean(b) => b.to_string(),
                    _ => continue,
                };
                m.insert(key, yaml_to_json(v));
            }
            Value::Object(m)
        }
        _ => Value::Null,
    }
}

pub fn parse(file: &str, text: &str) -> Result<Fixture, Error> {
    let err = |message: String| Error {
        file: file.to_owned(),
        message,
    };
    let docs = YamlLoader::load_from_str(text).map_err(|e| err(format!("invalid YAML: {e}")))?;
    let v = docs.first().map(yaml_to_json).unwrap_or(Value::Null);
    let o = v
        .as_object()
        .ok_or_else(|| err("top level must be a mapping".into()))?;
    let known = [
        "id",
        "description",
        "kind",
        "repeat",
        "core",
        "steps",
        "pass",
        "fail",
    ];
    if let Some(k) = o.keys().find(|k| !known.contains(&k.as_str())) {
        return Err(err(format!("unknown key `{k}`")));
    }
    let id = o
        .get("id")
        .and_then(Value::as_str)
        .ok_or_else(|| err("missing `id`".into()))?
        .to_owned();
    let kind = match o.get("kind").and_then(Value::as_str).unwrap_or("chat") {
        "chat" => Kind::Chat,
        "template" => Kind::Template,
        k => return Err(err(format!("unknown kind `{k}`"))),
    };
    let steps = o
        .get("steps")
        .and_then(Value::as_array)
        .map(|s| {
            s.iter()
                .map(|st| {
                    Ok(Step {
                        stream: st["stream"].as_bool().unwrap_or(false),
                        body: st
                            .get("body")
                            .cloned()
                            .ok_or_else(|| err("a step needs a `body`".into()))?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?
        .unwrap_or_default();
    if kind == Kind::Chat && steps.is_empty() {
        return Err(err("a chat fixture needs `steps`".into()));
    }
    let pass = match o.get("pass") {
        Some(c) => cond(c).map_err(|m| err(format!("pass: {m}")))?,
        None => Cond::Always,
    };
    let fail = o
        .get("fail")
        .and_then(Value::as_array)
        .map(|rules| {
            rules
                .iter()
                .map(|r| {
                    let class = r["class"]
                        .as_str()
                        .ok_or_else(|| err("a fail rule needs `class`".into()))?
                        .to_owned();
                    let when = match r.get("if") {
                        Some(c) => cond(c).map_err(|m| err(format!("fail {class}: {m}")))?,
                        None => Cond::Always,
                    };
                    Ok(FailRule { class, when })
                })
                .collect::<Result<Vec<_>, Error>>()
        })
        .transpose()?
        .unwrap_or_default();
    Ok(Fixture {
        id,
        description: o
            .get("description")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned(),
        kind,
        repeat: o.get("repeat").and_then(Value::as_u64).unwrap_or(3) as u32,
        core: o.get("core").and_then(Value::as_bool).unwrap_or(false),
        steps,
        pass,
        fail,
    })
}

/// `"1:$.a.b"` or `"$.a.b"`.
pub fn parse_ref(s: &str) -> Result<Ref, String> {
    let (exchange, path) = match s.split_once(':') {
        Some((n, p)) if n.bytes().all(|b| b.is_ascii_digit()) && !n.is_empty() => (
            Some(n.parse().map_err(|_| format!("bad exchange in `{s}`"))?),
            p,
        ),
        _ => (None, s),
    };
    serde_json_path::JsonPath::parse(path).map_err(|e| format!("bad JSONPath `{path}`: {e}"))?;
    Ok(Ref {
        exchange,
        path: path.to_owned(),
    })
}

fn exchange_of(o: &Map<String, Value>) -> Option<usize> {
    o.get("exchange")
        .and_then(Value::as_u64)
        .map(|n| n as usize)
}

fn cond(v: &Value) -> Result<Cond, String> {
    let o = v.as_object().ok_or("a condition must be a mapping")?;
    let list = |k: &str| -> Result<Vec<Cond>, String> {
        o[k].as_array()
            .ok_or(format!("`{k}` takes a list"))?
            .iter()
            .map(cond)
            .collect()
    };
    if o.contains_key("all") {
        return Ok(Cond::All(list("all")?));
    }
    if o.contains_key("any") {
        return Ok(Cond::Any(list("any")?));
    }
    if let Some(n) = o.get("not") {
        return Ok(Cond::Not(Box::new(cond(n)?)));
    }
    if let Some(n) = o.get("ran").and_then(Value::as_u64) {
        return Ok(Cond::Ran(n as usize));
    }
    if o.contains_key("status") || o.contains_key("status_class") {
        return Ok(Cond::Status {
            exchange: exchange_of(o),
            code: o.get("status").and_then(Value::as_u64).map(|c| c as u16),
            class: o
                .get("status_class")
                .and_then(Value::as_u64)
                .map(|c| c as u16),
        });
    }
    if let Some(d) = o.get("stream_done").and_then(Value::as_bool) {
        return Ok(Cond::StreamDone {
            exchange: exchange_of(o),
            done: d,
        });
    }
    if let Some(pair) = o.get("same").and_then(Value::as_array) {
        let [a, b] = pair.as_slice() else {
            return Err("`same` takes two references".into());
        };
        return Ok(Cond::Same(
            parse_ref(a.as_str().ok_or("`same` references are strings")?)?,
            parse_ref(b.as_str().ok_or("`same` references are strings")?)?,
        ));
    }
    let path = o.get("path").and_then(Value::as_str).ok_or(
        "unknown condition (expected path, status, stream_done, same, ran, all, any or not)",
    )?;
    let r = parse_ref(path)?;
    let test = if let Some(b) = o.get("exists").and_then(Value::as_bool) {
        Test::Exists(b)
    } else if let Some(n) = o.get("count").and_then(Value::as_u64) {
        Test::Count {
            min: Some(n),
            max: Some(n),
        }
    } else if o.contains_key("count_min") || o.contains_key("count_max") {
        Test::Count {
            min: o.get("count_min").and_then(Value::as_u64),
            max: o.get("count_max").and_then(Value::as_u64),
        }
    } else if let Some(e) = o.get("equals") {
        Test::Equals(e.clone())
    } else if let Some(c) = o.get("contains_any").and_then(Value::as_array) {
        Test::ContainsAny(
            c.iter()
                .filter_map(|s| s.as_str().map(str::to_owned))
                .collect(),
        )
    } else if let Some(j) = o.get("json").and_then(Value::as_bool) {
        Test::Json(j)
    } else if let Some(s) = o.get("schema") {
        jsonschema::validator_for(s).map_err(|e| format!("invalid schema: {e}"))?;
        Test::Schema {
            schema: s.clone(),
            parse: o.get("parse").and_then(Value::as_bool).unwrap_or(false),
        }
    } else {
        return Err(format!(
            "`{path}`: no test (exists, count, count_min, count_max, equals, contains_any, json, schema)"
        ));
    };
    Ok(Cond::Path(r, test))
}

/// The built-in fixtures, embedded at build time from `fixtures/` (every `*.yaml`).
pub fn builtin() -> Vec<Fixture> {
    include!(concat!(env!("OUT_DIR"), "/fixtures.rs"))
        .iter()
        .map(|(name, text)| parse(name, text).unwrap_or_else(|e| panic!("built-in {e}")))
        .collect()
}

/// Fixtures from a directory (every `*.yaml`, sorted by name).
pub fn load_dir(dir: &std::path::Path) -> Result<Vec<Fixture>, Error> {
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .map_err(|e| Error {
            file: dir.display().to_string(),
            message: e.to_string(),
        })?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml" || x == "yml"))
        .collect();
    files.sort();
    files
        .iter()
        .map(|p| {
            let text = std::fs::read_to_string(p).map_err(|e| Error {
                file: p.display().to_string(),
                message: e.to_string(),
            })?;
            parse(&p.display().to_string(), &text)
        })
        .collect()
}
