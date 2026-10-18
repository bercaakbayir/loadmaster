//! Condition evaluation over captured exchanges. Pure: no I/O, clock or randomness.

use serde_json::Value;

use crate::client::Exchange;
use crate::fixture::{Cond, Ref, Test};

fn exchange(xs: &[Exchange], n: Option<usize>) -> Option<&Exchange> {
    match n {
        Some(i) => xs.get(i),
        None => xs.last(),
    }
}

/// JSONPath matches in an exchange's (reassembled) response.
pub fn query(xs: &[Exchange], r: &Ref) -> Vec<Value> {
    let Some(x) = exchange(xs, r.exchange) else {
        return Vec::new();
    };
    match serde_json_path::JsonPath::parse(&r.path) {
        Ok(p) => p.query(&x.response).all().into_iter().cloned().collect(),
        Err(_) => Vec::new(),
    }
}

/// Strings holding JSON compare by their parsed value; other values as they are.
fn as_json(v: &Value) -> Value {
    match v {
        Value::String(s) => serde_json::from_str(s).unwrap_or_else(|_| v.clone()),
        other => other.clone(),
    }
}

pub fn eval(c: &Cond, xs: &[Exchange]) -> bool {
    match c {
        Cond::Always => true,
        Cond::All(cs) => cs.iter().all(|c| eval(c, xs)),
        Cond::Any(cs) => cs.iter().any(|c| eval(c, xs)),
        Cond::Not(c) => !eval(c, xs),
        Cond::Ran(n) => xs.len() > *n,
        Cond::Status {
            exchange: n,
            code,
            class,
        } => exchange(xs, *n).is_some_and(|x| {
            code.is_none_or(|c| x.status == c) && class.is_none_or(|k| x.status / 100 == k)
        }),
        Cond::StreamDone { exchange: n, done } => {
            exchange(xs, *n).is_some_and(|x| x.stream.as_ref().is_some_and(|s| s.done == *done))
        }
        Cond::Same(a, b) => {
            let (va, vb) = (query(xs, a), query(xs, b));
            !va.is_empty()
                && va.len() == vb.len()
                && va
                    .iter()
                    .zip(&vb)
                    .all(|(x, y)| !x.is_null() && as_json(x) == as_json(y))
        }
        Cond::Path(r, t) => {
            let ms = query(xs, r);
            match t {
                Test::Exists(want) => ms.iter().any(|v| !v.is_null()) == *want,
                Test::Count { min, max } => {
                    let n = match ms.as_slice() {
                        [Value::Array(a)] => a.len() as u64,
                        [Value::Null] | [] => 0,
                        many => many.len() as u64,
                    };
                    min.is_none_or(|m| n >= m) && max.is_none_or(|m| n <= m)
                }
                Test::Equals(v) => ms.first() == Some(v),
                Test::ContainsAny(needles) => ms
                    .iter()
                    .filter_map(Value::as_str)
                    .any(|s| needles.iter().any(|n| s.contains(n.as_str()))),
                Test::Json(want) => {
                    !ms.is_empty()
                        && ms.iter().all(|v| {
                            v.as_str()
                                .and_then(|s| serde_json::from_str::<Value>(s).ok())
                                .is_some()
                        }) == *want
                }
                Test::Schema { schema, parse } => {
                    let Ok(validator) = jsonschema::validator_for(schema) else {
                        return false;
                    };
                    !ms.is_empty()
                        && ms.iter().all(|v| {
                            let inst = if *parse {
                                v.as_str().and_then(|s| serde_json::from_str(s).ok())
                            } else {
                                Some(v.clone())
                            };
                            inst.is_some_and(|i| validator.is_valid(&i))
                        })
                }
            }
        }
    }
}
