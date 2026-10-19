//! The embedding check (`probe --embeddings`, and `up --verify` for embedding models):
//! `/v1/embeddings` returns finite vectors of one length, the same text gives the same
//! vector every time, and a batch of two texts gives two different vectors.

use serde_json::{Value, json};

use crate::client::Client;
use crate::run::{FixtureResult, Outcome, PROBE_VERSION, Report, verdict};

const TEXT: &str = "loadmaster checks that this sentence embeds the same way every time.";

/// The vectors in an `/v1/embeddings` response, or the failure class.
fn vectors(status: u16, body: &Value) -> Result<Vec<Vec<f64>>, String> {
    if !(200..300).contains(&status) {
        return Err(format!("embed_http_error {status}"));
    }
    let data = body["data"]
        .as_array()
        .filter(|d| !d.is_empty())
        .ok_or("embed_no_vector")?;
    data.iter()
        .map(|d| -> Result<Vec<f64>, String> {
            let v: Vec<f64> = d["embedding"]
                .as_array()
                .ok_or("embed_no_vector")?
                .iter()
                .map(|x| x.as_f64().ok_or("embed_not_finite"))
                .collect::<Result<_, _>>()?;
            if v.is_empty() {
                Err("embed_no_vector".into())
            } else if v.iter().any(|x| !x.is_finite()) {
                Err("embed_not_finite".into())
            } else {
                Ok(v)
            }
        })
        .collect()
}

fn result(id: &str, description: &str, core: bool, runs: Vec<Outcome>) -> FixtureResult {
    let outcome = runs
        .iter()
        .find(|o| !matches!(o, Outcome::Pass))
        .cloned()
        .unwrap_or(Outcome::Pass);
    FixtureResult {
        id: id.into(),
        description: description.into(),
        core,
        outcome,
        runs,
        latency_ms: 0,
    }
}

pub async fn probe(c: &Client, model: &str, repeat: u32) -> Report {
    let detected = crate::detect::detect(c).await;
    let mut shape = Vec::new();
    let mut first: Option<Vec<f64>> = None;
    let mut same = Vec::new();
    for _ in 0..repeat.max(1) {
        let (status, body) = c
            .post("/v1/embeddings", json!({ "model": model, "input": TEXT }))
            .await;
        match vectors(status, &body) {
            Ok(mut v) => {
                shape.push(Outcome::Pass);
                let v = v.swap_remove(0);
                match &first {
                    None => {
                        first = Some(v);
                        same.push(Outcome::Pass);
                    }
                    Some(f) => same.push(
                        if f.len() == v.len() && f.iter().zip(&v).all(|(a, b)| (a - b).abs() < 1e-4)
                        {
                            Outcome::Pass
                        } else {
                            Outcome::Fail("embed_nondeterministic".into())
                        },
                    ),
                }
            }
            Err(class) => {
                shape.push(Outcome::Fail(class.clone()));
                same.push(Outcome::Skip("no vector".into()));
            }
        }
    }
    let (status, body) = c
        .post(
            "/v1/embeddings",
            json!({ "model": model, "input": [TEXT, "A completely different sentence about the sea."] }),
        )
        .await;
    let batch = match vectors(status, &body) {
        Ok(v) if v.len() == 2 && v[0].len() == v[1].len() && v[0] != v[1] => Outcome::Pass,
        Ok(v) if v.len() == 2 && v[0] == v[1] => Outcome::Fail("embed_identical".into()),
        Ok(_) => Outcome::Fail("embed_batch_shape".into()),
        Err(class) => Outcome::Fail(class),
    };
    let results = vec![
        result(
            "embed.shape",
            "one input returns one finite, non-empty vector",
            true,
            shape,
        ),
        result(
            "embed.deterministic",
            "the same input returns the same vector every time",
            true,
            same,
        ),
        result(
            "embed.batch",
            "two inputs return two vectors of the same length that differ",
            false,
            vec![batch],
        ),
    ];
    let mut failures: Vec<String> = results
        .iter()
        .filter_map(|r| match &r.outcome {
            Outcome::Fail(c) => Some(c.split(' ').next().unwrap_or(c).to_owned()),
            _ => None,
        })
        .collect();
    failures.sort();
    failures.dedup();
    Report {
        probe_version: PROBE_VERSION.into(),
        url: c.base().to_owned(),
        model: model.into(),
        template_blake3: None,
        quant: None,
        verdict: verdict(&results).into(),
        detected,
        results,
        failures,
        known_issues: Vec::new(),
    }
}
