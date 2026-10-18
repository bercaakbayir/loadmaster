//! Checks against a real model file. Run with
//! `LOADMASTER_REAL_GGUF=/path/model.gguf cargo test --release -p loadmaster-model -- --ignored`.

use std::time::Instant;

use loadmaster_model::ModelInfo;
use loadmaster_model::gguf::Gguf;

#[test]
#[ignore = "needs LOADMASTER_REAL_GGUF"]
fn real_gguf_header() {
    let path = std::env::var("LOADMASTER_REAL_GGUF").expect("set LOADMASTER_REAL_GGUF");
    let start = Instant::now();
    let info = ModelInfo::from_gguf(&path).unwrap();
    let elapsed = start.elapsed();

    let g = Gguf::open(&path).unwrap();
    let payload = g.file_size - g.header_bytes();
    let diff = payload.abs_diff(info.weight_bytes) as f64 / payload as f64;
    eprintln!(
        "{}: {} {} layers={} weight_bytes={} file-header={} diff={:.5}% parsed in {:?}",
        info.name.as_deref().unwrap_or("?"),
        info.arch,
        info.quant,
        info.n_layer,
        info.weight_bytes,
        payload,
        diff * 100.0,
        elapsed
    );
    assert!(info.complete);
    assert!(diff < 0.001, "weight_bytes off by {:.4}%", diff * 100.0);
    assert!(elapsed.as_millis() < 50, "parse took {elapsed:?}");
}
