//! Ollama store: `manifests/<host>/<namespace>/<model>/<tag>` JSON manifests naming
//! `blobs/sha256-<hex>` layers. The model layer is a GGUF; a projector layer, if present,
//! is the vision projector.

use std::path::{Path, PathBuf};

use loadmaster_model::Source;
use serde_json::Value;

use crate::{Entry, Issue, Scan, sorted_dir};

const MODEL: &str = "application/vnd.ollama.image.model";
const PROJECTOR: &str = "application/vnd.ollama.image.projector";

pub fn scan(root: &Path, out: &mut Scan) {
    for host in sorted_dir(&root.join("manifests")) {
        let host_name = host
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for ns in sorted_dir(&host) {
            let ns_name = ns
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            for model in sorted_dir(&ns) {
                let model_name = model
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                for tag in sorted_dir(&model) {
                    let tag_name = tag
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default();
                    let name = match (host_name.as_str(), ns_name.as_str()) {
                        ("registry.ollama.ai", "library") => format!("{model_name}:{tag_name}"),
                        ("registry.ollama.ai", _) => format!("{ns_name}/{model_name}:{tag_name}"),
                        _ => format!("{host_name}/{ns_name}/{model_name}:{tag_name}"),
                    };
                    manifest(root, &tag, name, out);
                }
            }
        }
    }
}

/// Blob path for a digest. Current Ollama writes `sha256-<hex>`; old versions `sha256:<hex>`.
fn blob(root: &Path, digest: &str) -> Option<PathBuf> {
    let dash = root.join("blobs").join(digest.replacen(':', "-", 1));
    if dash.exists() {
        return Some(dash);
    }
    let colon = root.join("blobs").join(digest);
    colon.exists().then_some(colon)
}

fn manifest(root: &Path, path: &Path, name: String, out: &mut Scan) {
    let issue = |problem: String| Issue {
        path: path.to_path_buf(),
        problem,
    };
    let v: Value = match std::fs::read_to_string(path).map(|t| serde_json::from_str(&t)) {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => return out.issues.push(issue(format!("manifest is not JSON: {e}"))),
        Err(e) => return out.issues.push(issue(format!("unreadable manifest: {e}"))),
    };
    // Schema 2 (current): `layers[].mediaType/digest`. Schema 1 (early Ollama) used
    // `fsLayers[].blobSum` without media types; the first GGUF blob is the model.
    let layers: Vec<(String, String)> = if let Some(layers) = v["layers"].as_array() {
        layers
            .iter()
            .filter_map(|l| {
                Some((
                    l["mediaType"].as_str()?.to_owned(),
                    l["digest"].as_str()?.to_owned(),
                ))
            })
            .collect()
    } else if let Some(fs) = v["fsLayers"].as_array() {
        fs.iter()
            .filter_map(|l| l["blobSum"].as_str())
            .filter(|d| blob(root, d).is_some_and(|p| is_gguf_blob(&p)))
            .take(1)
            .map(|d| (MODEL.to_owned(), d.to_owned()))
            .collect()
    } else {
        return out.issues.push(issue("manifest has no layers".into()));
    };
    let Some((_, digest)) = layers.iter().find(|(t, _)| t == MODEL) else {
        return; // embedding-only or adapter manifests have no model layer
    };
    let Some(model_blob) = blob(root, digest) else {
        return out.issues.push(issue(format!(
            "model blob {digest} is missing (partial pull?)"
        )));
    };
    let projector = layers
        .iter()
        .find(|(t, _)| t == PROJECTOR)
        .and_then(|(_, d)| blob(root, d));
    let source = Source::Ollama {
        name: name.clone(),
        digest: digest.clone(),
    };
    out.entries
        .push(Entry::new(name, model_blob, source, projector));
}

fn is_gguf_blob(p: &Path) -> bool {
    use std::io::Read;
    let mut magic = [0u8; 4];
    std::fs::File::open(p)
        .and_then(|mut f| f.read_exact(&mut magic))
        .is_ok()
        && &magic == b"GGUF"
}
