//! Building a model from headers in memory (as fetched from the Hub) gives exactly what
//! reading the same files from disk gives.

use std::path::{Path, PathBuf};

use loadmaster_model::gguf::Gguf;
use loadmaster_model::safetensors::{self, HfParts, StHeader};
use loadmaster_model::{Mmproj, ModelInfo, mmproj_candidates, split_part_names};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

/// A GGUF header as the remote fetcher returns it, with the path and size the caller sets.
fn in_memory(path: &Path) -> Gguf {
    let bytes = std::fs::read(path).unwrap();
    let mut g = Gguf::parse_bytes(&bytes).unwrap();
    g.path = path.to_path_buf();
    g.file_size = bytes.len() as u64;
    g
}

#[test]
fn gguf_parts_match_disk() {
    for dir in std::fs::read_dir(fixtures().join("gguf")).unwrap() {
        let dir = dir.unwrap().path();
        let names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        let candidates = mmproj_candidates(names.iter().map(String::as_str));
        for name in names.iter().filter(|n| !candidates.contains(&n.as_str())) {
            let path = dir.join(name);
            let disk = ModelInfo::from_gguf(&path).unwrap();
            let mut mem = ModelInfo::from_gguf_parts(&path, &[in_memory(&path)]).unwrap();
            mem.mmproj = candidates.first().map(|c| {
                let p = dir.join(c);
                Mmproj::from_gguf(&in_memory(&p), &p).unwrap()
            });
            assert_eq!(mem, disk, "{name}");
        }
    }
}

#[test]
fn projector_and_part_count_are_checked() {
    let p = fixtures().join("gguf/gemma-3-12b/mmproj-gemma-3-12b-it-F16.gguf");
    assert!(ModelInfo::from_gguf_parts(&p, &[in_memory(&p)]).is_err());
    let m = fixtures().join("gguf/qwen3-8b/Qwen3-8B-Q4_K_M.gguf");
    assert!(ModelInfo::from_gguf_parts(&m, &[in_memory(&m), in_memory(&m)]).is_err());
    assert!(ModelInfo::from_gguf_parts(&m, &[]).is_err());
}

#[test]
fn split_names_and_projector_order() {
    assert_eq!(
        split_part_names("Q-UD-Q4_K_XL-00002-of-00003.gguf", 3).unwrap(),
        [
            "Q-UD-Q4_K_XL-00001-of-00003.gguf",
            "Q-UD-Q4_K_XL-00002-of-00003.gguf",
            "Q-UD-Q4_K_XL-00003-of-00003.gguf"
        ]
    );
    assert_eq!(split_part_names("model.gguf", 2), None);
    assert_eq!(
        mmproj_candidates(["a.gguf", "mmproj-Q8_0.gguf", "mmproj-F16.gguf", "x.txt"]),
        ["mmproj-F16.gguf", "mmproj-Q8_0.gguf"]
    );
}

#[test]
fn hf_parts_match_disk() {
    for dir in std::fs::read_dir(fixtures().join("hf")).unwrap() {
        let dir = dir.unwrap().path();
        let disk = ModelInfo::from_hf_dir(&dir).unwrap();
        let read = |n: &str| std::fs::read(dir.join(n)).ok();
        let json = |n: &str| read(n).map(|b| serde_json::from_slice(&b).unwrap());
        let config_path = dir.join("config.json");
        let shards = match json("model.safetensors.index.json") {
            Some(index) => safetensors::index_shards(&index),
            None => vec!["model.safetensors".into()],
        };
        let headers = shards
            .iter()
            .map(|s| {
                let p = dir.join(s);
                let bytes = std::fs::read(&p).unwrap();
                StHeader::from_bytes(&p, &bytes, bytes.len() as u64).unwrap()
            })
            .collect();
        let mem = ModelInfo::from_hf_parts(HfParts {
            path: dir.clone(),
            name: disk.name.clone(),
            config: json("config.json").unwrap(),
            config_path,
            headers,
            chat_template: safetensors::chat_template_from(
                read("chat_template.jinja").map(|b| String::from_utf8(b).unwrap()),
                json("tokenizer_config.json").as_ref(),
            ),
            pooling: safetensors::sentence_transformers_pooling(
                safetensors::SENTENCE_TRANSFORMERS_MARKERS
                    .iter()
                    .any(|m| dir.join(m).exists()),
                json("1_Pooling/config.json").as_ref(),
            ),
        })
        .unwrap();
        assert_eq!(mem, disk, "{}", dir.display());
    }
}

#[test]
fn safetensors_header_bytes_are_validated() {
    let p = Path::new("x.safetensors");
    assert!(StHeader::from_bytes(p, &[1, 2, 3], 3).is_err());
    let mut bytes = 100u64.to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(
        StHeader::from_bytes(p, &bytes, 1000).is_err(),
        "ends inside its header"
    );
}
