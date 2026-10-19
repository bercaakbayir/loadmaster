//! `cargo xtask fetch-headers`: download only the GGUF header of each model listed in
//! `validation/*.toml` (HTTP range requests through `curl`), so validation runs anywhere
//! without multi-GB model files. Headers land in `validation/headers/<sha256>.gguf`
//! (git-ignored).

use std::path::Path;
use std::process::Command;

use anyhow::Context;
use loadmaster_store::remote::Error;

pub fn fetch_all(root: &Path) -> anyhow::Result<()> {
    let out_dir = root.join("validation/headers");
    std::fs::create_dir_all(&out_dir)?;
    let mut wanted = Vec::new();
    for file in ["validation/observed.toml", "validation/cases.toml"] {
        let Ok(text) = std::fs::read_to_string(root.join(file)) else {
            continue;
        };
        let v: toml::Value = toml::from_str(&text).with_context(|| file.to_owned())?;
        for key in ["model", "case"] {
            for m in v.get(key).and_then(|m| m.as_array()).into_iter().flatten() {
                if let (Some(sha), Some(url)) = (
                    m.get("sha256").and_then(|s| s.as_str()),
                    m.get("header_url").and_then(|s| s.as_str()),
                ) {
                    wanted.push((sha.to_owned(), url.to_owned()));
                }
            }
        }
    }
    wanted.sort();
    wanted.dedup();
    for (sha, url) in wanted {
        let dest = out_dir.join(format!("{sha}.gguf"));
        if dest.exists() {
            println!("{sha}: have it");
            continue;
        }
        let bytes = fetch_header(&url)?;
        std::fs::write(&dest, &bytes)?;
        println!("{sha}: {} KiB header from {url}", bytes.len() / 1024);
    }
    Ok(())
}

/// The GGUF header at `url`, fetched by growing range requests through `curl`.
fn fetch_header(url: &str) -> anyhow::Result<Vec<u8>> {
    Ok(loadmaster_store::remote::fetch_gguf_header(
        url,
        |url, n| {
            let out = Command::new("curl")
                .args(["-sSfL", "-r", &format!("0-{}", n - 1), url])
                .output()
                .map_err(|e| Error::Http {
                    url: url.to_owned(),
                    reason: format!("running curl: {e}"),
                })?;
            if !out.status.success() {
                return Err(Error::Http {
                    url: url.to_owned(),
                    reason: String::from_utf8_lossy(&out.stderr).trim().to_owned(),
                });
            }
            Ok(out.stdout)
        },
    )?)
}
