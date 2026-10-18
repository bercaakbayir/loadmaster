//! LM Studio models directory: `<publisher>/<repo>/<file>.gguf`, or an HF-style directory.

use std::path::Path;

use loadmaster_model::Source;

use crate::{Scan, models_in_dir, sorted_dir};

pub fn scan(root: &Path, out: &mut Scan) {
    for publisher in sorted_dir(root) {
        if !publisher.is_dir() {
            continue;
        }
        let p = publisher
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        for repo_dir in sorted_dir(&publisher) {
            if !repo_dir.is_dir() {
                continue;
            }
            let r = repo_dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let repo = format!("{p}/{r}");
            let source = Source::LmStudio { repo: repo.clone() };
            models_in_dir(&repo_dir, &source, out, Some(repo.clone()), |file| {
                format!("{repo}/{file}")
            });
        }
    }
}
