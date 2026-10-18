//! Hugging Face hub cache: `models--<org>--<name>/{refs,snapshots,blobs}`. Snapshot files
//! are symlinks into `blobs/`; a revision is listed if a ref points at it, otherwise the
//! newest snapshot by name.

use std::path::Path;

use loadmaster_model::Source;

use crate::{Issue, Scan, models_in_dir, sorted_dir};

pub fn scan(root: &Path, out: &mut Scan) {
    for repo_dir in sorted_dir(root) {
        let Some(dir_name) = repo_dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
        else {
            continue;
        };
        let Some(rest) = dir_name.strip_prefix("models--") else {
            continue;
        };
        let repo = rest.replacen("--", "/", 1);
        let mut revisions: Vec<String> = sorted_dir(&repo_dir.join("refs"))
            .iter()
            .filter_map(|r| std::fs::read_to_string(r).ok())
            .map(|s| s.trim().to_owned())
            .filter(|rev| repo_dir.join("snapshots").join(rev).is_dir())
            .collect();
        revisions.sort();
        revisions.dedup();
        if revisions.is_empty()
            && let Some(last) = sorted_dir(&repo_dir.join("snapshots")).last()
        {
            revisions.push(
                last.file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default(),
            );
        }
        if revisions.is_empty() {
            out.issues.push(Issue {
                path: repo_dir.clone(),
                problem: "no snapshots (download never completed?)".into(),
            });
        }
        for rev in revisions {
            let snap = repo_dir.join("snapshots").join(&rev);
            let source = Source::HfCache {
                repo: repo.clone(),
                revision: rev.clone(),
            };
            scan_snapshot(&snap, &snap, &repo, &source, out);
        }
    }
}

/// GGUF files can sit in subfolders of a snapshot (e.g. `Q4_K_M/`).
fn scan_snapshot(snap: &Path, dir: &Path, repo: &str, source: &Source, out: &mut Scan) {
    let hf_name = (dir == snap).then(|| repo.to_owned());
    let prefix = dir
        .strip_prefix(snap)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();
    models_in_dir(dir, source, out, hf_name, |file| {
        if prefix.is_empty() {
            format!("{repo}/{file}")
        } else {
            format!("{repo}/{prefix}/{file}")
        }
    });
    for sub in sorted_dir(dir) {
        if sub.is_dir() && !sub.is_symlink() {
            scan_snapshot(snap, &sub, repo, source, out);
        }
    }
}
