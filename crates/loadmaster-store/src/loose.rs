//! Loose directories (`scan --path`): GGUF files and HF model directories, up to four
//! levels deep. Named by their path relative to the root.

use std::path::Path;

use loadmaster_model::Source;

use crate::{Scan, models_in_dir, sorted_dir};

const MAX_DEPTH: usize = 4;

pub fn scan(root: &Path, out: &mut Scan) {
    walk(root, root, 0, out);
}

fn walk(root: &Path, dir: &Path, depth: usize, out: &mut Scan) {
    let rel = |p: &Path| {
        p.strip_prefix(root)
            .map(|r| r.display().to_string())
            .unwrap_or_else(|_| p.display().to_string())
    };
    let dir_name = if dir == root {
        root.display().to_string()
    } else {
        rel(dir)
    };
    let prefix = rel(dir);
    models_in_dir(dir, &Source::Loose, out, Some(dir_name), |file| {
        if prefix.is_empty() {
            file.to_owned()
        } else {
            format!("{prefix}/{file}")
        }
    });
    if depth < MAX_DEPTH {
        for sub in sorted_dir(dir) {
            if sub.is_dir() && !sub.is_symlink() {
                walk(root, &sub, depth + 1, out);
            }
        }
    }
}
