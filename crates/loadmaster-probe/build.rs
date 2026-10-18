//! Embed every `fixtures/*.yaml` so a new built-in fixture needs no code change.

use std::fmt::Write;

fn main() {
    let dir = std::path::Path::new("fixtures");
    println!("cargo:rerun-if-changed=fixtures");
    let mut files: Vec<_> = std::fs::read_dir(dir)
        .expect("fixtures/ exists")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "yaml"))
        .collect();
    files.sort();
    let mut out = String::from("[\n");
    for f in &files {
        let abs = std::fs::canonicalize(f).expect("fixture path");
        println!("cargo:rerun-if-changed={}", abs.display());
        writeln!(
            out,
            "    ({:?}, include_str!({:?})),",
            f.file_name().unwrap().to_string_lossy(),
            abs.display().to_string()
        )
        .unwrap();
    }
    out.push(']');
    let dest = std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("fixtures.rs");
    std::fs::write(dest, out).unwrap();
}
