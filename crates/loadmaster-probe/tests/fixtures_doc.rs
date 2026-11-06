//! The failure-class table in `docs/fixtures.md` matches the built-in fixtures: one row
//! per `fail` class (not per condition key such as `status_class`), listing exactly the
//! fixtures that can produce it, each with a meaning.

use std::collections::BTreeMap;

use loadmaster_probe::fixture;

/// Classes the probe itself produces, not a fixture's `fail` rule.
const BUILTIN: &[(&str, &str)] = &[
    ("template_mismatch", "template.hash"),
    ("unclassified", "any"),
    ("unreachable", "any"),
];

fn expected() -> BTreeMap<String, String> {
    let mut by_class: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for f in fixture::builtin() {
        for r in &f.fail {
            let ids = by_class.entry(r.class.clone()).or_default();
            if !ids.contains(&f.id) {
                ids.push(f.id.clone());
            }
        }
    }
    let mut out: BTreeMap<String, String> = by_class
        .into_iter()
        .map(|(c, mut ids)| {
            ids.sort();
            (c, ids.join(", "))
        })
        .collect();
    for (c, f) in BUILTIN {
        out.insert((*c).into(), (*f).into());
    }
    out
}

/// `(class, fixtures, meaning)` for each row of the table under `## Failure classes`.
fn documented(md: &str) -> Vec<(String, String, String)> {
    md.split_once("## Failure classes")
        .expect("docs/fixtures.md has a `## Failure classes` section")
        .1
        .lines()
        .skip_while(|l| !l.starts_with('|'))
        .take_while(|l| l.starts_with('|'))
        .skip(2) // header and separator
        .map(|l| {
            let cells: Vec<&str> = l.trim_matches('|').split(" | ").map(str::trim).collect();
            let cell = |i: usize| cells.get(i).copied().unwrap_or("").to_owned();
            (cell(0).trim_matches('`').to_owned(), cell(1), cell(2))
        })
        .collect()
}

#[test]
fn failure_class_table_matches_the_fixtures() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../docs/fixtures.md");
    let md = std::fs::read_to_string(path).unwrap();
    let rows = documented(&md);
    let want = expected();
    let got: BTreeMap<String, String> = rows
        .iter()
        .map(|(c, f, _)| (c.clone(), f.clone()))
        .collect();
    let table: String = want
        .iter()
        .map(|(c, f)| format!("| `{c}` | {f} | … |\n"))
        .collect();
    assert_eq!(
        got, want,
        "docs/fixtures.md failure classes are out of date; the class and fixture columns should be:\n{table}"
    );
    assert_eq!(rows.len(), want.len(), "a class is listed twice");
    for (c, _, meaning) in &rows {
        assert!(
            !meaning.is_empty(),
            "`{c}` has no meaning in docs/fixtures.md"
        );
    }
}
