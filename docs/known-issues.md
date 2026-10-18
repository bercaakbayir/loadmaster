# Known issues

`loadmaster probe` prints an upstream issue link when a failure matches a signature in
`crates/loadmaster-probe/fixtures/known_issues.toml` (failure class, optionally runtime and
model). Extra signatures can be passed with `--known-issues FILE`.

| Failure | Runtime | Model | Issue | Note |
|---|---|---|---|---|

The list is empty: entries are added only for issues confirmed against a probe failure, with
a link to the real upstream report. When `RESULTS.md` is first generated, each failure
signature it shows should get an entry here (spec §9, v0.3.0).
