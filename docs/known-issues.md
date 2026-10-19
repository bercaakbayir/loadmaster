# Known issues

`loadmaster probe` prints an upstream issue link when a failure matches a signature in
`crates/loadmaster-probe/fixtures/known_issues.toml` (failure class, optionally runtime and
model). Extra signatures can be passed with `--known-issues FILE`.

| Failure | Runtime | Model | Issue | Note |
|---|---|---|---|---|

The list is empty: entries are added only for issues confirmed against a probe failure, with
a link to the real upstream report. When `RESULTS.md` is first generated, each failure
signature it shows should get an entry here (spec §9, v0.3.0).

## Speed estimates (loadmaster's own)

- Bands are fitted to one model shape (7B, 32 layers) on llama.cpp; other sizes extrapolate.
- CUDA rests on 8 points from 5 cards; the band is wider than Metal's.
- CPU-offload speed uses an assumed host RAM bandwidth (40-90 GB/s) that is not detected.
- Prompt processing is not estimated.

Run `loadmaster report` on your hardware to add a measured point (ADR 0009).
