# 0003 — Read GGUF headers with buffered I/O, not memmap2

Status: accepted · 2026-10-18

## Context

The spec names `memmap2` for GGUF reads. loadmaster only needs the header: metadata and the
tensor table, never tensor data. Long arrays (vocabularies, merges) sit in the header and
are skipped without materialising them.

## Decision

Parse with a `BufReader` over `std::fs::File`. Drop `memmap2`.

- Mapping a file that another process truncates (a download in progress, a re-download)
  raises `SIGBUS`; a buffered read returns an error we can report.
- `Mmap::map` is `unsafe`; the workspace denies `unsafe_code`.
- Speed is not a concern: a 1.3 GB Llama 3.2 1B Q8_0 (128k-token vocabulary) parses in 15 ms
  in release mode, against the 50 ms budget, and `weight_bytes` equals file size minus
  header exactly.

## Consequences

If a later feature needs tensor data (it should not; see non-goals), revisit this.
