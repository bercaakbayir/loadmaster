# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Cargo workspace with `loadmaster-{model,hw,plan,store,lock,serve,probe}` crates and the
  `loadmaster` binary (CLI surface for `scan`, `gpu`, `plan`; not yet functional).
- `cargo xtask check-deps` enforcing crate boundaries.
- CI (fmt, clippy, tests, MSRV 1.85, cargo-deny). ADRs 0001 (naming) and 0002 (dependencies).
- GGUF v2/v3 header reader (metadata, tensor table, split files, truncation detection) and a
  deterministic writer for synthetic fixtures.
- `ModelInfo` from real model files: per-layer bytes and KV dimensions, routed-expert bytes,
  tied output heads, sliding-window layers, MLA, quant name, chat-template hash, sibling
  `mmproj` detection.
- Architecture table (llama, qwen2/3, qwen3moe, gemma2/3, gpt-oss, deepseek2, and others).
- `cargo xtask gen-fixtures [--check]`: six header-only model fixtures plus a Gemma 3 mmproj.
- `loadmaster-hw`: `HwProvider` trait, `FakeHw` presets, NVIDIA via NVML (loaded at runtime,
  `cuda` feature on by default), Apple Silicon unified memory with the Metal working-set
  default (estimated until Metal is queried directly), host RAM from sysctl/vm_stat or
  `/proc/meminfo`.
- `loadmaster-plan`: footprint terms (weights, KV with SWA and MLA, compute buffer, mmproj,
  headroom) with per-term explanations; f16 → q8_0 → q4_0 KV selection; MoE expert offload
  and layer offload search; `CannotFit` with the closest miss and two fixes; default-context
  search; `llama-server` command emitter; `--explain` renderer (ADR 0004).
- `loadmaster gpu [--json]` and `loadmaster plan <model.gguf> [--gpu] [--ctx] [--explain] [--json]`
  for llama.cpp. Exit codes: 0 fits, 1 cannot fit, 2 error. A unified-memory warning when RAM
  free right now is below what the plan needs.

### Fixed
- `-ngl` placement: llama.cpp offloads the output head first, then the last `ngl - 1`
  layers. Partial-offload plans were one layer off.

- Unified memory (Apple Silicon): a plan fits only if GPU and host sides together fit in
  RAM free now. Previously loadmaster said "fits" for a configuration that reset an 8 GB M2
  (ADR 0005).
- Compute buffer: replaced the spec formula (3-5x too high) with one fitted to 28 observed
  runs; never below observed, at most 1.33x above (ADR 0005). Host-side compute buffer is
  now counted.

### Added
- `validation/observed.toml`: buffer sizes llama.cpp reported for 28 runs (Llama 3.2 1B,
  Qwen3 0.6B) on an M2, plus one recorded incident, replayed by ignored tests. Weights and KV
  match within 0.3 MiB.
