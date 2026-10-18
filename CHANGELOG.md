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
