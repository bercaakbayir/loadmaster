# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased] (0.3.0 scope; not tagged)

### Added (planning without the hardware)
- `--pretend <hw>` on `plan`, `scan`, `gpu`, `init`, `add`: plan for a card from a built-in
  catalog (`rtx 4090`, `a6000`), a Mac with its RAM (`m4 pro 48gb`), or a raw size (`24GB`,
  `48GB-unified`), with optional host RAM (`rtx 3090+128GB`). `gpu --pretend list` prints the
  catalog; `up` refuses it.
- Memory bandwidth in the hardware snapshot: from the catalog (20 NVIDIA cards and 15 Apple
  chips, each figure cited in ADR 0009), else NVML's bus width and memory clock; `--pretend
  24GB@936` sets it for raw sizes. `gpu` and `gpu --pretend list` show it.
- `plan hf://org/repo/file.gguf` plans a Hub model from its header alone (a few MB, never
  the weights): split GGUFs and a sibling mmproj included, headers cached under
  `~/.loadmaster/hf`, `--offline` for cache only, `HF_TOKEN` for gated repos, `HF_ENDPOINT`
  for mirrors. The command uses `--hf-repo/--hf-file` so llama-server fetches the same file.
- `plan hf://org/repo` compares every GGUF quant in a repo: one header planned exactly, the
  rest scaled to their file sizes (within a few hundred bytes of the exact plan on Llama
  3.2 1B), with the largest quant that fits entirely on the GPU suggested. `--json` too.
- `plan hf://org/repo --runtime vllm` plans a safetensors repo from `config.json`, the
  shard index and each shard's header (AWQ/GPTQ/FP8 included); the command is
  `vllm serve org/repo`.

### Added (probe, verify, eval)
- `loadmaster probe <url> --model <name>`: ten declarative YAML fixtures (template, tool
  single/parallel/nested/round-trip, strict/union/enum-pattern schemas, streamed tool calls,
  reasoning leaks), three agreeing runs each, classified failures, runtime detection for
  llama.cpp, Ollama, vLLM, LM Studio, SGLang and TabbyAPI, raw captures, `--json`, exit
  0/1/2 for pass/partial/fail. Adding a fixture is adding a file.
- `loadmaster up --verify` writes `[model.verify]`; `status` marks verdicts stale when the
  runtime version changes.
- `loadmaster probe eval` (JSONL prompts, deterministic sampling, SQLite) and
  `probe compare` (two-proportion z-test, `n.s.` below p = 0.05).
- `cargo xtask results select | run | render` for RESULTS.md: models chosen from the HF API
  by recorded rules (`results/matrix.toml`); no results yet (needs a 24 GB GPU).
- Docs: lockfile reference, fixture authoring and failure classes, known issues (empty until
  confirmed upstream), launch drafts. Windows target and CI job (untested by hand).

### Changed
- CLI errors exit 3, so 0-2 always mean a result.
- Unified memory: plans keep 15% of available RAM unused when they can (smaller default
  context, q8_0 KV instead of f16), and warn `tight on RAM` when they cannot (ADR 0007).
- `scan` marks fits that need a q4_0 KV cache or break the RAM margin with `*`; `--json`
  has them as `caveats`.

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
- Hugging Face model directories (`config.json` + safetensors, single or sharded): exact
  tensor bytes from safetensors headers, `text_config` nesting for multimodal configs,
  vision towers as the projector term, SWA from `layer_types`/`sliding_window_pattern`,
  AWQ/GPTQ/FP8/EXL3 detection, chat template from `tokenizer_config.json` or
  `chat_template.jinja`. `ModelInfo::open` takes a `.gguf` file or a model directory.
- Three HF fixture directories (Llama-3.1-8B BF16 sharded, Qwen3-8B AWQ, Gemma 3 12B
  multimodal) generated by `cargo xtask gen-fixtures`.
- `--runtime vllm`: solves `--gpu-memory-utilization` from free memory and the largest
  `--max-model-len` whose KV blocks fit what weights, activations and the vision tower leave;
  falls back to fp8 KV; emits `vllm serve`. NVIDIA and HF model directories only.
- `--runtime exl3`: the full-GPU search with tabbyAPI's 2048-token chunks; emits a tabbyAPI
  `config.yml` model section. EXL3 quants on NVIDIA only.
- Alternatives table under every plan (and in `--json`): next context up, one GGUF quant
  step up and down (size estimated from bits per weight), and the no-offload context when
  the plan offloads, each with GPU bytes, the change from the plan, and a slowdown band.
- `loadmaster-store` and `loadmaster scan [--path DIR] [--json]`: read-only index of the HF
  hub cache (refs and snapshots, blobs counted once, GGUFs in subfolders), Ollama (schema 1
  and 2 manifests, projector layers), LM Studio and loose directories, sorted by size, with
  arch, quant, training context, projector, template hash and fit at 8k/32k/128k. Broken
  symlinks, missing blobs and non-generative models are reported, not fatal.
- `loadmaster plan <name>` resolves names from the stores (`llama3.2:1b`, `org/repo`,
  `hf://org/repo/file.gguf`, or a unique file stem).
- `lm` alias: a second binary in the `loadmaster` crate, so every install path has both.
- Release config for cargo-dist (`dist-workspace.toml`): macOS arm64/x86_64 and Linux
  x86_64/aarch64 glibc builds (glibc because NVML is dlopen()ed; ADR 0006), shell and
  Homebrew installers.
- Validation: `validation/cases.toml` (13 real cases from an M2 8 GB, 13/13 agree),
  `cargo xtask validate [--markdown]`, `cargo xtask fetch-headers` (GGUF headers by HTTP range
  request, no weights), `scripts/collect-case.sh` to record new cases safely, and
  `docs/validation.md`.

### Changed
- macOS available memory is `kern.memorystatus_level` x RAM (the kernel's own figure),
  not vm_stat's free + inactive, which refused runs that work (ADR 0005, amended).
- `loadmaster-lock`: `models.lock` schema (spec §5 plus `load_timeout_s`, `mmproj`, `env`,
  per-model `idle_unload_s`), unknown keys rejected, every problem reported at once,
  canonical writer that round-trips byte for byte (proptest), versioning, diff, entries from
  plans with machine-specific arguments stripped.
- llama-swap importer: macros (global, per-model, `${env.X}`, `${MODEL_ID}`), multi-line
  `cmd` with comments, `llama-server` and `vllm serve`, aliases, `ttl`, `globalTTL`, `env`;
  containers and unknown servers skipped with a reason, unsupported settings reported.
- `loadmaster-serve` and `loadmaster up | down | swap <name> | status`: lazy loading with a
  pure footprint-based eviction scheduler (LRU, `keep_warm` last, models in use never;
  property-tested), runtime processes with logs, `/health` readiness, crash restarts with
  backoff (3, then `failed` with the log path), SIGTERM shutdown, idle unload, an
  OpenAI-compatible proxy on one port (routing by name and alias, unbuffered streaming,
  503 + `Retry-After` while loading), `/v1/models`, `/metrics`, admin endpoints.
- `loadmaster init`, `add`, `plan --update <name>` and `import llama-swap`.
- On unified memory the supervisor's budget is also capped by free RAM.
- Packaging: Dockerfile (CUDA/CPU on llama.cpp's server images), systemd unit, AUR
  PKGBUILD. Not yet built or tested; see `packaging/README.md`.
- A fake OpenAI-compatible runtime (`loadmaster __fake-runtime`) that can misbehave in each
  way the probe classifies; the end-to-end supervisor test runs on it.
