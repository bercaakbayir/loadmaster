# loadmaster

**Will this model fit on this machine at this context length, what is the exact command to run
it, and does it actually emit tool calls?** loadmaster answers all three for local LLMs, from the
real model file, with every number shown.

> Status: pre-release. Nothing below is usable yet; see [CHANGELOG.md](CHANGELOG.md).

## Why "loadmaster"?

On a cargo aircraft the loadmaster decides whether the flight can happen at all: they weigh
every pallet against maximum takeoff weight, place the load so the centre of gravity stays in
limits, and sign off the preflight. Running a local LLM is the same job in different units.

- **Weight** — do the tensors, KV cache, compute buffers and vision projector fit in GPU memory
  at the context length you want?
- **Balance** — which layers stay on the GPU, whether MoE experts are offloaded, f16 or q8_0 KV
  cache, and which model gets unloaded when several share one card.
- **Preflight** — does a model that fits also fly: tool calls in `tool_calls` rather than plain
  text, JSON schemas honoured, streams intact?

loadmaster does these in that order, reads the real model file instead of a spec sheet, prints
every number it used, and writes the result to `models.lock` so the same aircraft is loaded the
same way tomorrow.

## Commands

| Command | Does | Ships in |
|---|---|---|
| `loadmaster scan` | Index HF cache, Ollama, LM Studio and loose GGUF dirs (read-only) | v0.1 |
| `loadmaster gpu` | Detect GPUs and usable memory | v0.1 |
| `loadmaster plan <model> [--ctx N] [--explain]` | Memory budget and exact launch command for llama.cpp, vLLM or exl3 | v0.1 |
| `loadmaster init` / `add` | Write `models.lock` with frozen flags and footprints | v0.2 |
| `loadmaster up` / `down` / `swap` / `status` | Serve several models on one port, evicting by footprint | v0.2 |
| `loadmaster import llama-swap <config.yaml>` | Migrate a llama-swap config | v0.2 |
| `loadmaster probe <url> --model <name>` | Tool-call and structured-output conformance verdict | v0.3 |

`lm` is installed as a short alias for `loadmaster`.

## Building

```sh
cargo build --release
cargo test --workspace
cargo xtask check-deps
```

No GPU is needed to build or test.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
