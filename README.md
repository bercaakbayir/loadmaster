# loadmaster

**Will this model fit on this machine at this context length, what is the exact command to run
it, and does it actually emit tool calls?** loadmaster answers all three for local LLMs, from the
real model file, with every number shown.

```console
$ loadmaster plan Qwen3-30B-A3B-Q4_K_M.gguf --ctx 32768
Qwen3 30B A3B · qwen3moe · Q4_K_M · 48 layers · MoE  ·  ctx 32768
fits: 11.52 GiB of 11.74 GiB usable on NVIDIA GeForce RTX 3060; all 48 layers and the output head on GPU, routed experts of the first 22 on CPU, q8_0 KV

llama-server -m Qwen3-30B-A3B-Q4_K_M.gguf -c 32768 -ngl 49 --n-cpu-moe 22 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja

alternatives:
  more context (40960)     11.63 GiB ( +0.11 GiB)  experts of 23 layers on CPU, q8_0 KV    2.3-5.0x slower (estimate)
  quant up (Q5_K_M)        11.54 GiB ( +0.02 GiB)  experts of 26 layers on CPU, q8_0 KV    2.5-5.5x slower (estimate)
                           size estimated from 5.70 bpw; plan the real file to confirm
  quant down (Q3_K_M)      11.49 GiB ( -0.03 GiB)  experts of 15 layers on CPU, q8_0 KV    1.9-3.6x slower (estimate)
                           size estimated from 3.91 bpw; plan the real file to confirm
```

`--explain` prints every term of the budget (weights, KV cache, compute buffer, vision
projector, headroom) with its inputs, and every configuration that was tried. When the GPU's
memory bandwidth is known, plans also estimate generation speed, as a band fitted to
published llama.cpp benchmarks (`speed: ~103-144 tok/s generating at 4k context`; ADR 0009).

### Try it without the GPU or the download

`hf://` models are planned from their headers (a few MB), never the weights, and `--pretend`
stands in for hardware you don't have:

```console
$ loadmaster plan hf://unsloth/Qwen3-30B-A3B-GGUF --pretend "rtx 4090" --ctx 32768
Qwen3-30B-A3B · 26 quants · NVIDIA GeForce RTX 4090 (pretend) (23.50 GiB usable)
QUANT             SIZE  FIT                      CTX  KV           GPU
Q3_K_M       13.70 GiB  gpu                    32768  f16    17.50 GiB  *
Q4_K_M       17.28 GiB  gpu                    32768  f16    21.05 GiB
Q5_K_M       20.23 GiB  gpu                    32768  q8_0   22.56 GiB  *
Q6_K         23.37 GiB  offload                32768  q8_0   23.20 GiB  *
Q8_0         30.25 GiB  offload                32768  q8_0   23.42 GiB  *
BF16         56.90 GiB  no, 30.52 GiB short        -  -       5.05 GiB  *
...
* estimated: the Q4_K_M header scaled to that file's size
```

Name a file (`hf://org/repo/file.gguf`) for the exact plan; the command then uses
`--hf-repo/--hf-file`, so llama-server downloads that same file. Safetensors repos work with
`--runtime vllm`. `--pretend` takes a card (`rtx 3090`, `a6000`), a Mac with its RAM
(`m4 pro 48gb`) or a size (`24GB`, `48GB-unified`); `loadmaster gpu --pretend list` shows the
catalog. Gated repos need `HF_TOKEN`; headers are cached under `~/.loadmaster/hf`.

## Why "loadmaster"?

On a cargo aircraft the loadmaster decides whether the flight can happen at all: they weigh
every pallet against maximum takeoff weight, place the load so the centre of gravity stays in
limits, and sign off the preflight. Running a local LLM is the same job in different units.

- **Weight**: do the tensors, KV cache, compute buffers and vision projector fit in GPU memory
  at the context length you want?
- **Balance**: which layers stay on the GPU, whether MoE experts are offloaded, f16 or q8_0 KV
  cache, and which model gets unloaded when several share one card.
- **Preflight**: does a model that fits also fly: tool calls in `tool_calls` rather than plain
  text, JSON schemas honoured, streams intact?

loadmaster does these in that order, reads the real model file instead of a spec sheet, prints
every number it used, and writes the result to `models.lock` so the same aircraft is loaded the
same way tomorrow.

## Commands

| Command | Does |
|---|---|
| `loadmaster scan [--path DIR] [--json]` | Index the HF cache, Ollama, LM Studio and loose dirs (read-only), with fit at 8k/32k/128k |
| `loadmaster gpu [--json] [--pretend HW\|list]` | GPUs and usable memory (NVML, Apple unified memory), or a pretend one |
| `loadmaster plan <model> [--ctx N] [--runtime llama.cpp\|vllm\|exl3] [--explain] [--json]` | Memory budget, the exact launch command, alternatives |
| `loadmaster plan hf://org/repo[/file.gguf] [--offline]` | The same from Hub headers only; a bare GGUF repo compares every quant |
| `loadmaster init <names…\|--all>` / `add <model>` | Write `models.lock` with frozen flags and footprints |
| `loadmaster plan <name> --update` | Re-plan one lockfile entry and show the diff |
| `loadmaster up [--verify]` / `down` / `swap <name>` / `status` | Serve every model on one OpenAI-compatible port, evicting by footprint |
| `loadmaster import llama-swap <config.yaml>` | Convert a llama-swap config into `models.lock` |
| `loadmaster probe <url> --model <name> [--json]` | Tool-call and structured-output conformance; exit 0 pass, 1 partial, 2 fail |
| `loadmaster probe eval <prompts.jsonl> --model a --model b` / `probe compare <run> <run>` | Your prompts, deterministic sampling, SQLite results, significance-tested diffs |

`<model>` is a `.gguf` file, a Hugging Face model directory, a name from `scan`
(`llama3.2:1b`, `org/repo`), or `hf://org/repo[/file]`: from the local cache when it is there,
else from the Hub's headers. `--pretend HW` works on `plan`, `scan`, `gpu`, `init` and `add`.
`lm` is installed as a short alias. Errors exit 3.

## How it works

- **Plan.** GGUF and safetensors headers give exact per-layer bytes and KV dimensions
  (GQA, sliding-window layers, MLA). The planner follows llama.cpp's own placement
  (`-ngl` offloads the output head, then the last layers; `--n-cpu-moe` keeps routed experts
  of the first layers on the CPU), prefers f16 KV, then q8_0, then q4_0 with a warning, then
  offloads experts before whole layers. On Apple Silicon a plan must fit in the RAM free
  right now and should leave 15% of it unused; one that cannot gets a `tight on RAM`
  warning. vLLM plans solve `--gpu-memory-utilization` and the
  largest `--max-model-len`; exl3 plans emit a tabbyAPI config. Same inputs, same bytes out.
- **Serve.** `up` starts runtimes on demand behind one proxy port, unloads the least recently
  used idle model when a request needs room (`keep_warm` models last, models mid-request
  never), restarts crashes with backoff, and exposes `/metrics`. On Apple Silicon the budget
  is also capped by free RAM.
- **Probe.** Ten declarative fixtures (tool calls single, parallel, nested, round-trip;
  strict, union and enum/pattern schemas; streamed tool calls; reasoning leaks; the chat
  template) run three times each against any OpenAI-compatible endpoint; failures are
  classified (`call_in_content`, `schema_dropped_silent`, `stream_truncated`, …).
  `up --verify` records the verdict in `models.lock`; `status` flags it stale when the
  runtime version changes.

## Status

Pre-release. All commands are implemented and tested without a GPU (fixtures, a scriptable
fake OpenAI server, fake hardware). What is and is not validated against real runs:

| | Validated | Not yet |
|---|---|---|
| Weights and KV footprint | exact vs llama.cpp (28 runs, Apple M2, Metal) | NVIDIA/CUDA |
| Compute buffer | never under, at most 1.33x over (same runs) | CUDA; MoE models |
| Fit / OOM | [13/13 cases](docs/validation.md) on one M2 8 GB | the 30-case NVIDIA set; llmfit and `--fit` columns |
| vLLM, exl3 plans | unit-tested | real runs |
| Probe | every failure class reproduced on the fake runtime | [`RESULTS.md`](RESULTS.md) for 20 models (needs a 24 GB GPU) |
| Docker, systemd, AUR, Windows | files written | built and run |

Help close these: [docs/validation.md](docs/validation.md) explains how to record a case
safely on your machine.

## Install

```sh
cargo install loadmaster     # once published; until then:
cargo install --path crates/loadmaster
```

Release binaries and a Homebrew formula come from cargo-dist (`dist-workspace.toml`).
llama.cpp, vLLM or tabbyAPI are separate installs; loadmaster plans for them and runs them.

## Building

```sh
cargo build --release
cargo test --workspace                 # no GPU needed
cargo xtask check-deps                 # crate boundaries
cargo xtask gen-fixtures --check       # synthetic model fixtures are current
cargo xtask fetch-headers && cargo xtask validate
```

## Docs

[models.lock reference](docs/lockfile.md) ·
[probe fixtures and failure classes](docs/fixtures.md) ·
[validation](docs/validation.md) ·
[known issues](docs/known-issues.md) ·
[design decisions](docs/decisions/) ·
[changelog](CHANGELOG.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
