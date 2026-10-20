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
catalog; `--pretend "2x rtx 3090"` plans a split across two cards. Gated repos need
`HF_TOKEN`; headers are cached under `~/.loadmaster/hf`.

## Examples

The outputs below are real runs on `--pretend` hardware with `hf://` headers, trimmed. You
can paste any of them without a GPU or a download. The `probe` and `report` outputs come from
loadmaster's fake runtime, so their numbers are illustrative.

**Find what you already have.** `scan` reads the HF cache, Ollama and LM Studio:

```console
$ lm scan
NAME                    SOURCE   KIND         SIZE  ARCH       QUANT            CTX  MMPROJ TEMPLATE  8k      32k     128k
llama3.2:1b             ollama   chat     1.23 GiB  llama      Q8_0          131072  -      da36d51c  gpu     gpu     gpu*
BAAI/bge-small-en-v1.5  hf       embed    0.12 GiB  bert       F32              512  -      -         -       -       -
```

**Speed.** When the GPU's bandwidth is known, plans and alternatives show tokens per second
([docs/speed.md](docs/speed.md)):

```console
$ lm plan hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf --pretend "rtx 4090" --ctx 32768
fits: 2.66 GiB of 23.50 GiB usable on NVIDIA GeForce RTX 4090 (pretend); all 16 layers and the output head on GPU, f16 KV
speed: ~366-638 tok/s generating at 4k context, ~269-400 at 32k (estimate)
```

**Several GPUs.** Layers are split across the cards; vLLM plans pick a tensor-parallel size
([docs/multi-gpu.md](docs/multi-gpu.md)):

```console
$ lm plan hf://bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf --pretend "2x rtx 3090" --ctx 32768
fits: 46.47 GiB across 2 GPUs (GPU0 23.4 / 23.5 GiB, GPU1 23.1 / 23.5 GiB); all 80 layers and the output head on GPU, q8_0 KV
speed: ~18-21 tok/s generating at 4k context, ~16-19 at 32k (estimate)

llama-server --hf-repo bartowski/Llama-3.3-70B-Instruct-GGUF --hf-file Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 32768 -ngl 81 --device CUDA0,CUDA1 --split-mode layer --tensor-split 41,40 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
```

**Vision models.** The projector (`mmproj`) is found next to the model and counted, image
tokens and encoder included ([docs/vision-embeddings.md](docs/vision-embeddings.md)):

```console
$ lm plan hf://ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf --pretend "rtx 3060" --ctx 16384
fits: 4.81 GiB of 11.51 GiB usable on NVIDIA GeForce RTX 3060 12GB (pretend); all 34 layers and the output head on GPU, f16 KV

llama-server --hf-repo ggml-org/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q4_K_M.gguf --mmproj-url https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf -c 16384 -ngl 35 -fa on -np 1 --jinja
```

**Embedding models** get the embedding flags, and BERT-style encoders no KV cache:

```console
$ lm plan hf://CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --pretend "rtx 3060"
fits: 0.83 GiB of 11.51 GiB usable on NVIDIA GeForce RTX 3060 12GB (pretend); all 12 layers on GPU, no KV

llama-server --hf-repo CompendiumLabs/bge-small-en-v1.5-gguf --hf-file bge-small-en-v1.5-q8_0.gguf -c 512 -ngl 13 -fa on -np 1 --embeddings --pooling cls -ub 512 -b 512
```

**Serve several models on one port**, unloading idle ones when a request needs room:

```console
$ lm init llama3.2:1b qwen3:8b       # plan each model, write models.lock
$ lm up --verify                     # serve on 127.0.0.1:8080, probe each model as it starts
$ lm status                          # what is loaded, its footprint, its probe verdict
```

**Check that tool calls work** on any OpenAI-compatible endpoint (`--embeddings` for an
embedding model):

```console
$ lm probe http://127.0.0.1:8080 --model qwen3-8b
  tool.single*             FAIL  call_in_content
  tool.parallel            FAIL  parallel_in_content
  schema.strict*           pass
  ...
verdict: fail (failures: call_in_content, no_call, parallel_in_content); * = core fixture
```

**Tell us how the prediction did.** `report` runs the planned command once, stops after one
short request, and writes what llama.cpp really used next to the prediction:

```console
$ lm report Qwen3-8B-Q4_K_M.gguf --ctx 8192            # on a machine with a GPU
plan: fits, 6.61 GiB predicted on NVIDIA GeForce RTX 4090
will run: llama-server -m Qwen3-8B-Q4_K_M.gguf -c 8192 -ngl 37 -fa on -np 1 --jinja -lv 4
continue? [y/N] y
result: fits · generation 41.2 tok/s, prompt 1830 tok/s
wrote loadmaster-report-qwen3-8b-q4-k-m-on-nvidia-geforce-rtx-4090-at-ctx-8192-predicted-fits-observed-fits.toml
to share it, open this link, review the prefilled issue and submit (loadmaster sends nothing):
  https://github.com/bercaakbayir/loadmaster/issues/new?template=validation.yml&...
```

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
| `loadmaster probe <url> --model <name> [--embeddings] [--json]` | Tool-call and structured-output conformance (or embedding checks); exit 0 pass, 1 partial, 2 fail |
| `loadmaster report <model> [--ctx N]` | Run once, record real buffers and speed next to the prediction, get a prefilled issue to share |
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

Help close these with one command: `loadmaster report model.gguf --ctx 32768` runs the model
once, writes what it really used next to the prediction, and gives you a prefilled issue to
submit ([CONTRIBUTING.md](CONTRIBUTING.md)). Nothing is uploaded unless you submit it.

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

[speed estimates](docs/speed.md) ·
[several GPUs](docs/multi-gpu.md) ·
[vision and embedding models](docs/vision-embeddings.md) ·
[models.lock reference](docs/lockfile.md) ·
[probe fixtures and failure classes](docs/fixtures.md) ·
[validation](docs/validation.md) ·
[known issues](docs/known-issues.md) ·
[design decisions](docs/decisions/) ·
[changelog](CHANGELOG.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
