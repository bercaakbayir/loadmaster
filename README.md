# loadmaster

**Will this model fit on this machine at this context length, what is the exact command to run
it, and does it actually emit tool calls?** loadmaster answers all three for local LLMs, from the
real model file, with every number shown.

```console
$ loadmaster plan hf://unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf --pretend "rtx 3060" --ctx 32768
Qwen3-30B-A3B  ·  Q4_K_M  ·  48 layers · MoE  ·  ctx 32768

✓ Fits on RTX 3060 12GB (pretend) with CPU offload
  memory     11.5 of 11.5 GiB
  CPU RAM    8.5 of 24.0 GiB available
  layers     all 48 on GPU, experts of the first 24 on CPU
  KV cache   q8_0
  speed      ~36-60 tok/s at 32k context (estimate)

Run:
  llama-server --hf-repo unsloth/Qwen3-30B-A3B-GGUF --hf-file Qwen3-30B-A3B-Q4_K_M.gguf \
    -c 32768 -ngl 49 --n-cpu-moe 24 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja

Other options:
  ctx 40960   11.3 GiB  experts of 26 layers on CPU, q8_0 KV  ~33-55 tok/s
  Q5_K_M     ~11.4 GiB  experts of 28 layers on CPU, q8_0 KV  ~30-52 tok/s
  Q3_K_M     ~11.3 GiB  experts of 18 layers on CPU, q8_0 KV  ~45-73 tok/s
  ~ = size estimated; plan that file to confirm
```

`--explain` prints every term of the budget (weights, KV cache, compute buffer, vision
projector, headroom) with its inputs, and every configuration that was tried. When the GPU's
memory bandwidth is known, plans also estimate generation speed, as a band fitted to
published llama.cpp benchmarks (ADR 0009). On a terminal the output is coloured and the
command wrapped; piped, it is plain with the command on one line, and `--command` prints only
that line. `--json` is the stable format for scripts.

### Try it without the GPU or the download

`hf://` models are planned from their headers (a few MB), never the weights, and `--pretend`
stands in for hardware you don't have:

```console
$ loadmaster plan hf://unsloth/Qwen3-30B-A3B-GGUF --pretend "rtx 4090" --ctx 32768
Qwen3-30B-A3B  ·  26 quants  ·  RTX 4090 (pretend), 23.5 GiB usable

QUANT            SIZE  FIT                   CTX  KV       TOK/S
Q3_K_M       13.7 GiB  ~gpu                32768  f16    ~98-151
Q4_K_M       17.3 GiB  gpu                 32768  f16    ~94-142
Q5_K_M       20.2 GiB  ~gpu                32768  q8_0  ~106-171
UD-Q5_K_XL   20.2 GiB  ~gpu                32768  q8_0  ~106-171  ← best fit
Q6_K         23.4 GiB  ~offload            32768  q8_0   ~73-127
Q8_0         30.3 GiB  ~offload            32768  q8_0    ~40-76
...
BF16         56.9 GiB  ✗ ~30.5 GiB short       -  -            -

~ = planned from the Q4_K_M header, scaled to that file's size
exact plan for the best fit: loadmaster plan hf://unsloth/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-UD-Q5_K_XL.gguf
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

**Not sure what to run?** `suggest` searches the Hugging Face Hub for models that run here,
best fit first ([docs/suggest.md](docs/suggest.md)):

```console
$ lm suggest --pretend "rtx 4090" --ctx 32768
Models for RTX 4090 (pretend), 23.5 GiB usable  ·  llama.cpp  ·  ctx 32768  ·  trending GGUF on Hugging Face

MODEL                                        PARAMS  QUANT            SIZE  FIT      CTX      TOK/S
unsloth/Qwen3.6-35B-A3B-GGUF                  34.7B  UD-Q4_K_XL   20.8 GiB  ~gpu   32768   ~124-196
openbmb/MiniCPM-V-4.7-35B-A3B-gguf            34.7B  Q4_1         20.3 GiB  ~gpu   32768   ~125-199
unsloth/Qwen3.8-27B-GGUF                      27.3B  UD-Q5_K_XL   19.4 GiB  ~gpu   32768     ~33-40
...
exact plan: loadmaster plan hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

`lm suggest qwen`, `--kind vision|audio|embed`, `--runtime vllm` and `--sort downloads` narrow it.

**Find what you already have.** `scan` reads the HF cache, Ollama and LM Studio:

```console
$ lm scan
NAME                    SOURCE    KIND         SIZE  QUANT       8k       32k      128k
llama3.2:1b             ollama    chat      1.2 GiB  Q8_0        gpu      gpu      gpu*
BAAI/bge-small-en-v1.5  hf        embed     0.1 GiB  F32         -        -        -

on Apple M2: gpu = fits on GPU · offload = needs CPU
* = only with q4_0 KV or under the 15% free-RAM margin; `loadmaster plan <name> --ctx <n>` says which
! 1 directory skipped (--wide lists it)
```

**Speed.** When the GPU's bandwidth is known, plans and alternatives show tokens per second
([docs/speed.md](docs/speed.md)):

```console
$ lm plan hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf --pretend "rtx 4090" --ctx 32768
Llama 3.2 1B Instruct  ·  Q4_K_M  ·  16 layers  ·  ctx 32768

✓ Fits on RTX 4090 (pretend)
  memory     2.7 of 23.5 GiB
  layers     all 16 on GPU
  KV cache   f16
  speed      ~269-400 tok/s at 32k context (estimate)
```

**Several GPUs.** Layers are split across the cards; vLLM plans pick a tensor-parallel size
([docs/multi-gpu.md](docs/multi-gpu.md)):

```console
$ lm plan hf://bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf --pretend "2x rtx 3090" --ctx 32768
Llama 3.3 70B Instruct  ·  Q4_K_M  ·  80 layers  ·  ctx 32768

✓ Fits on 2× RTX 3090 (pretend)
  memory     46.5 of 47.0 GiB   GPU0 23.4/23.5 · GPU1 23.1/23.5
  layers     all 80 on GPU
  KV cache   q8_0
  speed      ~16-19 tok/s at 32k context (estimate)

! split across GPUs: not validated yet

Run:
  llama-server --hf-repo bartowski/Llama-3.3-70B-Instruct-GGUF \
    --hf-file Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 32768 -ngl 81 --device CUDA0,CUDA1 \
    --split-mode layer --tensor-split 41,40 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
```

**Vision models.** The projector (`mmproj`) is found next to the model and counted, image
tokens and encoder included ([docs/multimodal.md](docs/multimodal.md)):

```console
$ lm plan hf://ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf --pretend "rtx 3060" --ctx 16384
Gemma 3 4b It  ·  Q4_K_M  ·  34 layers  ·  ctx 16384

✓ Fits on RTX 3060 12GB (pretend)
  memory     4.8 of 11.5 GiB
  layers     all 34 on GPU
  KV cache   f16
  projector  1.1 GiB on GPU, one image
  speed      ~81-105 tok/s at 16k context (estimate)

Run:
  llama-server --hf-repo ggml-org/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q4_K_M.gguf \
    --mmproj-url https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf \
    -c 16384 -ngl 35 -fa on -np 1 --jinja
```

**Embedding models** get the embedding flags, and BERT-style encoders no KV cache:

```console
$ lm plan hf://CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --pretend "rtx 3060"
bge-small-en-v1.5  ·  Q8_0  ·  12 layers  ·  ctx 512

✓ Fits on RTX 3060 12GB (pretend)
  memory     0.8 of 11.5 GiB
  layers     all 12 on GPU
  embeddings cls pooling

Run:
  llama-server --hf-repo CompendiumLabs/bge-small-en-v1.5-gguf \
    --hf-file bge-small-en-v1.5-q8_0.gguf -c 512 -ngl 13 -fa on -np 1 --embeddings \
    --pooling cls -ub 512 -b 512
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
loadmaster probe 0.3.0 · http://127.0.0.1:8080 · model qwen3-8b · llama.cpp b6700-abc1234
  tool.single*             FAIL  call_in_content
  tool.parallel            FAIL  parallel_in_content
  schema.strict*           pass
  ...
verdict: fail (failures: call_in_content, no_call, parallel_in_content); * = core fixture
```

**Tell us how the prediction did.** `report` runs the planned command once, stops after one
short request, and writes what llama.cpp really used next to the prediction:

```console
$ lm report Qwen3-8B-Q4_K_M.gguf --ctx 32768           # on a machine with a GPU
plan: fits on RTX 4090, 10.1 GiB predicted
will run: llama-server -m Qwen3-8B-Q4_K_M.gguf -c 32768 -ngl 37 -fa on -np 1 --jinja -lv 4
(stops after one 128-token request; the log is kept for you)
continue? [y/N] y
✓ Prediction held: fits on RTX 4090
                    predicted      observed
  GPU buffers         9.3 GiB       9.1 GiB
  speed         103-144 tok/s   118.6 tok/s
  load time                 -         0.5 s

wrote loadmaster-report-qwen3-8b-q4-k-m-on-nvidia-geforce-rtx-4090-at-ctx-32768-predicted-fits-observed-fits.toml (no paths, hosts or users; --show prints it)
review and submit it here (loadmaster sends nothing):
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
| `loadmaster scan [--path DIR] [--wide] [--json]` | Index the HF cache, Ollama, LM Studio and loose dirs (read-only), with fit at 8k/32k/128k |
| `loadmaster gpu [--json] [--pretend HW\|list]` | GPUs and usable memory (NVML, Apple unified memory), or a pretend one |
| `loadmaster plan <model> [--ctx N] [--runtime llama.cpp\|vllm\|exl3] [--explain\|--json\|--command]` | Memory budget, the exact launch command, alternatives |
| `loadmaster plan hf://org/repo[/file.gguf] [--offline]` | The same from Hub headers only; a bare GGUF repo compares every quant |
| `loadmaster suggest [QUERY] [--kind chat\|vision\|audio\|embed] [--runtime vllm]` | Hub models that run on this machine, best fit first |
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
`lm` is installed as a short alias. `--color auto|always|never` works everywhere (`NO_COLOR`
is honoured). Errors exit 3.

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

[finding models](docs/suggest.md) ·
[speed estimates](docs/speed.md) ·
[several GPUs](docs/multi-gpu.md) ·
[vision, audio and embedding models](docs/multimodal.md) ·
[models.lock reference](docs/lockfile.md) ·
[probe fixtures and failure classes](docs/fixtures.md) ·
[validation](docs/validation.md) ·
[known issues](docs/known-issues.md) ·
[design decisions](docs/decisions/) ·
[changelog](CHANGELOG.md)

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or [MIT license](LICENSE-MIT)
at your option.
