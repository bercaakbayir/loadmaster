# Speed estimates

When loadmaster knows the GPU's memory bandwidth, every llama.cpp plan estimates how fast the
model generates at the planned context (`--explain` adds 4k context):

```console
$ lm plan hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf --pretend "rtx 4090" --ctx 32768
Llama 3.2 1B Instruct  ·  Q4_K_M  ·  16 layers  ·  ctx 32768

✓ Fits on RTX 4090 (pretend)
  memory     2.7 of 23.5 GiB
  layers     all 16 on GPU
  KV cache   f16
  speed      ~269-400 tok/s at 32k context (estimate)

Run:
  llama-server --hf-repo bartowski/Llama-3.2-1B-Instruct-GGUF \
    --hf-file Llama-3.2-1B-Instruct-Q4_K_M.gguf -c 32768 -ngl 17 -fa on -np 1 --jinja

Other options:
  ctx 65536   3.8 GiB  all on GPU, f16 KV  ~207-281 tok/s
  Q5_K_M     ~2.8 GiB  all on GPU, f16 KV  ~259-380 tok/s
  Q3_K_M     ~2.5 GiB  all on GPU, f16 KV  ~282-427 tok/s
  ~ = size estimated; plan that file to confirm
```

The alternatives show tokens per second too, so "a bigger quant, or more layers on the CPU"
becomes a concrete trade.

## Where the bandwidth comes from

| Hardware | Source |
|---|---|
| Detected NVIDIA card in the catalog | vendor spec sheet |
| Detected NVIDIA card not in the catalog | NVML: bus width x max memory clock x 2 |
| Detected Apple Silicon | chip table (`machdep.cpu.brand_string`) |
| `--pretend "rtx 4090"`, `--pretend "m4 pro 48gb"` | catalog (`lm gpu --pretend list`) |
| `--pretend "16GB@448"` | the number after `@`, in GB/s |
| `--pretend 24GB` | unknown: no speed line |

No bandwidth, no estimate: loadmaster never guesses one.

```console
$ lm plan …Llama-3.2-1B-Instruct-Q4_K_M.gguf --pretend "16GB@448" --ctx 8192
Llama 3.2 1B Instruct  ·  Q4_K_M  ·  16 layers  ·  ctx 8192

✓ Fits on 16 GiB GPU (pretend)
  memory     1.9 of 15.5 GiB
  layers     all 16 on GPU
  KV cache   f16
  speed      ~234-331 tok/s at 8k context (estimate)
```

## How it is worked out

`--explain` prints a `SPEED` section:

```
SPEED    generation, memory-bound (estimate)
  bandwidth          1008 GB/s (pretend: 1008 GB/s, NVIDIA GeForce RTX 4090 spec sheet)
  CUDA fit           efficiency 0.95-1.00, 40-110 us per layer per token (ADR 0009)
  at 4096 tokens     read 0.87 GiB from GPU per token: 366-638 tok/s
  at 32768 tokens    read 1.74 GiB from GPU per token: 269-400 tok/s
```

Each generated token reads every weight it uses plus the KV cache so far, so generation is
limited by memory bandwidth:

```
time per token = GPU bytes / (bandwidth x efficiency) + CPU bytes / host bandwidth + layers x fixed cost
```

- MoE models read only the experts a token uses (`n_expert_used / n_expert`).
- Layers or experts offloaded to the CPU are read at host RAM speed: 40-90 GB/s assumed next
  to a dedicated GPU, a third to two thirds of the unified bandwidth on a Mac.
- Several GPUs (`--pretend "2x rtx 3090"`) process each token in turn, so their times add.

The efficiency and fixed-cost ranges are fitted to 48 published llama.cpp benchmarks
(`validation/speed.toml`), and a test checks that every one of them stays inside its band.
Sources and the fit: [ADR 0009](decisions/0009-speed-estimates.md).

## Not estimated

- Prompt processing (compute-bound).
- vLLM and ExLlamaV3 plans.
- Embedding models (they don't generate).
- Several requests at once: plans assume one (`-np 1`).

## Make it better

The CUDA fit rests on 8 points from one model shape. If you have a GPU,
`lm report model.gguf` measures real generation and prompt speed and puts them next to the
estimate ([CONTRIBUTING.md](../CONTRIBUTING.md)).
