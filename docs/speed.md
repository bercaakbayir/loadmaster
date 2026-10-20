# Speed estimates

When loadmaster knows the GPU's memory bandwidth, every llama.cpp plan estimates how fast the
model generates, at 4k context and at the planned context:

```console
$ lm plan hf://bartowski/Llama-3.2-1B-Instruct-GGUF/Llama-3.2-1B-Instruct-Q4_K_M.gguf --pretend "rtx 4090" --ctx 32768
fits: 2.66 GiB of 23.50 GiB usable on NVIDIA GeForce RTX 4090 (pretend); all 16 layers and the output head on GPU, f16 KV
speed: ~366-638 tok/s generating at 4k context, ~269-400 at 32k (estimate)
...
alternatives:
  more context (65536)      3.76 GiB ( +1.09 GiB)  all on GPU, f16 KV                     ~366-638 tok/s (estimate)
  quant up (Q5_K_M)         2.79 GiB ( +0.12 GiB)  all on GPU, f16 KV                     ~348-589 tok/s (estimate)
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
fits: 1.86 GiB of 15.51 GiB usable on 16 GiB GPU (pretend); all 16 layers and the output head on GPU, f16 KV
speed: ~253-367 tok/s generating at 4k context, ~234-331 at 8k (estimate)
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
