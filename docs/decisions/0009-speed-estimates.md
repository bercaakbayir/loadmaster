# 0009 — Speed estimates from memory bandwidth

Status: accepted · 2026-10-19 · bands fitted to published llama.cpp benchmarks; narrow them
with `loadmaster report` data

## Bandwidth sources

Peak memory bandwidth, GB/s (10^9 B/s), as stated by the vendor. Fetched 2026-10-19.

| Hardware | GB/s | Source |
|---|---|---|
| RTX 3060 12GB, 3080 10GB, 3080 12GB, 3090 | 360, 760, 912, 936 | https://en.wikipedia.org/wiki/GeForce_30_series (NVIDIA's compare page lists bus width only) |
| RTX 4060 Ti 16GB, 4070 (GDDR6X), 4070 Ti Super, 4080, 4090 | 288, 504, 672, 716.8, 1008 | https://en.wikipedia.org/wiki/GeForce_40_series |
| RTX 5070 Ti, 5080, 5090 | 896, 960, 1792 | https://www.nvidia.com/en-us/geforce/graphics-cards/compare/ |
| RTX A4000 | 448 | https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a4000/nvidia-rtx-a4000-datasheet.pdf |
| RTX A5000 | 640 | https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs21/rtx-a5000/nvidia-rtx-a5000-datasheet.pdf |
| RTX A6000 | 768 | NVIDIA RTX A6000 datasheet (proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9) |
| RTX 6000 Ada | 960 | https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/rtx-6000/proviz-print-rtx6000-datasheet-web-2504660.pdf |
| L40S | 864 | https://www.nvidia.com/en-us/data-center/l40s/ |
| A100 40GB; 80GB PCIe; 80GB SXM | 1555; 1935; 2039 | https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf |
| H100 80GB SXM | 3350 | https://www.nvidia.com/en-us/data-center/h100/ |
| H100 80GB PCIe | 2000 | https://www.nvidia.com/content/dam/en-zz/Solutions/gtcs22/data-center/h100/PB-11133-001_v01.pdf |
| Apple M1 | 68.25 | https://en.wikipedia.org/wiki/Apple_M1 (no Apple figure found) |
| M1 Pro, M1 Max | 200, 400 | https://www.apple.com/newsroom/2021/10/introducing-m1-pro-and-m1-max-the-most-powerful-chips-apple-has-ever-built/ |
| M1 Ultra | 800 | https://www.apple.com/newsroom/2022/03/apple-unveils-m1-ultra-the-worlds-most-powerful-chip-for-a-personal-computer/ |
| M2 | 100 | https://www.apple.com/newsroom/2022/06/apple-unveils-m2-with-breakthrough-performance-and-capabilities/ |
| M2 Pro, M2 Max | 200, 400 | https://www.apple.com/newsroom/2023/01/apple-unveils-m2-pro-and-m2-max-next-generation-chips-for-next-level-workflows/ |
| M2 Ultra | 800 | https://www.apple.com/newsroom/2023/06/apple-introduces-m2-ultra/ |
| M3 | 100 | https://support.apple.com/en-gb/118552 |
| M3 Pro; M3 Max 30-/40-core GPU | 150; 300 / 400 | https://support.apple.com/en-us/117736 |
| M3 Ultra | 819 | https://support.apple.com/en-us/122211 |
| M4 | 120 | https://support.apple.com/en-us/121552 |
| M4 Pro; M4 Max 32-/40-core GPU | 273; 410 / 546 | https://support.apple.com/en-us/121553 |

Chips sold with two bandwidths list the lower one and say so. Not verified, so not in the
catalog: the RTX 4070 GDDR6 variant. A detected NVIDIA card missing from the catalog uses NVML:
bus width x maximum memory clock x 2 (reproduces the 4090, 3060 and A100 figures above).

## Model

Generating a token reads every weight it uses and the KV cache so far, so decode is
memory-bound:

```
t = gpu_bytes / (BW x e) + cpu_bytes / BW_host + n_layer x c        tok/s = 1 / t
```

- `gpu_bytes`, `cpu_bytes`: weights read per token on each side (MoE routed experts at
  `n_expert_used / n_expert`, shared experts and attention in full, token embeddings not at
  all), plus KV read at the context fill (sliding-window layers at most the window).
- `e`: the fraction of peak bandwidth reached; `c`: a fixed cost per layer per token (kernel
  launches, synchronisation). A single efficiency fraction cannot fit the data: big chips
  (A100, M2 Ultra) look inefficient on a 7B model only because the fixed cost dominates.
- The output is a band: the slow end uses low `e`, high `c` and slow host RAM; the fast end
  the opposite. Shown at 4k context and at the planned context.

## Fit

Against the 48 published points in `validation/speed.toml` (llama-bench tg128, LLaMA 2 7B):
the narrowest `e` and `c` ranges (grid search, `e` at most 1) that contain every point are

| Backend | `e` | `c` (us/layer) | mean band width (hi/lo) |
|---|---|---|---|
| Metal (14 configurations, 40 points) | 0.86-0.97 | 140-180 | 1.16x |
| CUDA (5 cards, 8 points) | 0.98-0.99 | 45-105 | 1.40x |

Shipped widened (Metal 0.84-0.97 / 130-190, CUDA 0.95-1.00 / 40-110) because every point
is one model shape (32 layers, 7B); other layer counts and sizes extrapolate. A test asserts
every point stays inside its band.

## Not covered (labelled in the output)

- **CPU-side reads.** Next to a dedicated GPU, host RAM is assumed 40-90 GB/s (dual-channel
  DDR4-3200 to DDR5-6000), since it cannot be detected; on unified memory the CPU cores are
  assumed to reach 1/3 to 2/3 of the bandwidth. No offload benchmark backs either yet.
- **Prompt processing** is compute-bound and not estimated.
- **vLLM and ExLlamaV3** get no estimate; the fit is llama.cpp only.
- **Batching**: one sequence (`-np 1`), as planned.

`loadmaster report` records measured generation and prompt speed with the plan, so each
submitted run is a new point to check and refit against.
