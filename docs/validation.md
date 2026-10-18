# Validating the planner

Two files hold ground truth:

- `validation/observed.toml`: buffer sizes llama.cpp reported for real runs, term by term.
  `cargo test -p loadmaster-plan --test observed -- --ignored` replays them; weights and KV
  must match within 1 MiB, the compute buffer must not be underestimated. It also holds
  **incidents**: configurations that took a machine down, which the planner must refuse.
- `validation/cases.toml`: whole runs, `fits` or `oom`, for the 30-case agreement table
  (spec task 1.11). `cargo xtask validate --markdown` prints the table for the README.

Neither needs the model files: `cargo xtask fetch-headers` downloads just the GGUF header of
every listed model (a few MB, by HTTP range request) into `validation/headers/`.

## Status (2026-10-18)

- 28 observed runs and 13 cases, all on one Apple M2 8 GB with llama.cpp build 11669
  (Metal). Weights and KV match exactly; the compute buffer is at most 1.33x high; 13/13
  cases agree.
- The unified-memory rule (ADR 0005) was chosen with these runs in view, so on that rule the
  13 cases are a consistency check, not an independent test.
- **No CUDA data yet.** The compute-buffer formula, headroom and NVML `free` budget are
  unvalidated on NVIDIA, as is everything about MoE compute buffers, vLLM and exl3.

## Current table

`cargo xtask validate --markdown`, 2026-10-18:

| case | GPU | ctx | truth | loadmaster | llmfit | llama.cpp --fit |
| --- | --- | ---: | --- | --- | --- | --- |
| llama3.2-1b-q8_0-m2-4k-f16 | Apple M2 8 GB | 4096 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-8k-f16 | Apple M2 8 GB | 8192 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-16k-f16 | Apple M2 8 GB | 16384 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-24k-f16 | Apple M2 8 GB | 24576 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-32k-f16 | Apple M2 8 GB | 32768 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-48k-f16 | Apple M2 8 GB | 49152 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-64k-f16 | Apple M2 8 GB | 65536 | fits | fits | - | - |
| llama3.2-1b-q8_0-m2-8k-q8_0 | Apple M2 8 GB | 8192 | fits | fits | - | - |
| qwen3-0.6b-q4_k_m-m2-4k-f16 | Apple M2 8 GB | 4096 | fits | fits | - | - |
| qwen3-0.6b-q4_k_m-m2-8k-f16 | Apple M2 8 GB | 8192 | fits | fits | - | - |
| qwen3-0.6b-q4_k_m-m2-16k-f16 | Apple M2 8 GB | 16384 | fits | fits | - | - |
| qwen3-0.6b-q4_k_m-m2-32k-f16 | Apple M2 8 GB | 32768 | fits | fits | - | - |
| qwen3-0.6b-q4_k_m-m2-40k-f16 | Apple M2 8 GB | 40960 | oom | oom | - | - |

loadmaster agrees with ground truth on 13 of 13 cases.

## Collecting a case

On the machine under test, with llama.cpp and a model file:

```sh
scripts/collect-case.sh Qwen3-8B-Q4_K_M.gguf 32768          # f16 KV
scripts/collect-case.sh Qwen3-8B-Q4_K_M.gguf 65536 q8_0
```

It starts `llama-server` with full offload, records `fits` (it reached "listening") or
`oom`, the buffer sizes llama.cpp logged, and what loadmaster predicted, and prints a
`[[case]]` block to paste into `validation/cases.toml`. Add a `header_url` (the HF
`resolve/main/...gguf` URL) so others can replay it.

Good cases sit near the boundary: for each (model, GPU), the largest context that loads and
the next step up that does not. On NVIDIA an OOM is a failed process; collect both sides.
**On Apple Silicon the script refuses configurations loadmaster predicts will not fit**: a
real OOM there can reset the machine (it did once, see the incident in `observed.toml`).

For the llmfit and `llama-server --fit` columns, run those tools on the same case and add
`llmfit = "fits" | "no"` and `llama_fit = "fits" | "no"`.
