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

On the machine under test, with llama.cpp and a GGUF model:

```sh
loadmaster report Qwen3-8B-Q4_K_M.gguf --ctx 32768
```

It plans the model, shows the exact `llama-server` command and asks before starting it,
waits for the load, times one 128-token request, and stops the server. The report records
every buffer llama.cpp allocated per device, `fits` or `oom`, generation and prompt speed,
and what loadmaster predicted, with no paths, hostnames or user names, and prints a link
that opens a prefilled GitHub issue. Nothing is uploaded unless you submit that issue.

- **Already serving with `loadmaster up`?** Add `"-lv", "4"` to the model's `flags`, load
  it once, then `loadmaster report --from-log <name>`.
- **On a dedicated GPU,** check a "does not fit" prediction with `--try-anyway` (every layer
  on the GPU; a failed load is just a failed process there).
- **On Apple Silicon** report refuses plans that do not fit or leave under 15% of free RAM
  unused (`--unsafe-unified` overrides), and stops the run if free RAM falls below that: a
  real out-of-memory there can reset the machine (it did once, see `observed.toml`).

Good cases sit near the boundary: for each (model, GPU), the largest context that loads and
the next step up that does not.

Maintainers turn a submitted report into a case with
`cargo xtask intake <report.toml | issue-body.md>` (`--check` to validate only, `--add-speed`
to add its speed to `validation/speed.toml`), then `cargo xtask fetch-headers && cargo xtask
validate`. For the llmfit and `llama-server --fit` columns, run those tools on the same case
and add `llmfit = "fits" | "no"` and `llama_fit = "fits" | "no"`.
