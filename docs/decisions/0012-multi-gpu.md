# 0012 — Planning across several GPUs

Status: accepted · 2026-10-19 · rules from llama.cpp f174553 and vLLM v0.32.0; per-GPU
compute buffers and pipeline parallelism are unvalidated

## Context

llama.cpp uses every visible GPU by default (`llama_model_params.devices` null), splitting
layers in proportion to each GPU's free memory (src/llama-model.cpp:1634). A single-GPU plan
on a two-GPU machine therefore described a configuration that never ran.

## Decision: llama.cpp

- **Selection.** `--gpu auto` (default) and `all` plan across every NVIDIA GPU; `--gpu 0,1`
  picks some; `--gpu 1` picks one. A plan that uses fewer GPUs than the machine has pins them
  with `--device CUDA<i>,...`.
- **Layer assignment.** llama.cpp places layer slot `il` (layers in order, then the output
  head as the last slot) on the first device whose cumulative `--tensor-split` share exceeds
  `il / slots`. loadmaster fills GPUs in order, each up to its own budget, and emits the slot
  counts as `--tensor-split a,b,...` with `--split-mode layer`: integer counts reproduce the
  assignment exactly.
- **Per GPU**: its layers' weights and KV (each layer's KV lives on its device,
  llama-kv-cache.cpp:216), plus a full compute buffer and the headroom on every GPU, and the
  vision projector on the first. A plan fits only if every GPU fits its own budget.
- **Unvalidated.** The per-GPU compute buffer is assumed to be the full single-GPU size, and
  llama.cpp enables pipeline parallelism when every layer is offloaded across several GPUs
  (llama-context.cpp:517), which copies split-boundary inputs `GGML_SCHED_MAX_COPIES` (4)
  times. Plans say so in a warning until reports measure it.
- **Device order.** NVML indexes GPUs in PCI bus order; CUDA numbers them fastest first
  unless `CUDA_DEVICE_ORDER=PCI_BUS_ID`. With different cards, plans warn to set it.
- **Speed.** Devices process each token in turn: the time is the sum of each GPU's bytes over
  its own bandwidth (ADR 0009), plus the per-layer overhead once.

## Decision: vLLM

Tensor parallel size: the largest of 8, 4, 2, 1 that is at most the GPU count and divides the
attention heads (vLLM's only check, config/model.py:1544; KV heads are replicated when fewer
than the size). Each GPU holds a 1/size shard of the weights plus the activation reserve and
any vision tower; `--gpu-memory-utilization` is one fraction of every GPU's total, so the
tightest GPU sets it and every GPU is sized like the smallest. Emits
`--tensor-parallel-size N`.

## Not covered

`loadmaster up` treats several GPUs as one pool (the sum of their budgets) and warns: which
model sits on which GPU is not modelled, so an eviction may free the wrong card. `init` and
`add` still plan lockfile entries on one GPU.
