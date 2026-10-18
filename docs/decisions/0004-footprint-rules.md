# 0004 — Footprint and placement rules for llama.cpp

Status: accepted · 2026-10-18 · to be re-checked against the validation set (task 1.11)

The spec's §7 formulas are the starting point. Where llama.cpp's source says otherwise,
loadmaster follows llama.cpp, because the goal is to predict what `llama-server` allocates.

## Placement (`llama_model::load_tensors`)

- `-ngl N` counts the output head as a layer and offloads it first: the output head and
  final norm go to the GPU whenever `N >= 1`, plus the **last** `min(N - 1, n_layer)`
  repeating layers. Full offload is `-ngl n_layer+1`. (Amended 2026-10-18: the first version
  of this ADR, written from memory of the source, said the head moves only when
  `N > n_layer`; observed runs at `-ngl 1, 8, 16` on llama.cpp build 11669 disproved it.)
- Token embeddings (`token_embd*`) always stay on the CPU, whatever `-ngl` is. (§7 says
  "unless -ngl covers all layers"; observed `CPU_Mapped` 266 MiB at every `-ngl`.)
- A tied output head (no `output.weight`) is loaded as a second copy of `token_embd` on the
  output device, so its bytes count on the GPU.
- `--n-cpu-moe k` keeps the routed-expert tensors (`*_exps*`) of layers `0..k` on the CPU.
- KV cache rows live on the device of their layer.

## Verification

`validation/observed.toml` records buffer sizes llama.cpp reported for real runs; the
`observed` test in `loadmaster-plan` replays them. As of 2026-10-18 (18 runs, one model,
Metal) weights and KV match to within 0.3 MiB on both devices, at `-ngl` 0, 1, 8, 16, 17.

## Terms

- **KV**: per layer `(row(K) + row(V)) × cells`, with ggml row sizes (q8_0 = 34 B per 32,
  q4_0 = 18 B per 32). `cells` = ctx padded to 256. Sliding-window layers use
  `min(cells, pad256(window × n_seq + n_ubatch))`, as llama.cpp's `size_swa`.
  MLA layers store K = `kv_lora_rank + rope_dim` and V = `kv_lora_rank`. This is
  conservative if the runtime keeps V as a view of K; validation decides.
- **Compute buffer**: uses `n_ubatch` (default 512), not `n_batch`, because llama.cpp sizes
  its graph per ubatch. Without flash attention the attention scores are f32 (4 B), not 2 B.
  Flash attention is always emitted `on` for llama.cpp. When no layer is on the GPU the
  GPU compute buffer is ~0 (observed 0.5 MiB at `-ngl 1`).
  Superseded by the fitted formula in ADR 0005.
- **Headroom** is part of the footprint `F`. `F` is compared with `usable` (NVML `free`, or
  the Metal working set), so headroom is subtracted exactly once. (§7 subtracts it from both
  sides.)
- `-np 1` is always emitted so the planned KV cache is the one allocated.

## Search order

1. Every layer on GPU with f16, then q8_0, then q4_0 KV (q4_0 gets a warning).
2. Offload with q8_0 KV. For MoE models first move routed experts of the fewest leading
   layers (`--n-cpu-moe k`); if even all experts on CPU do not fit, also drop layers.
   Dense models drop the fewest layers (`-ngl`).
3. Otherwise `CannotFit`, reporting the closest miss, which budget it misses (GPU or host),
   by how much, and up to two fixes: the largest context that fits fully on the GPU, freeing
   memory other processes hold when that covers the gap, and a smaller quant.

With no `--ctx`, the context is the training context, or the largest power of two below it
(down to 4096) that fits with every layer on GPU at f16 or q8_0 KV. A longer context with
q8_0 KV is preferred over a shorter one with f16.
