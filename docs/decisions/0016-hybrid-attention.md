# 0016 — Hybrid-attention models: KV on attention layers, fixed recurrent state

Status: accepted · 2026-11-06 · llama.cpp sources at commit 6c01da6

## Context

Qwen3.5, Qwen3.5-MoE and Qwen3-Next (GGUF archs `qwen35`, `qwen35moe`, `qwen3next`) mix
two kinds of layer. Every `full_attention_interval`th layer is ordinary GQA attention with
a KV cache. All the others are Gated DeltaNet (linear attention) layers, which keep a small
fixed state per sequence instead. loadmaster did not know these archs, so it warned
`architecture unknown` and gave every layer a KV cache. That overstated KV 4x: 4 GiB
instead of 1 GiB for Qwen3.5-9B at 32k context in f16. As a result,
`plan hf://Lucebox/Qwen3.5-9B-GGUF/Qwen3.5-9B-Q8_0.gguf` reported "doesn't fit" on a
24 GB Mac.

## Decision

- **Layer pattern.** A layer is recurrent when `{arch}.attention.recurrent_layers` says so.
  That key is a flag per layer, or one value for all. Without it, layer `il` is recurrent
  iff `(il + 1) % n != 0`, where `n` is `{arch}.full_attention_interval` (default 4). This
  is `load_hparams` in src/models/qwen35.cpp (lines 17-23), qwen35moe.cpp and
  qwen3next.cpp. In config.json, `layer_types` entries of `linear_attention` mark the
  recurrent layers; without them, `full_attention_interval` applies the same rule. The arch
  table (`arch.rs`) has a `RecurrentPattern` beside `SwaPattern`. `Layer::kind()` is
  `attention`, `sliding` or `recurrent`.
- **KV.** A recurrent layer has `k_dim = v_dim = 0`, so every per-layer KV sum skips it:
  footprint, vLLM pool and speed reads. Attention layers are sized as before (ADR 0004).
  `--explain` says `6 of 24 layers`.
- **Recurrent state.** For one sequence, a layer holds one row of `cache_r_l` (conv state)
  and one of `cache_s_l` (recurrent state). Both are f32, since llama.cpp's `type_r` and
  `type_s` are `GGML_TYPE_F32` in `llama_model::create_memory`. The row widths come from
  src/llama-hparams.cpp `n_embd_r` / `n_embd_s`:

  | Row | Elements |
  |---|---|
  | conv (`n_embd_r`) | (d_conv − 1) x (d_inner + 2 x n_group x d_state) |
  | state (`n_embd_s`) | d_state x d_inner |

  Here d_conv, d_inner, d_state and n_group are `ssm.conv_kernel`, `ssm.inner_size`,
  `ssm.state_size` and `ssm.group_count`. From config.json they are
  `linear_conv_kernel_dim`, `linear_value_head_dim x linear_num_value_heads`,
  `linear_key_head_dim` and `linear_num_key_heads`, as llama.cpp's converter maps them (conversion/qwen.py lines 406-418).
  The rows are allocated `mem_size x (1 + n_rs_seq)` times, with
  `mem_size = max(1, n_seq_max)` (src/llama-memory-recurrent.cpp line 102). `n_rs_seq`
  defaults to 0. So the term is `n_seq x Σ recurrent layers x (n_embd_r + n_embd_s) x 4 B`.
  It does not grow with context. Qwen3.5-9B: 24 x 2.09 MiB = 50 MiB per sequence.
- **Where it lives.** The state follows its layer's device, like KV. It is the term
  `recurrent` (`recurrent state` in `--explain`), or `recurrent_cpu` for CPU layers. Both
  are added only when the model has recurrent layers, so other plans keep the same terms
  and bytes.
- **Speed.** A recurrent layer reads its state once per generated token, in place of KV.
- **vLLM.** One sequence's KV adds the state at f32. This is an upper bound and is
  unvalidated.
- **Missing keys are an error.** A hybrid file without the `ssm.*` keys (or `linear_*` in
  config.json) is rejected, not planned with a zero state.

## Out of scope

- Speculative decoding's extra state rows (`n_rs_seq > 0`), and MTP layers (`n_layer_all`
  beyond `block_count`).
- Other recurrent and hybrid families (Mamba, RWKV, LFM2, Jamba, Kimi KDA, MiniMax). Their
  `n_embd_r` / `n_embd_s` differ, and their archs stay unknown until they get a row.
