# 0005 — Fitted compute buffer, and unified memory counts against free RAM

Status: accepted · 2026-10-18 · Metal only so far; re-fit when CUDA observations exist

## Compute buffer

The spec's §7 compute formula (`n_batch x (n_embd + n_vocab) x 4 + 128 MiB`) was 3-5x
above what llama.cpp allocates: llama.cpp computes logits for one output token during graph
reservation, and the buffer is the *peak* of live tensors, not their sum. Fitted on 28
observed runs of Llama 3.2 1B and Qwen3 0.6B (llama.cpp build 11669, Metal), rounded up so
every run is covered:

```
flash attention:  16 MiB + 2 B x ub x ctx                       (mask, live throughout)
                  + max( 16 B x ub x ffn_width,                  (FFN peak)
                         12 B x ub x (n_embd + q + k + v)
                         + 2 B x ctx x (k + v) )                 (attention peak)
no flash attn:    32 MiB + ub x ctx x (4 B x n_head + 4 B)       (f32 scores and mask)
host side:        2 B x ub x ctx + 16 B x ub x n_embd            (all layers on GPU; exact)
                  GPU formula                                    (any layer on CPU)
```

`ffn_width` is the dense FFN width, or `n_ff_exp x (n_expert_used + n_shared)` for MoE,
whichever is larger. The MoE form is **not yet observed**.

Accuracy on the observations: never below observed; typically 10-20 MiB above, at most 1.33x
(ubatch 256). The ctx slope is exact on both models (`2 (k + v) + 2 ub` bytes per token).

## Unified memory

On 2026-10-18 a validation run (Qwen3 0.6B, ctx 40960, f16 KV) reset the Apple M2 8 GB
machine running it. llama.cpp wired ~5.1 GiB of Metal buffers with ~1.1 GiB of RAM free.
loadmaster had said "fits", because it compared only against the Metal working-set limit
(5.33 GiB) and merely warned about free RAM.

On unified memory the GPU side is the same RAM, and Metal buffers cannot be paged out. A
configuration now fits only if, in addition to the working-set limit,

```
(GPU footprint - headroom) + host footprint <= RAM available now (vm_stat)
```

Otherwise the result is `CannotFit` against "free RAM", with "free X GiB held by other
processes" as a fix when that covers the gap. Partial offload does not help on unified
memory and the search reports it as such. The incident is recorded in
`validation/observed.toml`, and a test asserts the planner refuses it.

## Consequences

- Plans on a busy Mac are smaller (or refused) than the Metal limit alone would allow. That
  is the point.
- "RAM available now" is a snapshot. A plan made with apps closed can fail later with them
  open; `loadmaster up` (phase 2) should re-check before launching.
