# 0007 — A 15% free-RAM safety margin on unified memory

Status: accepted · 2026-10-19 · Apple Silicon only; dedicated GPUs are unchanged

## Context

ADR 0005 made a unified-memory plan fit only if its RAM need is at most the RAM available
now (`kern.memorystatus_level`). That is a point-in-time figure: free RAM falls as apps open,
and Metal buffers are wired, so they cannot be paged out when it does. On an 8 GB M2 the
planner's default for Llama 3.2 1B was the full 131072 context, needing 3.63 of 4.00 GiB
free, with 0.37 GiB to spare. A similar small-model, large-context run is the recorded
crash in `validation/observed.toml`.

## Decision

On unified memory a plan should leave `UNIFIED_RAM_MARGIN_PCT` (15%) of available RAM
unused. The margin changes choices but never turns a fit into a refusal:

- Default context: the largest candidate that fits fully on the GPU *and* leaves the margin;
  only if none does, the largest that fits. The `ctx` reason names the larger context that
  fits without the margin.
- KV type at a given context: a type that only fits under the margin is passed over for the
  next one (f16 to q8_0, never to q4_0 unless forced) that leaves it; the trial is listed as
  "(under the RAM margin)".
- A chosen plan still under the margin (an explicit `--ctx`, or nothing better) gets a
  `tight on RAM` warning with the largest `--ctx` that leaves the margin.

15% is a judgement, not a measurement: about 0.6 GiB on an 8 GB Mac with half its RAM
free, enough for opening a browser tab or two. Revisit once there are observed runs near
the limit.
