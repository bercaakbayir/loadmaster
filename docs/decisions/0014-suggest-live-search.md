# 0014 — `suggest`: a live Hub search, not a catalog

Status: accepted · 2026-10-20 · Hub API behaviour checked live on that date

## Context

Users ask which models their machine and runtime can run. A catalog shipped with loadmaster
would need maintaining (a scheduled refresh, reviews) and would always lag new releases.

## Decision

- **Search the Hub live** with `GET /api/models`. Verified behaviour (anonymous requests,
  2026-10-20; the endpoint is not in the Hub's OpenAPI spec):
  - `filter` matches tags and repeats as AND (`gguf`, `safetensors`, `text-generation`,
    `image-text-to-text`, `feature-extraction`, `sentence-similarity`);
  - `search` matches the repo id loosely; `sort` takes `trendingScore`, `downloads`,
    `createdAt` (descending only); `limit` caps at 1000; `library` is ignored;
  - with `expand[]`, only `id` and the expanded fields come back; `gguf` gives `total`
    (parameters) and `architecture` for one representative file; `safetensors` gives counts
    per dtype; `baseModels` gives the parent and the relation (`quantized`, `finetune`);
  - rate limit: 500 API requests per 5 minutes anonymous, more with a token
    (https://huggingface.co/docs/hub/rate-limits); 429 when exceeded.
- **Known publishers by default.** Trending GGUF is dominated by community fine-tunes, and a
  fine-tuner's GGUF has its own fine-tune as base model, so "same organisation as the base"
  does not separate them. Two short lists in `suggest.rs` do: quantizers and model makers.
  They change rarely; `--any-publisher` turns the filter off.
- **Size before headers.** A repo's file list rules out quants bigger than the machine can
  hold; headers (5-15 MB each) are fetched only for the rest, biggest model first, stopping
  once `--limit` models fit on the GPU.
- **Ranking:** fits on the GPU, then with offload; then parameter count descending (a rough
  proxy for quality); then the Hub's order.
- **Planning reuses `plan hf://org/repo`** (`plan_repo_quants`), so a suggestion and its
  `exact plan` command agree.

## Consequences

- `suggest` needs the network (or a cache from an earlier run) and downloads headers.
- New models appear as soon as the Hub lists them; architectures loadmaster does not know
  yet show as unreadable until its tables catch up.
- Non-model GGUFs found along the way (`imatrix*.gguf`, `MTP/mtp-*.gguf`) are excluded from
  quant tables; a quant below 1.2 bits per weight of the Hub's parameter count is dropped.
