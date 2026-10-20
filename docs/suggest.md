# Finding models: `loadmaster suggest`

`suggest` answers "what can I run here?" from the Hugging Face Hub, live:

```console
$ lm suggest --pretend "rtx 4090" --ctx 32768
Models for RTX 4090 (pretend), 23.5 GiB usable  ·  llama.cpp  ·  ctx 32768  ·  trending GGUF on Hugging Face

MODEL                                        PARAMS  QUANT            SIZE  FIT      CTX      TOK/S
unsloth/Qwen3.6-35B-A3B-GGUF                  34.7B  UD-Q4_K_XL   20.8 GiB  ~gpu   32768   ~124-196
openbmb/MiniCPM-V-4.7-35B-A3B-gguf            34.7B  Q4_1         20.3 GiB  ~gpu   32768   ~125-199
unsloth/Qwen3.8-27B-GGUF                      27.3B  UD-Q5_K_XL   19.4 GiB  ~gpu   32768     ~33-40
unsloth/gemma-4-26B-A4B-it-GGUF               25.2B  UD-Q5_K_XL   19.8 GiB  ~gpu   32768   ~117-162
LiquidAI/LFM2.5-24B-A2B-GGUF                  24.0B  Q6_K         18.3 GiB  ~gpu   32768   ~137-231
JetBrains/Mellum2.1-12B-A2.5B-Thinking-GGUF   12.1B  BF16         22.6 GiB  ~gpu   32768   ~125-173

searched 120 repos · 4 too large · 3 duplicates · 85 by other publishers (--any-publisher) · 1 unreadable · 15 smaller, not needed · 6 gated (set HF_TOKEN)
exact plan: loadmaster plan hf://unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
```

(Captured 2026-10-20; the Hub's list changes daily.)

## Common uses

| You want | Run |
|---|---|
| The best models for this machine | `lm suggest` |
| A family or size | `lm suggest qwen`, `lm suggest 3b`, `lm suggest "gemma 12b"` |
| Vision or embedding models | `lm suggest --kind vision`, `lm suggest --kind embed` |
| vLLM instead of llama.cpp | `lm suggest --runtime vllm` |
| A GPU you don't have | `lm suggest --pretend "2x rtx 3090"` |
| The classics rather than this week's | `lm suggest --sort downloads` |
| More rows, or what doesn't fit | `--limit 20`, `--all` |
| Fine-tunes by anyone | `--any-publisher` |
| No network | `--offline` (reuses the last results) |
| Scripts | `--json` |

## How it works

1. **Search.** One request to the Hub's model listing (GGUF repos for llama.cpp, safetensors
   for vLLM), sorted by trending (default), downloads or newest. The listing already carries
   each model's parameter count and base model.
2. **Pick candidates.** By default only known publishers: quantizers (ggml-org, unsloth,
   bartowski, lmstudio-community) and model makers (Qwen, google, meta-llama, mistralai,
   nvidia, ...; the list is `MAKERS` in `crates/loadmaster/src/cmd/suggest.rs`, a one-line PR
   to extend). Trending GGUF is otherwise dominated by community fine-tunes. One repo per base
   model; gated repos only with `HF_TOKEN`. The first 40 (`--candidates`) are kept.
3. **Rule out by size.** Each repo's file list (one small request) gives every quant's size;
   if the smallest is bigger than the GPUs plus free RAM (on a Mac: the RAM free now), the
   repo is skipped without downloading anything else.
4. **Plan exactly.** The rest are planned from their headers, every quant of a GGUF repo
   like `lm plan hf://org/repo`, and the best quant kept: the largest that fits entirely on
   the GPU, else the smallest that runs with offload. `~gpu` means that quant's size was
   scaled from another quant's header.
5. **Rank.** Fits entirely on the GPU first, then with CPU offload; within each, the largest
   model first, then the Hub's order. Planning goes biggest first and stops once `--limit`
   models fit on the GPU, since nothing smaller could outrank them.

## Cost

- **Network:** a first run downloads roughly 50-150 MB of GGUF headers (a header holds the
  tokenizer, often 5-15 MB) and takes 10-20 s; vLLM runs read every shard's header and take
  longer. Searches are cached for an hour, file lists for a day, headers for good
  (`~/.loadmaster/hf`; delete it to reclaim the space), so repeat runs take seconds.
- **Rate limits:** anonymous users get 500 Hub API requests per 5 minutes; a run uses about
  40-50. On HTTP 429 `suggest` stops and says so; `HF_TOKEN` raises the limit.

## Limits

- **PARAMS is the Hub's count.** For packed formats (NVFP4, some AWQ) it counts packed
  values, so a 35B model can show as 18B; ranking uses the same number.
- **Unreadable repos** are counted, not shown: image and speech models in GGUF, and files
  using ggml types loadmaster does not know yet (some new NVIDIA formats).
- **Small machines:** trending candidates are often too large; when few fit, `suggest` says
  so. A search term (`lm suggest 1b`) or `--candidates 100` looks further.
- exl3 is not searched yet.

Design: [ADR 0014](decisions/0014-suggest-live-search.md).
