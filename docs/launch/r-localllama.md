# Draft: r/LocalLLaMA post

Post within 48 hours of the next major open-weights release (spec 3.9), once RESULTS.md has
real numbers. Everything in [brackets] must come from RESULTS.md or docs/validation.md;
delete any claim that does not have a number behind it.

---

**Title:** I tested tool calling on [20] popular GGUFs across llama.cpp, vLLM and LM Studio.
[N] of [M] combinations put tool calls in the wrong place. Here's the tool and the table.

Two questions I kept answering by trial and error: will this model fit at this context, and
does it actually emit tool calls once it runs? loadmaster answers both from the real model
file.

- `loadmaster plan model.gguf --ctx 32768` reads the GGUF header and prints the exact
  llama-server command: `-ngl`, `--n-cpu-moe`, KV type, flash attention, mmproj, with every
  term of the memory budget behind it (`--explain`). Weights and KV match llama.cpp's own
  buffer sizes [to within 0.3 MiB on N runs]; validation: [13/13 → N/N] fit/OOM cases.
- `loadmaster up` serves several models from one port and unloads by memory footprint, not
  model count. `loadmaster import llama-swap config.yaml` converts your config.
- `loadmaster probe http://localhost:8080 --model X` runs ten tool-call and structured-output
  checks and classifies failures (`call_in_content`, `schema_dropped_silent`,
  `stream_truncated`, …). Exit code 0/1/2, so it works as a CI step.

What the table shows: [two or three findings straight from RESULTS.md, e.g. "vLLM's json_schema
was silently ignored for X", "parallel calls serialised on Y"].

Table: [link to RESULTS.md]. Repo: [link]. It's MIT/Apache, Rust, one binary. I'd like more
validation cases from other GPUs; docs/validation.md explains the safe way to record one.
