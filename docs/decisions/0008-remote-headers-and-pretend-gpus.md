# 0008 — Plan from Hub headers, and for pretend hardware

Status: accepted · 2026-10-19

## Context

Trying loadmaster needed a model file on disk and the GPU in question. Both are what a
person deciding *which* model or GPU to get does not have yet.

## Decision: `hf://` reads headers from the Hub

- **Network only on an explicit `hf://`.** A path or a bare store name never touches the
  network. An `hf://` spec is resolved from the local HF cache first; only a miss goes to the
  Hub. `--offline` uses the header cache alone.
- **Headers only.** GGUF: growing range requests (4 MiB, doubling, 256 MiB cap) until the
  header parses, truncated to the start of tensor data; split models fetch every part's
  header, and a sibling `mmproj*.gguf` is picked by the same rule as on disk. Safetensors:
  `config.json`, the shard index, the chat template, and each shard's header (8 bytes, then
  the JSON). File sizes come from the tree API, so `complete` is computed, not assumed.
- **Same model code.** `ModelInfo::from_gguf_parts` and `from_hf_parts` take bytes; the disk
  loaders are thin wrappers, and equivalence tests on every fixture keep them identical.
- **Cache.** `~/.loadmaster/hf/<repo>/<rev>/`: headers never expire (the file at a path in a
  revision does not change underneath, and a commit hash never does); a branch's file list
  is re-fetched after 24 h. Writes go through a rename, so an interrupted fetch leaves nothing.
- **Auth and mirrors.** `HF_TOKEN`, else the `huggingface-cli login` token file; `HF_ENDPOINT`
  for mirrors (and tests). The Hub answers 401 for both missing and gated repos, and the error
  says so.
- **Commands.** llama.cpp gets `--hf-repo/--hf-file` (and `--mmproj-url`), so it downloads
  exactly the planned file; vLLM gets `vllm serve org/repo`. Lockfile entries still need the
  weights on disk.
- **Bare GGUF repos** print every quant: one header planned exactly (Q4_K_M, else the median
  size), the others scaled by file size via `alternatives::requantized`. On Llama 3.2 1B the
  estimates for IQ3_M, Q8_0 and f16 matched exact plans to within 300 bytes.

The HTTP client is `reqwest` (blocking, rustls), already in ADR 0002; `loadmaster-store`
gains it, as the crate that knows where models come from.

## Decision: `--pretend`

One flag, separate from `--gpu` (which keeps selecting among detected GPUs), so a pretend
4090 can never be confused with a real one. It takes a catalog name, a Mac with its RAM, or a
raw size, plus an optional `+<size>` for host RAM.

- Dedicated cards: VRAM as `nvidia-smi` reports it, minus 500 MiB for the driver and a
  desktop session (as `FakeHw::rtx_4090`); host RAM defaults to 32 GiB, 3/4 of it free.
- Macs: the Metal working-set default (ADR 0005) and 4 GiB held by a freshly booted macOS.
- GPU names carry ` (pretend)` and every basis starts with `pretend:`, so output and JSON
  are labelled without changes to the printers. `up` refuses `--pretend`.

Sizes are GiB whatever the unit says (vendors' memory "GB" are GiB), and need a unit, so
`4090` is a card, not 4090 GiB.
