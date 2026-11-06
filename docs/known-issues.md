# Known issues

`loadmaster probe` prints an upstream issue link when a failure matches a signature in
`crates/loadmaster-probe/fixtures/known_issues.toml` (failure class, optionally runtime and
model). Extra signatures can be passed with `--known-issues FILE`.

| Failure | Runtime | Model | Issue | Note |
|---|---|---|---|---|

The list is empty: entries are added only for issues confirmed against a probe failure, with
a link to the real upstream report. When `RESULTS.md` is first generated, each failure
signature it shows should get an entry here.

## Models loadmaster does not plan (loadmaster's own)

`plan` and `scan` say "not supported" for these, rather than reading them as broken files:

- Image and video generation models: diffusion GGUFs (`flux`, `sd3`, `wan`, ...) and
  diffusers repos (`model_index.json`, such as black-forest-labs/FLUX.2-dev).
- Speech-to-text models (Whisper, Parakeet).
- Encoder-decoder models (T5, BART), including sentence-transformers T5 such as gtr-t5.

## Speed estimates (loadmaster's own)

- Bands are fitted to one model shape (7B, 32 layers) on llama.cpp; other sizes extrapolate.
- CUDA rests on 8 points from 5 cards; the band is wider than Metal's.
- CPU-offload speed uses an assumed host RAM bandwidth (40-90 GB/s) that is not detected.
- Prompt processing is not estimated.

Run `loadmaster report` on your hardware to add a measured point (ADR 0009).

## Vision and audio models (loadmaster's own)

- The vision and audio encoders' compute buffers are estimates with no observed run behind
  them.
- `loadmaster report` does not yet tell the audio encoder's buffers apart from the vision
  encoder's.
- Speech-to-text models (Whisper, Parakeet) and vLLM audio models are not planned.
- Dynamic-resolution projectors (Qwen-VL, Pixtral) are planned at llama.cpp's default
  maximum image size; Llama 4 tiling is not modelled.

`loadmaster report` on a vision or audio model records the real buffers.

## Hybrid-attention models (loadmaster's own)

- One measured run backs the formulas: Qwen3.5-0.8B at 32k on Metal, where KV and the
  recurrent state match llama.cpp's buffers exactly (`validation/observed.toml`). Qwen3-Next
  and the MoE variant have no measured run, and neither does more than one sequence (`-np`).
- vLLM plans count the state at f32 per sequence, an upper bound; vLLM's own state dtype and
  its page alignment of attention blocks to the state size are not modelled.
- Speculative decoding's extra state rows (llama.cpp `n_rs_seq`) are not counted.

## Model search (loadmaster's own)

- `suggest` shows the Hub's parameter count, which for packed formats (NVFP4, some AWQ)
  counts packed values: a 35B model can show as 18B and rank lower than it should.
- GGUFs using ggml types newer than loadmaster's table (seen: 40 and 42 in NVIDIA Nemotron
  3.5 builds) are counted as unreadable, as are image generation and speech-to-text models.
- A first run downloads tens of MB of headers; see [docs/suggest.md](suggest.md).

## Multi-GPU (loadmaster's own)

- Split plans assume a full-size compute buffer on every GPU and do not model pipeline
  parallelism's extra copies; neither is measured yet.
- `loadmaster up` pools all GPUs into one budget; `init` and `add` plan on one GPU.
