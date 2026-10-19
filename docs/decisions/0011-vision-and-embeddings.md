# 0011 — Vision projectors and embedding models

Status: accepted · 2026-10-19 · rules from llama.cpp commit f174553 and vLLM v0.32.0;
the vision compute estimate is unvalidated

## Vision: tokens per image

`loadmaster-model/src/vision.rs` mirrors `clip_n_output_tokens` (tools/mtmd/clip.cpp:4333)
per `clip.projector_type`, from the GGUF keys `clip.vision.image_size`, `patch_size`,
`projector.scale_factor`, `spatial_merge_size` and `clip.minicpmv_query_num`:

| Projectors | Tokens per image |
|---|---|
| mlp, mlp_norm, janus_pro, phi4 | `(image/patch)^2` |
| ldp, ldpv2, adapter | grid / 4 |
| gemma3 (scale 4), idefics3, internvl, llama4, cohere2v, nemotron (scale 2) | grid / scale^2; scale from metadata when present |
| resampler (MiniCPM-V) | `minicpmv_query_num`, default 64 |
| qwen2vl/qwen2.5vl/qwen3vl_merger, glm4v | dynamic; up to 4096 (llama.cpp's default maximum) |
| pixtral, lightonocr | dynamic; up to 1024 |
| kimivl 1024, lfm2 256, gemma4v 1120 | dynamic; default maximum |
| gemma3nv | `image/patch` |
| anything else | `(image/patch)^2`, with the projector named in the basis |

Dynamic projectors are planned at their default maximum (a large image); `--image-max-tokens`
in llama.cpp lowers it. Llama 4's tiling multiplies its per-tile count and is not modelled.
Checked on real headers (Qwen2.5-VL 3B, SmolVLM 256M, Pixtral 12B via `hf://`): the types and
keys are read as above.

## Vision: compute buffer

llama.cpp reserves the encoder's compute buffer once, for a warmup image: the projector's
`image_size` for fixed-size types, 46x46 merged tokens for the Qwen-VL family, 256 tokens for
Pixtral, Kimi-VL and Gemma 4 (clip.cpp:1411, 1764, clip-model.h:225). clip runs flash
attention whenever the backend supports it (`CLIP_FLASH_ATTN_TYPE_AUTO`, clip.cpp:3928), so
the estimate is ADR 0005's flash-attention shape with the warmup patches as one batch:

```
16 MiB + 2 B x patches^2 + max(16 B x patches x ffn, 12 B x patches x 4 x width + 4 B x patches x width)
```

It is a new footprint term, `vision compute`, labelled "estimate, unvalidated". `loadmaster
report` records clip's `compute buffer size` lines, so it can be fitted like ADR 0005 once
reports arrive. Images larger than the warmup may grow the buffer at encode time (not
verified).

## Vision: placement

The projector's weights and encoder go on the GPU by default, as in llama.cpp
(`mmproj_use_gpu`). `plan --no-mmproj-offload` moves both terms to the host side and emits
the flag; the alternatives offer it when the plan offloads layers, falls back to q4_0 KV, or
is tight on RAM, and a no-fit plan suggests it when it would free enough.
