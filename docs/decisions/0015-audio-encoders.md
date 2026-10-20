# 0015 — Audio encoders in projector files

Status: accepted · 2026-10-20 · llama.cpp sources at commit f174553 (as ADR 0011)

## Context

llama.cpp runs speech-input models (Ultravox, Voxtral, Qwen2-Audio, Qwen2.5-Omni, Gemma 4)
through `tools/mtmd`, loading the audio encoder from the same `mmproj` file as a vision
encoder. loadmaster sized every projector as an image encoder. Checked against real headers
on 2026-10-20, that went wrong three ways:
- audio-only files were planned as one image under an unknown projector type;
- Gemma 4's file, which holds both encoders, had its audio encoder left out;
- Qwen2.5-Omni's `qwen2.5o` type fell back to the plain patch grid even for vision.

## Decision

- **Modality.** `clip.has_vision_encoder` and `clip.has_audio_encoder` say which encoders a
  file holds. A file with neither is an older vision projector. The type is
  `clip.projector_type`, else `clip.vision.projector_type` / `clip.audio.projector_type`
  (`clip_model_loader::load_hparams`). `qwen2.5o` means `qwen2.5vl_merger` for vision and
  `qwen2a` for audio (the `PROJECTOR_TYPE_QWEN25O` comment in clip-impl.h).
- **One 30 s clip is the unit.** clip warms the audio encoder up with
  `warmup_audio_size = 3000` mel frames (clip-model.h), and mtmd-audio.cpp splits longer
  audio into 3000-frame chunks: 30 s at a 160-sample hop and 16 kHz. Tokens per clip follow
  `clip_n_output_tokens`, with F = 3000:

  | Type | Rule |
  |---|---|
  | `ultravox`, `meralion` | ceil(F / stack) / 2 |
  | `voxtral` | ceil(F / stack) / 2 / 2 (avg pool) |
  | `qwen2a`, `musicflamingo` | F / 2 / 2 |
  | `qwen3a` | F / 100 x 13 |
  | `glma` | F / 2 / stack + 2 |
  | `voxtral_rt` | (802 − 2) / 2 / stack, its streaming step |
  | `lfm2a`, `d1omni_a` | three stride-2 halvings, rounding up |
  | `gemma4a` | two stride-2 convolutions |
  | other | F / 2, marked unknown |

  `stack` is `clip.audio.projector.stack_factor`.
- **Compute buffer.** It is sized with the vision encoder's flash-attention estimate
  (ADR 0011), applied to the frames the encoder runs on after its convolutions:
  - F / 2 for Whisper-style encoders;
  - the token count for `qwen3a`, `lfm2a` and `gemma4a`;
  - 400 for `voxtral_rt`.

  Width, FFN and head count come from `clip.audio.*`. It is unvalidated.
- **Both encoders are counted.** `clip_init` creates and warms up a separate context for
  each encoder, so their compute buffers add. The output-embedding buffer is the larger of
  one image and one clip, because mtmd encodes one chunk at a time.
- **Names.** The `mmproj` term is labelled `projector`. The new terms are `audio_compute`
  and `audio_compute_cpu`. Plans say `one 30 s clip` or `one image or 30 s clip`.
- **`suggest --kind audio`** searches the Hub's `audio-text-to-text`, `any-to-any` and
  `automatic-speech-recognition` pipelines, and the header decides. It covers llama.cpp
  only.

## Out of scope

- **Speech-to-text models (Whisper, Parakeet):** they run in whisper.cpp or NeMo, not
  llama.cpp, and their memory is rarely the question.
- **vLLM audio models.**
- **Splitting `lm report`'s clip buffers by modality.**

## Consequences

Real headers checked on 2026-10-20 give these token counts per 30 s:
- ggml-org Ultravox (stack 8): 187;
- Voxtral Mini (stack 4): 187;
- Qwen2.5-Omni 3B: 750;
- Gemma 4 E2B: 750, plus its vision encoder.

The compute estimates need a measured run; `lm report` on an audio model provides one. The
`mmproj` JSON fields moved under `vision` / `audio`.
