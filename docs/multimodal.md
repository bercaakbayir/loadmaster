# Vision, audio and embedding models

## Vision models

A vision model is a language model plus a projector file (`mmproj*.gguf`). loadmaster finds
the projector next to a local model or in the same Hub repo, counts it in the budget, and adds
it to the command:

```console
$ lm plan hf://ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf --pretend "rtx 3060" --ctx 16384
Gemma 3 4b It  ·  Q4_K_M  ·  34 layers  ·  ctx 16384

✓ Fits on RTX 3060 12GB (pretend)
  memory     4.8 of 11.5 GiB
  layers     all 34 on GPU
  KV cache   f16
  projector  1.1 GiB on GPU, one image
  speed      ~81-105 tok/s at 16k context (estimate)

Run:
  llama-server --hf-repo ggml-org/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q4_K_M.gguf \
    --mmproj-url https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf \
    -c 16384 -ngl 35 -fa on -np 1 --jinja
```

`--explain` shows the two vision terms:

```
  vision projector     0.79 GiB   0.79 GiB weights + one image: 256 tokens ((896/14)^2 patches / 4^2, gemma3 projector) x n_embd 2560 x 2 B
  vision compute       0.31 GiB   estimate, unvalidated: 16 MiB + mask 2 B x 4096^2 patches + peak of FFN …
```

- **Image tokens** follow llama.cpp's rule for each projector type: Gemma 3, LLaVA,
  Qwen-VL, SmolVLM/Idefics3, InternVL, Pixtral, MiniCPM-V, Llama 4 and others. Models that
  accept any image size (the Qwen-VL family, Pixtral) are planned for llama.cpp's default
  maximum. An unknown projector type falls back to the plain patch grid and is named in the
  output.
- **Vision compute** is the encoder's working memory. It is an estimate that no real run has
  checked yet.

### Projector on the CPU

When memory is tight, `--no-mmproj-offload` keeps the projector and its encoder in system
RAM. Images encode more slowly, but the GPU gets that memory back:

```console
$ lm plan …gemma-3-4b-it-Q4_K_M.gguf --pretend "rtx 3060" --ctx 16384 --no-mmproj-offload
Gemma 3 4b It  ·  Q4_K_M  ·  34 layers  ·  ctx 16384

✓ Fits on RTX 3060 12GB (pretend)
  memory     3.7 of 11.5 GiB
  layers     all 34 on GPU
  KV cache   f16
  projector  1.1 GiB on CPU (--no-mmproj-offload)
  speed      ~81-105 tok/s at 16k context (estimate)

Run:
  llama-server --hf-repo ggml-org/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q4_K_M.gguf \
    --mmproj-url https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf \
    --no-mmproj-offload -c 16384 -ngl 35 -fa on -np 1 --jinja
```

Plans that offload layers or fall short suggest this themselves.

## Audio input

Speech-input models in llama.cpp (Ultravox, Voxtral, Qwen2-Audio, Qwen2.5-Omni, Gemma 4) use
the same `mmproj` file as vision models. loadmaster reads which encoders the file holds and
sizes each one with llama.cpp's own rules:

```console
$ lm plan hf://ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/Llama-3.2-1B-Instruct-Q8_0.gguf --pretend "rtx 3060" --ctx 8192
Llama 3.2 1B Instruct  ·  Q8_0  ·  16 layers  ·  ctx 8192

✓ Fits on RTX 3060 12GB (pretend)
  memory     3.8 of 11.5 GiB
  layers     all 16 on GPU
  KV cache   f16
  projector  1.4 GiB on GPU, one 30 s clip
  speed      ~157-199 tok/s at 8k context (estimate)

Run:
  llama-server --hf-repo ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF \
    --hf-file Llama-3.2-1B-Instruct-Q8_0.gguf \
    --mmproj-url https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf \
    -c 8192 -ngl 17 -fa on -np 1 --jinja
```

`--explain` shows the audio terms:

```
  projector            1.28 GiB   1.28 GiB weights + one 30 s clip: 187 tokens (3000 mel frames / stack 8 / 2, ultravox projector) x n_embd 2048 x 2 B
  audio compute        0.13 GiB   estimate, unvalidated: 16 MiB + mask 2 B x 1500^2 frames + …
```

- **Audio is planned per 30 s clip.** llama.cpp splits longer audio into 30 s chunks
  (3000 mel frames), so each further 30 s adds the same number of tokens to the context:

  | Projector | Tokens per 30 s |
  |---|---|
  | Ultravox, Voxtral | 187 |
  | Qwen2-Audio, Qwen2.5-Omni, Music Flamingo | 750 |
  | Gemma 4 | 750 |
  | Qwen3 audio | 390 |
  | LFM2 audio | 375 |

  An unknown audio projector is planned as 1500 tokens per clip and named in the output.
- **Vision and audio in one file** (Qwen2.5-Omni, Gemma 4): both encoders are counted,
  since llama.cpp reserves working memory for each. The plan says `one image or 30 s clip`.
- **Audio compute** is the encoder's working memory. Like vision compute, it is an estimate
  that no real run has checked yet.
- `--no-mmproj-offload` keeps both encoders on the CPU.
- `lm suggest --kind audio` finds speech-input models on the Hub ([suggest](suggest.md)).

### Not covered

Speech-to-text models such as Whisper and Parakeet run in their own runtimes
(whisper.cpp, NeMo), not llama.cpp, so loadmaster doesn't plan them. vLLM audio models
aren't planned yet either.

## Embedding models

`lm scan` labels them `embed`. `lm plan` gives them the embedding flags in place of chat ones:

```console
$ lm plan hf://CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --pretend "rtx 3060"
bge-small-en-v1.5  ·  Q8_0  ·  12 layers  ·  ctx 512

✓ Fits on RTX 3060 12GB (pretend)
  memory     0.8 of 11.5 GiB
  layers     all 12 on GPU
  embeddings cls pooling

Run:
  llama-server --hf-repo CompendiumLabs/bge-small-en-v1.5-gguf \
    --hf-file bge-small-en-v1.5-q8_0.gguf -c 512 -ngl 13 -fa on -np 1 --embeddings \
    --pooling cls -ub 512 -b 512
```

- **Detected** from the GGUF architecture (`bert`, `nomic-bert`, `jina-bert-v2`, …) or a
  stated pooling type, which also catches Qwen3-Embedding. For Hugging Face directories,
  detection uses sentence-transformers files such as `modules.json`.
- **BERT-style encoders** have no KV cache, and each input must fit in one batch, so `-b`
  and `-ub` equal the context: the longest input the server accepts.
- **Decoder embedders** (Qwen3-Embedding) keep a KV cache and are planned like chat models:

  ```
  llama-server --hf-repo Qwen/Qwen3-Embedding-0.6B-GGUF --hf-file Qwen3-Embedding-0.6B-Q8_0.gguf -c 8192 -ngl 29 -fa on -np 1 --embeddings --pooling last
  ```

- **vLLM**: `--runner pooling`, e.g.
  `vllm serve BAAI/bge-small-en-v1.5 --max-model-len 512 --gpu-memory-utilization 0.89 --runner pooling`.
- No speed estimate, since nothing is generated.

### Checking an embedding endpoint

```console
$ lm probe http://127.0.0.1:8080 --model bge-small --embeddings
```

This checks `/v1/embeddings`:
- the vector is finite and non-empty;
- the same text gives the same vector on every run;
- two texts in one request give two different vectors.

`lm up --verify` runs these checks for embedding models in the lockfile.

Design and llama.cpp sources: [ADR 0011](decisions/0011-vision-and-embeddings.md) (vision,
embeddings) and [ADR 0015](decisions/0015-audio-encoders.md) (audio).
