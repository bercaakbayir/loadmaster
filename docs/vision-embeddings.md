# Vision and embedding models

## Vision models

A vision model is a language model plus a projector file (`mmproj*.gguf`). loadmaster finds
the projector next to a local model or in the same Hub repo, counts it in the budget, and adds
it to the command:

```console
$ lm plan hf://ggml-org/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf --pretend "rtx 3060" --ctx 16384
fits: 4.81 GiB of 11.51 GiB usable on NVIDIA GeForce RTX 3060 12GB (pretend); all 34 layers and the output head on GPU, f16 KV

llama-server --hf-repo ggml-org/gemma-3-4b-it-GGUF --hf-file gemma-3-4b-it-Q4_K_M.gguf --mmproj-url https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/mmproj-model-f16.gguf -c 16384 -ngl 35 -fa on -np 1 --jinja
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
fits: 3.71 GiB of 11.51 GiB usable on NVIDIA GeForce RTX 3060 12GB (pretend); …
llama-server … --mmproj-url … --no-mmproj-offload -c 16384 -ngl 35 -fa on -np 1 --jinja
```

Plans that offload layers or fall short suggest this themselves.

## Embedding models

`lm scan` labels them `embed`. `lm plan` gives them the embedding flags in place of chat ones:

```console
$ lm plan hf://CompendiumLabs/bge-small-en-v1.5-gguf/bge-small-en-v1.5-q8_0.gguf --pretend "rtx 3060"
fits: 0.83 GiB of 11.51 GiB usable on NVIDIA GeForce RTX 3060 12GB (pretend); all 12 layers on GPU, no KV

llama-server --hf-repo CompendiumLabs/bge-small-en-v1.5-gguf --hf-file bge-small-en-v1.5-q8_0.gguf -c 512 -ngl 13 -fa on -np 1 --embeddings --pooling cls -ub 512 -b 512
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

Design and llama.cpp sources: [ADR 0011](decisions/0011-vision-and-embeddings.md).
