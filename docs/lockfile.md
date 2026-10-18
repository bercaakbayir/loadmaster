# models.lock reference

`models.lock` is TOML, written by `loadmaster init`, `add`, `plan --update`, `import` and
`up --verify`, and meant to be committed. Unknown keys are errors. Files loadmaster writes
round-trip byte for byte; hand edits are fine and are re-validated on every load.

Plan-derived fields are **frozen**: `ctx`, `footprint_gb` and `flags` change only with
`loadmaster plan --update <name>`, which prints a diff.

```toml
version = 1                              # schema version; newer files are refused

[host]                                   # where the entries were planned (informational)
gpu = "NVIDIA GeForce RTX 4090"          # a different GPU at `up` is a warning
vram_gb = 24.0
headroom_gb = 0.78                       # shared reserve, counted once across models

[proxy]
listen = "127.0.0.1:8080"                # one OpenAI-compatible port for every model
idle_unload_s = 600                      # unload after this long without requests; 0 = never
load_timeout_s = 300                     # a request waits this long for a load, then 503

[[model]]
name = "qwen3-coder"                     # what clients send as `model`
source = "hf://unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
sha256 = "..."                           # optional; from the store when it names files by hash
runtime = "llama.cpp"                    # llama.cpp | vllm | exl3
runtime_bin = "llama-server"             # on PATH, or absolute
ctx = 65536                              # frozen
footprint_gb = 21.4                      # frozen; GPU GiB including headroom
flags = ["-ngl", "49", "--n-cpu-moe", "12", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0", "-np", "1", "--jinja"]
mmproj = "..."                           # optional vision projector (llama.cpp --mmproj)
env = ["CUDA_VISIBLE_DEVICES=0"]         # optional KEY=VALUE for the runtime process
keep_warm = true                         # preloaded, never idle-unloaded, evicted last
aliases = ["gpt-4o", "coder"]            # other names routed to this model
idle_unload_s = 0                        # optional per-model override of proxy.idle_unload_s

[model.verify]                           # written by `loadmaster up --verify`
probe_version = "0.3.0"
verdict = "partial"                      # pass | partial | fail
failures = ["parallel_serialised"]
runtime_version = "llama.cpp b6700"      # `status` marks the verdict stale if this changes
checked_at = "2026-12-01T10:00:00Z"
```

## Sources

| Form | Meaning |
|---|---|
| `hf://org/repo/file.gguf` | a GGUF in the Hugging Face cache |
| `hf://org/repo` | a safetensors model directory in the HF cache (vLLM, exl3) |
| `ollama://name:tag` | a model in the Ollama store |
| `/path/to/model.gguf` or a directory | a file or directory on disk |

## What `up` runs

`flags` never contain the model, context, host or port; `up` adds them:

- llama.cpp: `runtime_bin -m <model> [--mmproj <mmproj>] -c <ctx> <flags> --host 127.0.0.1 --port <port>`
- vLLM: `runtime_bin serve <model> --max-model-len <ctx> --served-model-name <name> <flags> --host 127.0.0.1 --port <port>`
- exl3: `runtime_bin <flags> --model-dir <dir> --model-name <name> --max-seq-len <ctx> --host 127.0.0.1 --port <port>`
  (set `runtime_bin` to your tabbyAPI start command)

Each model may use `footprint_gb - headroom_gb`; the sum of resident models stays within the
GPU's usable memory minus one `headroom_gb` (on Apple Silicon, also within free RAM). When a
request needs a model that does not fit, the least recently used idle model is unloaded
first, `keep_warm` models last, and models serving a request never.
