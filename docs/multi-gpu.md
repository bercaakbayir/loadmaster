# Several GPUs

## llama.cpp: layers split across cards

llama.cpp uses every visible GPU by default, so loadmaster does too. It fills the GPUs in
order, each up to its own budget, and writes the split into the command:

```console
$ lm plan hf://bartowski/Llama-3.3-70B-Instruct-GGUF/Llama-3.3-70B-Instruct-Q4_K_M.gguf --pretend "2x rtx 3090" --ctx 32768
Llama 3.3 70B Instruct  ·  Q4_K_M  ·  80 layers  ·  ctx 32768

✓ Fits on 2× RTX 3090 (pretend)
  memory     46.5 of 47.0 GiB   GPU0 23.4/23.5 · GPU1 23.1/23.5
  layers     all 80 on GPU
  KV cache   q8_0
  speed      ~16-19 tok/s at 32k context (estimate)

! split across GPUs: not validated yet

Run:
  llama-server --hf-repo bartowski/Llama-3.3-70B-Instruct-GGUF \
    --hf-file Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 32768 -ngl 81 --device CUDA0,CUDA1 \
    --split-mode layer --tensor-split 41,40 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
```

`--tensor-split 41,40` is a count of layer slots per GPU (80 layers plus the output head), so
llama.cpp places exactly the layers loadmaster planned.

Different cards work too. Layers that fit on neither go to the CPU:

```console
$ lm plan …Llama-3.3-70B-Instruct-Q4_K_M.gguf --pretend "rtx 4090 + rtx 3060" --ctx 16384
Llama 3.3 70B Instruct  ·  Q4_K_M  ·  80 layers  ·  ctx 16384

✓ Fits on RTX 4090 + RTX 3060 12GB (pretend) with CPU offload
  memory     34.4 of 35.0 GiB   GPU0 23.1/23.5 · GPU1 11.3/11.5
  CPU RAM    10.2 of 24.0 GiB available
  layers     62 of 80 on GPU, 18 on CPU
  KV cache   q8_0
  speed      ~3.2-5.9 tok/s at 16k context (estimate)

! mixed GPUs: run with CUDA_DEVICE_ORDER=PCI_BUS_ID
! split across GPUs: not validated yet

Run:
  llama-server --hf-repo bartowski/Llama-3.3-70B-Instruct-GGUF \
    --hf-file Llama-3.3-70B-Instruct-Q4_K_M.gguf -c 16384 -ngl 63 --device CUDA0,CUDA1 \
    --split-mode layer --tensor-split 44,19 -fa on -ctk q8_0 -ctv q8_0 -np 1 --jinja
```

## Choosing GPUs

| `--gpu` | Plans on |
|---|---|
| `auto` (default) | every NVIDIA GPU, as llama.cpp does; otherwise the largest GPU |
| `all` | every GPU |
| `0,1` | those GPUs (indices from `lm gpu`) |
| `1` or `3090` | one GPU, by index or part of its name |

A plan that uses fewer GPUs than the machine has adds `--device CUDA<i>`, so llama.cpp doesn't
spread the model over cards it wasn't planned for.

`--pretend` takes `2x rtx 3090`, `rtx 4090 + rtx 3060`, or `2x 24GB`. Mixing NVIDIA with Apple,
or several unified-memory chips, is rejected.

## vLLM: tensor parallel

vLLM plans use the largest tensor-parallel size out of 8, 4, 2 and 1 that fits the GPU count
and divides the model's attention heads:

```console
$ lm plan hf://Qwen/Qwen2.5-14B-Instruct --runtime vllm --pretend "2x rtx 3090" --ctx 16384
Qwen2.5-14B-Instruct  ·  BF16  ·  48 layers  ·  ctx 16384

✓ Fits on 2× RTX 3090 (pretend)
  memory     46.7 of 47.0 GiB   GPU0 23.3/23.5 · GPU1 23.3/23.5
  layers     all 48 on GPU
  KV cache   auto

Run:
  vllm serve Qwen/Qwen2.5-14B-Instruct --max-model-len 16384 --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.94
```

`--gpu-memory-utilization` is a single fraction applied to every GPU, so the tightest GPU
decides it.

## Limits

- **Not validated on real hardware.** Each GPU is assumed to need a full-size compute buffer.
  llama.cpp's pipeline parallelism, which it turns on when every layer is on a GPU, may use
  more. Plans warn about this.
- **`lm up` treats the GPUs as one pool.** Its budget is their sum, and it doesn't know which
  model sits on which card, so an eviction may free the wrong one. `init` and `add` plan
  lockfile entries on one GPU.
- NVIDIA only. ROCm and several Macs are not planned.

If you run a split model, `lm report` records each GPU's real buffers and helps fix these
([CONTRIBUTING.md](../CONTRIBUTING.md)). Design and sources:
[ADR 0012](decisions/0012-multi-gpu.md).
