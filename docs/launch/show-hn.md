# Draft: Show HN

Same timing and rule as the r/LocalLLaMA draft: numbers only from RESULTS.md and
docs/validation.md.

---

**Title:** Show HN: loadmaster – will a local LLM fit, and does its tool calling actually work?

loadmaster is a CLI for people running models locally behind OpenAI-compatible endpoints.

1. **Plan.** It reads the GGUF or safetensors header and computes the GPU footprint term by
   term (weights per layer, KV cache incl. sliding-window and MLA layers, compute buffer,
   vision projector, headroom), then emits the exact llama.cpp / vLLM / tabbyAPI launch
   command. It follows llama.cpp's real placement rules; on [N] measured runs weights and KV
   match llama.cpp's reported buffers [exactly].
2. **Serve.** A supervisor puts several models behind one port and evicts by footprint
   (LRU, pinned models last, never mid-request), with a committed `models.lock` so the setup
   is reproducible.
3. **Probe.** Declarative fixtures check tool calls, parallel calls, JSON-schema output and
   streaming against any endpoint and classify failures. [Headline finding from RESULTS.md.]

One thing I learned building it: on Apple Silicon, "fits in the Metal working set" is not
enough. A run that did exactly that took an 8 GB Mac down, because Metal buffers are wired
and free RAM was low. loadmaster now plans against the kernel's own available-memory figure
(ADR 0005).

[Repo link] · [RESULTS.md link]
