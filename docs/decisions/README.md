# Design decisions

One file per decision: the context, what was chosen, and why. New formulas, dependencies or
CLI surface come with a new one ([CONTRIBUTING.md](../../CONTRIBUTING.md)).

Older ADRs and the CHANGELOG mention "the spec" (or "spec §7", "task 1.11"). That was the
private build specification loadmaster started from; it is not published. Each ADR says what
the spec asked for where it matters, and the ADR, not the spec, is the decision.

| ADR | Decision |
|---|---|
| [0001](0001-name-and-surface.md) | Name, binary and CLI surface |
| [0002](0002-dependencies.md) | Dependency choices |
| [0003](0003-gguf-buffered-reads.md) | Read GGUF headers with buffered I/O, not memmap2 |
| [0004](0004-footprint-rules.md) | Footprint and placement rules for llama.cpp |
| [0005](0005-compute-buffer-and-unified-memory.md) | Fitted compute buffer; unified memory counts against free RAM |
| [0006](0006-release-targets.md) | Release targets: glibc on Linux, `lm` as a second binary |
| [0007](0007-unified-ram-safety-margin.md) | A 15% free-RAM safety margin on unified memory |
| [0008](0008-remote-headers-and-pretend-gpus.md) | Plan from Hub headers, and for pretend hardware |
| [0009](0009-speed-estimates.md) | Speed estimates from memory bandwidth |
| [0010](0010-validation-reports.md) | Validation reports from users |
| [0011](0011-vision-and-embeddings.md) | Vision projectors and embedding models |
| [0012](0012-multi-gpu.md) | Planning across several GPUs |
| [0013](0013-output-style.md) | Human-readable output |
| [0014](0014-suggest-live-search.md) | `suggest`: a live Hub search, not a catalog |
| [0015](0015-audio-encoders.md) | Audio encoders in projector files |
| [0016](0016-hybrid-attention.md) | Hybrid-attention models: KV on attention layers, fixed recurrent state |
