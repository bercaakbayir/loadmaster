# 0010 — Validation reports from users

Status: accepted · 2026-10-19

## Context

Every footprint and speed number loadmaster prints was checked on one 8 GB Apple M2. The
validation set grows only if collecting a case is easy and safe for anyone, and
contributing one needs no account setup beyond a browser.

## Decision

- **One command, `loadmaster report`.** It plans the model, runs the planned `llama-server`
  command (plus `-lv 4`, the only verbosity at which llama.cpp prints buffer sizes) after
  asking, waits for ready or failure, times one request (`/completion`, 512 words of prompt,
  128 tokens, temperature 0, seed 42, no prompt cache) and stops the server. It replaces
  `scripts/collect-case.sh`.
- **Log parsing in Rust** against llama.cpp's format strings at commit f174553 (cited in
  `fixtures/llama-log/README.md`): `model`, `KV`, `RS`, `compute` and `output` buffers per
  device, the vision projector's compute buffer and `model size`, the build, and the outcome
  ("model loaded" / "listening on", allocation failures, "exiting due to model loading error").
- **Its own report schema** (`[report] [model] [hardware] [prediction] [observed]`), not
  `cases.toml` rows: it carries more (per-device buffers, speed, prediction terms) and stays
  stable if the validation files change. `cargo xtask intake` converts it.
- **Nothing is uploaded.** The report is a local file; a prefilled GitHub issue link
  (`?template=validation.yml&report=...`, under 7,500 characters, else "paste the file")
  lets the user review and submit it themselves. No token, no endpoint to run.
- **What the terminal shows** (amended 2026-10-20): a verdict (`✓ Prediction held`, `✗
  Prediction was wrong`) and a predicted / observed table of GPU buffers, speed and load
  time, then the file name and the link. The whole file was printed at first; it is about 50
  lines and buried the result. It is now printed only with `--show`; the privacy review
  still happens, because the issue page shows the whole file before it is submitted.
- **Privacy by construction.** Model paths become `<model>`/`<mmproj>`, any other absolute
  path `<path>`, the runtime is named by file name only; the model is identified by sha256
  (reusing the HF or Ollama blob name when it already is one) and its store source.
- **Safety.** On unified memory, report runs only plans that fit within the 15% free-RAM
  margin (ADR 0007) unless `--unsafe-unified`, never a plan that does not fit, and polls free
  RAM during the load, stopping the runtime if it falls below that margin. On a dedicated GPU,
  `--try-anyway` loads every layer to check a "does not fit" prediction: a failed load there is
  only a failed process.
- **Not ground truth:** timeouts, aborted runs and pretend hardware. Intake rejects them.

## Consequences

- `loadmaster up` logs carry buffer sizes only if the user adds `-lv 4` to a model's flags
  (`report --from-log` says so); `up` does not add it, since `-lv 4` is verbose for a
  long-running server.
- Tested entirely with the fake runtime replaying log fixtures: no model is loaded.
