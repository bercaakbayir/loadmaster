# Contributing

## Report a run (2 minutes, no code)

The single most useful contribution: every prediction loadmaster makes is checked on very
little hardware so far, and a report checks one on yours.

```sh
loadmaster report path/to/model.gguf --ctx 32768
```

It shows the exact `llama-server` command and asks before starting it, loads the model once,
times one short request, stops the server, and writes `loadmaster-report-*.toml`: what was
predicted next to what llama.cpp actually allocated and how fast it ran. The file has no
paths, hostnames or user names; read it, then open the printed link to submit it as a GitHub
issue. loadmaster never uploads anything itself.

Most useful: models near the limit of your GPU (the largest context that loads, and the next
one up), MoE models, vision models, and any GPU or Mac not yet in
[docs/validation.md](docs/validation.md). On a dedicated GPU, `--try-anyway` checks a "does
not fit" prediction too. On Apple Silicon, `report` only runs plans with free-RAM headroom.

## Code

- `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace`
  must pass; no test needs a GPU or downloads a model.
- `cargo xtask check-deps` enforces the crate boundaries in `docs/decisions/0002-dependencies.md`.
- New formulas, dependencies or CLI surface come with an ADR in `docs/decisions/`, and
  constants taken from elsewhere cite their source.
- Conventional commits (`feat(plan): ...`, `fix(serve): ...`).

## Maintainers: turning a report into a case

```sh
cargo xtask intake issue-body.md --check     # validate only
cargo xtask intake issue-body.md --add-speed # case + report file + speed point
cargo xtask fetch-headers && cargo xtask validate
```

Reports whose observed result is not `fits` or `oom` (a timeout, an aborted run) are kept in
the issue but are not ground truth.
