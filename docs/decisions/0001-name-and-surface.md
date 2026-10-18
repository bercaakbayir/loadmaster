# 0001 — Name, binary and CLI surface

Status: accepted · 2026-10-18

## Context

The build spec (`plan.md`) calls the tool `lx`. The project ships as **loadmaster**, after the
aircrew role that weighs the cargo, balances it within limits and signs off the preflight —
the same three jobs the tool does for a local LLM (fit, placement, conformance).

## Decision

- Crates: `loadmaster-{model,hw,plan,store,lock,serve,probe}`; the binary crate is `loadmaster`
  so `cargo install loadmaster` works. All names were free on crates.io on 2026-10-18.
- Binary: `loadmaster`. Release archives also install an `lm` symlink. Docs always use
  `loadmaster`.
- Subcommands keep the spec's names: `scan gpu plan init add up down swap status import probe`.
  The analogy appears only in the README and as the `WEIGHT` / `BALANCE` / `PREFLIGHT`
  section headings of `--explain` output.
- Lockfile: `models.lock`. State directory: `~/.loadmaster/` (`logs/`, `probe/`, `eval.db`).
  Environment variables use the `LOADMASTER_` prefix.
- License: MIT OR Apache-2.0.

Every reference to `lx`, `lx-*` or `~/.lx` in `plan.md` maps to the names above.

## Consequences

Renaming subcommands later is a breaking change after v0.1.0 and needs a new ADR.
