# 0001 — Name, binary and CLI surface

Status: accepted · 2026-10-18

## Context

The original build spec called the tool `lx`. The project ships as **loadmaster**, after the
aircrew role that weighs the cargo, balances it within limits and signs off the preflight —
the same three jobs the tool does for a local LLM (fit, placement, conformance).

## Decision

- Crates: `loadmaster-{model,hw,plan,store,lock,serve,probe}`; the binary crate is `loadmaster`
  so `cargo install loadmaster` works. All names were free on crates.io on 2026-10-18.
- Binary: `loadmaster`, plus an `lm` alias (a second binary, see ADR 0006). Docs always
  use `loadmaster`.
- Subcommands keep the spec's names: `scan gpu plan init add up down swap status import probe`.
  The analogy appears only in the README and as the `WEIGHT` / `BALANCE` / `PREFLIGHT`
  section headings of `--explain` output.
- Lockfile: `models.lock`. State directory: `~/.loadmaster/` (`logs/`, `probe/`, `eval.db`).
  Environment variables use the `LOADMASTER_` prefix.
- License: MIT OR Apache-2.0.

Every `lx`, `lx-*` or `~/.lx` name in that spec maps to the names above.

## Consequences

Renaming subcommands later is a breaking change after v0.1.0 and needs a new ADR.
