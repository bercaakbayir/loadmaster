# 0013 — Human-readable output

Status: accepted · 2026-10-20

## Context

Users found the output hard to read. `plan` packed five facts into one sentence per line,
warnings ran past 170 characters, the launch command was one 300-character line, and every
estimated alternative had a footnote. `scan` printed twelve columns and wrapped on a laptop
screen; `report` printed its whole 50-line file before the link.

## Decision

- **Verdict first.** `✓ Fits on 2× RTX 3090`, `✓ … with CPU offload`, `! … tight on RAM`,
  `✗ Doesn't fit: 20.2 GiB short on …`; for `report`, `✓ Prediction held` /
  `✗ Prediction was wrong`.
- **One fact per line** under the verdict (memory, layers, KV cache, projector, speed), one
  decimal in GiB. Hardware is named without the vendor prefix, repeats counted.
- **Short warnings.** A warning has a short form (`! split across GPUs: not validated yet`)
  for the summary and a detail for `--explain`; `--json` keeps both as one string. Code that
  acts on a warning matches its `WarningKind`, never its text.
- **The command stays copyable.** On a terminal it wraps between flags with ` \`; piped, it
  is one line; `plan --command` prints only that line.
- **Fewer rows by default.** Up to three alternatives that fit; `scan` shows eight columns
  and `--wide` the rest; skipped directories are counted, not listed.
- **Details live in `--explain`**, which keeps every term and every configuration tried.
- **Colour only where it displays.** The plan crate writes plain ANSI codes when asked; the
  binary prints through `anstream`, which drops them when stdout is not a colour terminal,
  under `NO_COLOR` or `CLICOLOR=0`, and with `--color never` (ADR 0002).

## Consequences

- Scripts should use `--json` (or `plan --command`); the human format may change.
- Snapshot tests pin the plain form (`Style::PLAIN`); a format change is reviewed in the
  `.snap` diffs.
