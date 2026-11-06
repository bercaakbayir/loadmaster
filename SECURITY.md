# Security policy

## Supported versions

loadmaster is pre-1.0: only the latest release gets fixes.

## Reporting a vulnerability

Please report privately, not in a public issue: use **Report a vulnerability** on the
repository's [Security tab](https://github.com/bercaakbayir/loadmaster/security/advisories/new)
(GitHub private vulnerability reporting). Include the version (`loadmaster --version`), the
command or request that triggers it, and what an attacker gains.

You should get a first answer within a week. Once a fix is released, the advisory is published
with credit to you unless you prefer otherwise.

## What is in scope

- `loadmaster up`: the proxy and supervisor. It listens on `127.0.0.1` by default; anything
  that lets a request reach beyond the configured runtimes, read local files, or start
  arbitrary processes is a vulnerability.
- Parsing untrusted model files (GGUF, safetensors, `config.json`) and Hugging Face API
  responses: crashes, hangs or unbounded memory on crafted input.
- `loadmaster report`: anything that leaks paths, host names or user names into the report
  it writes, which it promises not to do.

Vulnerabilities in llama.cpp, vLLM or tabbyAPI themselves belong to those projects.
