# 0002 — Dependency choices

Status: accepted · 2026-10-18

Pinned in `[workspace.dependencies]`; substituting any of these needs a new ADR.

| Need | Crate | Why |
|---|---|---|
| CLI | `clap` 4 (derive) | standard, generates help and completions |
| Lockfile | `serde` + `toml` | human-diffable, committed file |
| GGUF reads | std buffered reads (see ADR 0003) | header and tensor table only |
| safetensors | `serde_json` on the header (the format is a u64 length + JSON; the `safetensors` crate is not needed) |
| Chat templates | `minijinja` | Jinja-compatible, with a llama.cpp shim |
| NVIDIA | `nvml-wrapper` behind `cuda` feature | no CUDA toolkit needed to build the default binary |
| Proxy | `axum` + `tokio` + `hyper` | streaming bodies without buffering |
| Probe results | `rusqlite` (bundled) | single static binary, no system SQLite |
| HTTP client | `reqwest` with rustls | no OpenSSL dependency for static builds |
| Hashing | `blake3` | chat-template hashes |
| Errors | `thiserror` in libraries, `anyhow` in the binary | typed errors at crate boundaries |
| Terminal colour | `anstream`, binary only (already built through `clap`) | drops styles when stdout is not a colour terminal, under `NO_COLOR`, and on old Windows consoles; libraries emit plain ANSI |
| Tests | `insta`, `proptest` | snapshot outputs, arithmetic properties |

Pending: Apple `recommendedMaxWorkingSetSize` needs Metal FFI (likely `objc2-metal`). It gets
its own ADR when `loadmaster-hw` lands the Apple provider.
