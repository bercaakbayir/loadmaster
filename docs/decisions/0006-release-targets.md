# 0006 — Release targets: glibc on Linux, `lm` as a second binary

Status: accepted · 2026-10-18

## Linux: glibc, not musl

The spec asks for static musl binaries on Linux. NVIDIA detection uses NVML, which
`nvml-wrapper` loads with `dlopen("libnvidia-ml.so.1")` at runtime so the binary builds and
runs without the CUDA toolkit. A fully static musl binary cannot `dlopen` the glibc-linked
NVIDIA driver library, so on every Linux GPU box it would report "no GPU". Linux releases are
therefore `x86_64-unknown-linux-gnu` and `aarch64-unknown-linux-gnu`: every Rust dependency
is still statically linked (rustls, bundled SQLite), only libc and the NVML driver library
are dynamic. A musl build without the `cuda` feature can be added later for CPU-only and
container use.

## `lm`

ADR 0001 planned `lm` as a symlink. cargo-dist and `cargo install` cannot create
symlinks, so `lm` is a second `[[bin]]` in the `loadmaster` crate that calls the same entry
point. Every install path (`cargo install loadmaster`, Homebrew, the shell installer) gets
both names.

## Windows

`x86_64-pc-windows-msvc` is in `targets` (v0.3, per the spec). Unix-only pieces are gated:
runtimes are stopped with a hard kill (no SIGTERM), the supervisor exits on Ctrl-C or
`loadmaster down`, host memory comes from `Win32_OperatingSystem`, and the process-based
tests are `cfg(unix)`. The `windows` CI job builds the workspace and runs the unit tests;
nothing has been run on Windows by hand yet.

## Releasing

1. `cargo install cargo-dist --version 0.32.0 --locked`, then `dist generate` writes
   `.github/workflows/release.yml` from `dist-workspace.toml`.
2. Create the tap repository and the `HOMEBREW_TAP_TOKEN` secret.
3. `cargo publish` the crates in dependency order (`cargo xtask publish-order` prints it).
4. Push a `vX.Y.Z` tag.
