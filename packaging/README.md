# Packaging

| What | File | Status |
|---|---|---|
| Release binaries, shell + Homebrew installers | `dist-workspace.toml` (cargo-dist) | config written; run `dist generate` to create the release workflow (ADR 0006) |
| Docker (CUDA / CPU) | `docker/Dockerfile` | written, **not yet built or run** |
| systemd | `systemd/loadmaster.service` | written, **not yet tested** on a systemd host |
| AUR | `aur/PKGBUILD` | written, **not yet built** with makepkg; set `sha256sums` on release |

The systemd unit uses the `/etc/systemd/system` path in its install comment, but
`/usr/lib/systemd/system` (as the AUR package installs it) works the same.
