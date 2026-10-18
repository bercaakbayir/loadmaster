# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Cargo workspace with `loadmaster-{model,hw,plan,store,lock,serve,probe}` crates and the
  `loadmaster` binary (CLI surface for `scan`, `gpu`, `plan`; not yet functional).
- `cargo xtask check-deps` enforcing crate boundaries.
- CI (fmt, clippy, tests, MSRV 1.85, cargo-deny). ADRs 0001 (naming) and 0002 (dependencies).
