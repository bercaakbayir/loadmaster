## What and why

<!-- One or two sentences. Link the issue if there is one: Fixes #123 -->

## Checklist

- [ ] `cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` pass
- [ ] `cargo xtask check-deps` passes (crate boundaries)
- [ ] New formulas, dependencies or CLI surface come with an ADR in `docs/decisions/`
- [ ] Constants taken from elsewhere cite their source
- [ ] Snapshot changes (`*.snap`) are intended and reviewed
- [ ] `CHANGELOG.md` has an entry under the next version
- [ ] Commit messages are conventional (`feat(plan): ...`, `fix(serve): ...`)
