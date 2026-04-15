# Software Development Life Cycle — ferx-nlme

## 1. Project Overview

**ferx-nlme** is a Rust-based Nonlinear Mixed Effects (NLME) modeling engine for population pharmacokinetics. It implements FOCE/FOCEI estimation methods with analytical PK solutions and an optional ODE solver.

| Attribute | Value |
|-----------|-------|
| Language | Rust |
| Current version | 0.1.0 (active development) |
| Binary name | `ferx` |
| Repository | GitHub (InsightRX/ferx-nlme) |
| Toolchain | Enzyme (custom Rust toolchain for automatic differentiation) |

## 2. Development Environment Setup

### Prerequisites

- Rust via the [Enzyme](https://enzyme.mit.edu/) toolchain (pinned in `rust-toolchain.toml`)
- Standard system C linker

### Build commands

```bash
# Debug build
cargo build

# Release build (with fat LTO)
cargo build --release

# Build with autodiff feature (default)
cargo build --release --features autodiff

# Compilation check (no artifact output)
cargo check

# Lint
cargo clippy
```

### Running

```bash
# Fit a model to data
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv

# Simulate from a model
cargo run --release -- examples/warfarin.ferx --simulate
```

## 3. Branching Strategy

| Branch | Purpose |
|--------|---------|
| `main` | Primary development branch; must always build cleanly |
| `feature/*` | Feature branches for new estimation methods, model types, or major changes |
| `gh-pages` | Auto-generated documentation site (mdBook output) |

### Workflow

1. Create a feature branch from `main` (e.g., `feature/gauss-newton`).
2. Develop and validate locally using example models.
3. Open a pull request targeting `main`.
4. Merge after code review and successful validation.

## 4. Code Quality

### Current practices

- **Linting**: `cargo clippy` for Rust idiom and correctness checks.
- **Compilation checks**: `cargo check` for fast feedback during development.
- **Formatting**: Rust default formatting via `rustfmt` (no custom configuration).
- **Code review**: Pull requests on GitHub before merging to `main`.

### Recommendations

- Add `rustfmt.toml` if the team wants to codify style preferences beyond defaults.
- Enforce `cargo clippy -- -D warnings` to treat warnings as errors in CI.
- Consider `cargo-audit` for dependency vulnerability scanning.

## 5. Testing & Validation

### Unit tests

The project has 57 unit tests covering core computational modules. Tests use the `approx` crate for floating-point comparisons.

```bash
# Run all unit tests (requires RUSTFLAGS for Enzyme autodiff)
RUSTFLAGS="-Z autodiff=Enable" cargo test --lib
```

**Tested modules:**

| Module | Tests | Coverage |
|--------|-------|----------|
| `stats/residual_error.rs` | 11 | Additive, proportional, combined error models; IWRES; MIN_VARIANCE floor |
| `pk/one_compartment.rs` | 13 | IV bolus, infusion, oral; singularity handling; guard clauses; predict dispatcher |
| `pk/two_compartment.rs` | 15 | IV bolus, infusion, oral; macro rates; bioavailability; guard clauses |
| `pk/mod.rs` | 4 | Superposition with multiple doses; future dose exclusion; compute_predictions |
| `estimation/parameterization.rs` | 7 | Pack/unpack round-trip; log-transform correctness; bounds; clamping |
| `ode/solver.rs` | 5 | Exponential decay, linear growth, two-state system; parameter passing |

### Validation models

End-to-end validation uses example models against known datasets (in `examples/` and `data/`):

| Model | Description |
|-------|-------------|
| `warfarin.ferx` | 1-compartment oral (warfarin PK) |
| `warfarin_saem.ferx` | 1-compartment oral with SAEM estimation |
| `two_cpt_iv.ferx` | 2-compartment IV bolus |
| `two_cpt_oral_cov.ferx` | 2-compartment oral with covariates (WT, CRCL) |
| `mm_oral.ferx` | Michaelis-Menten elimination (ODE-based) |

### Future testing roadmap

1. **Integration tests**: Add a `tests/` directory with end-to-end model fitting tests asserting parameter estimates within tolerance.
2. **Regression tests**: Automate comparison of fit results against stored baseline outputs.
3. **Property-based tests**: Consider `proptest` for numerical edge cases in PK solvers and likelihood computations.

## 6. Build & Release

### Build profiles

- **Debug**: Standard Rust debug build for development.
- **Release**: Fat LTO enabled (`lto = "fat"` in `Cargo.toml`) for maximum optimization.

### Versioning

The project uses [Semantic Versioning](https://semver.org/) (currently `0.1.0`). During the `0.x` phase, breaking changes may occur in minor releases.

### Release process (recommended)

1. Update version in `Cargo.toml`.
2. Update `CHANGELOG.md` with notable changes.
3. Create a git tag: `git tag -a v0.2.0 -m "Release v0.2.0"`.
4. Push tag: `git push origin v0.2.0`.
5. CI builds release binaries and creates a GitHub Release (when CI is implemented).

### Changelog

A `CHANGELOG.md` file should be maintained following [Keep a Changelog](https://keepachangelog.com/) format, documenting additions, changes, fixes, and breaking changes per release.

## 7. Documentation

### Sources

| Resource | Location | Purpose |
|----------|----------|---------|
| mdBook site | `docs/src/` | User-facing documentation, model DSL reference, estimation methods |
| README.md | Project root | Quick start, overview, model syntax examples |
| CLAUDE.md | Project root | Developer guidance, architecture, build commands |

### Building documentation

```bash
cd docs
mdbook build    # Output to docs/book/
mdbook serve    # Local preview at http://localhost:3000
```

### Deployment

The documentation site is deployed to the `gh-pages` branch and served via GitHub Pages.

## 8. CI/CD

A GitHub Actions CI pipeline runs on every push to `main` and on pull requests (`.github/workflows/ci.yml`).

### Pipeline stages

| Job | Command | Purpose |
|-----|---------|---------|
| **Check** | `cargo check` | Fast compilation verification |
| **Test** | `cargo test --lib` | Run 57 unit tests |
| **Clippy** | `cargo clippy -- -D warnings` | Lint with warnings-as-errors |
| **Format** | `cargo fmt -- --check` | Enforce consistent formatting |

All jobs use the Enzyme Rust toolchain with `RUSTFLAGS="-Z autodiff=Enable"`.

### Future CI additions

- **Validation**: Run example models and compare output to baselines
- **Docs**: Build mdBook and deploy to gh-pages on main branch pushes
- **Security**: `cargo audit` for dependency vulnerabilities
- **Release**: Tag push (`v*`) builds release binaries and creates a GitHub Release

## 9. Deployment & Packaging

### CLI binary

The `ferx` binary is the primary distribution artifact. It reads `.ferx` model files and NONMEM-format CSV data, and outputs:
- `{model}-fit.yaml` — parameter estimates
- `{model}-sdtab.csv` — per-subject diagnostics

### R package integration

An R package (`ferx`) wraps the Rust engine via the [extendr](https://extendr.github.io/) framework, providing an R interface for pharmacometricians who work in R.

### Future packaging considerations

- Pre-built binaries for Linux, macOS, and Windows via GitHub Releases
- Docker image for reproducible environments (especially for the Enzyme toolchain)
- Homebrew formula or cargo-binstall support for easier installation

## 10. Security & Compliance

### Dependency management

- Dependencies are specified in `Cargo.toml` with version constraints.
- Run `cargo audit` periodically to check for known vulnerabilities.
- Run `cargo update` to keep dependencies current within semver bounds.

### License

A license should be added to the repository root (`LICENSE` file) to clarify usage terms.

### Contributing

A `CONTRIBUTING.md` file should be created to document:
- How to set up the development environment (Enzyme toolchain)
- Code style expectations
- Pull request process
- How to run validation models

## 11. Development Workflow Summary

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Feature     │     │  Pull        │     │  Main       │
│  Branch      │────▶│  Request     │────▶│  Branch     │
│              │     │  + Review    │     │             │
└─────────────┘     └──────────────┘     └──────┬──────┘
                                                │
      ┌─────────────────────────────────────────┤
      │                                         │
      ▼                                         ▼
┌─────────────┐                          ┌─────────────┐
│  Validate   │                          │  Tag +      │
│  Examples   │                          │  Release    │
└─────────────┘                          └─────────────┘
```

1. **Plan**: Identify the feature or fix, create an issue if applicable.
2. **Branch**: Create `feature/<name>` from `main`.
3. **Develop**: Write code, run `cargo check` and `cargo clippy` iteratively.
4. **Validate**: Run example models against known datasets, compare results.
5. **Review**: Open PR, get code review, address feedback.
6. **Merge**: Squash-merge or merge into `main`.
7. **Release** (when ready): Tag, update changelog, build release artifacts.
8. **Document**: Update mdBook docs and README as needed.
