# Installation

## Prerequisites

FeRx requires the Enzyme Rust toolchain for automatic differentiation. This is a nightly Rust toolchain with Enzyme support.

### 1. Install the Enzyme Rust toolchain

```bash
rustup toolchain install enzyme
```

If the `enzyme` toolchain is not available via rustup, follow the [Enzyme project instructions](https://enzyme.mit.edu/Getting%20Started/RustInstallation/) to install it manually.

### 2. Clone and build

```bash
git clone https://github.com/insightrx/ferx-nlme.git
cd ferx-nlme

# Build in release mode (recommended for performance)
cargo build --release --features autodiff

# The binary is at target/release/ferx
```

### 3. Verify the installation

```bash
cargo run --release --features autodiff --bin ferx -- examples/warfarin.ferx --simulate
```

You should see output showing a successful model fit with parameter estimates.

## Build Options

```bash
# Debug build (faster compilation, slower execution)
cargo build --features autodiff

# Release build with fat LTO (slower compilation, fastest execution)
cargo build --release --features autodiff

# Check compilation without building
cargo check --features autodiff

# Run clippy lints
cargo clippy --features autodiff
```

## Dependencies

FeRx depends on the following crates (managed automatically by Cargo):

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (matrices, Cholesky) |
| `nlopt` | Nonlinear optimization (SLSQP, L-BFGS, MMA) |
| `rayon` | Parallel computation |
| `rand`, `rand_distr` | Random number generation (SAEM) |
| `csv` | CSV data file reading |
| `regex` | Model file expression parsing |

The `nlopt` crate requires the NLopt C library. On most systems this is handled automatically. If you encounter build errors related to NLopt, install it via your system package manager:

```bash
# macOS
brew install nlopt

# Ubuntu/Debian
sudo apt-get install libnlopt-dev

# Fedora
sudo dnf install NLopt-devel
```
