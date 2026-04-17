# Installation

FeRx requires a nightly Rust toolchain with the Enzyme LLVM plugin for automatic differentiation. As of 2026, Enzyme is not yet distributed via rustup, so a one-time plugin build is required.

This page covers:
- [Quick install (single user, dev machine)](#quick-install-single-user-dev-machine)
- [Shared install (multi-user server)](#shared-install-multi-user-server)
- [Building ferx-nlme from source](#building-ferx-nlme-from-source)
- [Installing the ferx R package](#installing-the-ferx-r-package)

---

## Quick install (single user, dev machine)

For a personal dev machine where you're the only user.

### 1. Install rustup + upstream nightly

**Do not use snap's rustup** — its filesystem confinement breaks on non-standard home directories.

```bash
# Remove snap rustup if you had it:
sudo snap remove rustup

# Official installer
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"

rustup toolchain install nightly
```

### 2. Install system dependencies and matching LLVM

Check which LLVM major version nightly needs:
```bash
rustc +nightly --version --verbose | grep LLVM
# e.g. "LLVM version: 22.1.2" — major is 22
```

Use that major version (`<MAJOR>`) below.

```bash
sudo apt install -y cmake ninja-build clang libssl-dev pkg-config \
                    python3 build-essential curl git libzstd-dev

# Install matching LLVM from apt.llvm.org (Ubuntu's defaults lag behind):
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh <MAJOR>

# Fix GPG keyring if apt warns:
sudo chmod 644 /etc/apt/trusted.gpg.d/apt.llvm.org.asc
sudo apt update

sudo apt install -y llvm-<MAJOR>-dev clang-<MAJOR>
```

### 3. Build and install the Enzyme plugin

```bash
git clone https://github.com/EnzymeAD/Enzyme /tmp/enzyme-build
cd /tmp/enzyme-build/enzyme
mkdir build && cd build

cmake -G Ninja .. \
  -DLLVM_DIR=/usr/lib/llvm-<MAJOR>/lib/cmake/llvm \
  -DENZYME_CLANG=OFF \
  -DENZYME_FLANG=OFF
ninja
# 15–30 min
```

Drop the built `.so` into nightly's target-specific sysroot. **This location is not obvious** — rustc looks in `lib/rustlib/<target>/lib/`, not just `lib/`:

```bash
SYSROOT=$(rustc +nightly --print sysroot)
TARGET=x86_64-unknown-linux-gnu   # adjust for other platforms

cp /tmp/enzyme-build/enzyme/build/Enzyme/LLVMEnzyme-<MAJOR>.so \
   $SYSROOT/lib/rustlib/$TARGET/lib/libEnzyme-<MAJOR>.so
```

Note the filename rewrite: `LLVMEnzyme-<N>.so` → `libEnzyme-<N>.so` (with `lib` prefix).

### 4. Register the toolchain as `enzyme`

The ferx build system pins to a toolchain named `enzyme`, so register nightly under that alias:

```bash
rustup toolchain link enzyme "$(rustc +nightly --print sysroot)"
rustc +enzyme --version
```

### 5. Verify

```bash
rustc +enzyme -Z autodiff=Enable - </dev/null 2>&1 | head
```

Expected output: `error[E0601]: `main` function not found`. That's the success signal — rustc + Enzyme loaded correctly. If you see `autodiff backend not found in the sysroot`, the `.so` is missing or in the wrong place.

---

## Shared install (multi-user server)

For a server where multiple users need ferx. A sysadmin builds once into `/opt/rust-nightly`; each user links it into their own rustup.

### Sysadmin steps (one-time)

Same as quick install steps 1–3, but stage the built toolchain in `/opt`:

```bash
sudo mkdir -p /opt/rust-nightly
sudo cp -a ~/.rustup/toolchains/nightly-x86_64-unknown-linux-gnu/. /opt/rust-nightly/
sudo chown -R root:root /opt/rust-nightly
sudo chmod -R a+rX /opt/rust-nightly
sudo chmod a+rx /opt/rust-nightly/bin/*
```

Then drop the Enzyme `.so` into the shared tree:

```bash
TARGET=x86_64-unknown-linux-gnu
sudo cp /tmp/enzyme-build/enzyme/build/Enzyme/LLVMEnzyme-<MAJOR>.so \
   /opt/rust-nightly/lib/rustlib/$TARGET/lib/libEnzyme-<MAJOR>.so
sudo chmod a+r /opt/rust-nightly/lib/rustlib/$TARGET/lib/libEnzyme-<MAJOR>.so
```

### Per-user steps

Each user who wants to use ferx runs these **once** in their account:

```bash
# Each user installs their own rustup, no default toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
source "$HOME/.cargo/env"

# Link the shared toolchain as `enzyme`
rustup toolchain link enzyme /opt/rust-nightly
rustc +enzyme --version
```

For R users, add to `~/.Renviron`:
```
PATH=/opt/rust-nightly/bin:${HOME}/.cargo/bin:${PATH}
RUSTUP_TOOLCHAIN=enzyme
```

Restart R afterwards.

---

## Building ferx-nlme from source

Once the Enzyme toolchain is set up:

```bash
git clone https://github.com/InsightRX/ferx-nlme
cd ferx-nlme

# Release build (recommended for performance)
cargo build --release --features autodiff

# Binary is at target/release/ferx
```

### Build options

```bash
# Debug build (faster compile, slower runtime)
cargo build --features autodiff

# Quick type-check without building
cargo check --features autodiff

# Lints
cargo clippy --features autodiff

# CI build without autodiff (uses finite differences — no Enzyme needed)
cargo build --release --no-default-features --features ci
```

The `ci` feature is useful if you want to develop/test without the full Enzyme toolchain setup — at the cost of much slower gradient computation.

### Verify the build

```bash
cargo run --release --features autodiff --bin ferx -- examples/warfarin.ferx --simulate
```

You should see a successful model fit with parameter estimates.

---

## Installing the ferx R package

If you want the R interface instead of the command-line binary:

Ensure you completed steps 1–4 of the [Quick install](#quick-install-single-user-dev-machine) (or the equivalent shared setup). Then:

```r
devtools::install_github("InsightRX/ferx")
```

The R package drives the Rust build via its `Makevars`. R uses `rustc` from your `PATH` and resolves the `enzyme` toolchain via rustup, so both must be set up correctly in your shell/Renviron before calling `install_github`.

See the [ferx R package README](https://github.com/InsightRX/ferx) for API usage.

---

## Dependencies

FeRx depends on these crates (managed automatically by Cargo):

| Crate | Purpose |
|-------|---------|
| `nalgebra` | Linear algebra (matrices, Cholesky) |
| `nlopt` | Nonlinear optimization (SLSQP, L-BFGS, MMA) |
| `rayon` | Parallel computation |
| `rand`, `rand_distr` | Random number generation (SAEM, SIR) |
| `csv` | CSV data file reading |
| `regex` | Model file expression parsing |

The `nlopt` crate requires the NLopt C library. Most platforms handle this automatically; if build fails on NLopt:
```bash
# macOS
brew install nlopt

# Ubuntu/Debian
sudo apt-get install libnlopt-dev

# Fedora
sudo dnf install NLopt-devel
```

---

## Troubleshooting

### `"error: the option `Z` is only accepted on the nightly compiler"`
Your R or shell is finding a non-nightly rustc. Check `rustc --version`, then verify `PATH` and `RUSTUP_TOOLCHAIN` in `~/.Renviron` (for R) or your shell rc (for CLI).

### `"autodiff backend not found in the sysroot: failed to find a libEnzyme-<N> folder"`
Despite the wording ("folder"), rustc is looking for a file. Causes:
- **Wrong directory**: the `.so` is in `<sysroot>/lib/` instead of `<sysroot>/lib/rustlib/<target>/lib/`
- **LLVM version mismatch**: rebuild Enzyme against the LLVM version rustc reports (`rustc --version --verbose | grep LLVM`)
- **Filename**: must be `libEnzyme-<MAJOR>.so`, not `LLVMEnzyme-<MAJOR>.so` (note the `lib` prefix)

### `"custom toolchain 'enzyme' specified in override file ... is not installed"`
You haven't registered the toolchain. Run `rustup toolchain link enzyme <path>`. See [Quick install step 4](#4-register-the-toolchain-as-enzyme).

### `"not a directory: '/<path>/lib'"` from `rustup toolchain link`
Permission issue — the user running `toolchain link` can't read the target path. For shared installs, run `sudo chmod -R a+rX /opt/rust-nightly`.

### `"incorrect value ... for unstable option autodiff"`
Valid autodiff values change between nightly builds. Currently accepted: `Enable` (most common). Verify with:
```bash
rustc +enzyme -Z autodiff=Enable - </dev/null 2>&1 | head
```
If `Enable` is rejected, try other values: `LooseTypes`, `PrintTA`, `Inline`. Whichever produces only the "missing main" error is valid.

### `"Enzyme: cannot handle (forward) unknown intrinsic llvm.maximumnum"`
Recent rustc lowers `f64::max()` / `f64::min()` to intrinsics Enzyme doesn't differentiate yet. The workaround lives in ferx-nlme source — AD-instrumented functions must use manual `if` comparisons instead. If you hit this, report upstream.

### `"cargo is unavailable for the active toolchain"` (info, not error)
Cargo wasn't copied into your linked toolchain. Either add it (`cp ~/.cargo/bin/cargo /opt/rust-nightly/bin/`) or ignore — rustup falls back to nightly's cargo, which works.

### Need a refresh after upstream nightly rolls forward
If a new ferx-nlme release references stdlib items your cached toolchain doesn't have (e.g. `autodiff_forward` not found), rebuild `/opt/rust-nightly` against the current nightly and rebuild Enzyme if LLVM major version changed.
