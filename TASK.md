# TASK.md — ferx-nlme optimizer additions

This file provides context for Claude Code. Read this alongside the existing CLAUDE.md.

## What we are adding

Two new optimizers to the outer optimization loop (population parameters):

1. **BOBYQA** — derivative-free, quadratic interpolation. Already in the NLopt
   dependency, just not wired up. Robust on poorly-scaled NLME surfaces.
2. **Newton Trust Region** — second-order method using exact AD gradients +
   finite-difference Hessian as a first pass. Requires adding the `argmin` crate.

---

## Task 1 — BOBYQA (ferx-nlme)

### What to change

**File:** `src/estimation/outer_optimizer.rs`

Find the match block that dispatches on `options.optimizer`. It currently handles
`"lbfgs"`, `"mma"`, and defaults to SLSQP. Add `"bobyqa"`:

```rust
let algorithm = match options.optimizer.as_deref() {
    Some("lbfgs")  => nlopt::Algorithm::Lbfgs,
    Some("mma")    => nlopt::Algorithm::Mma,
    Some("bobyqa") => nlopt::Algorithm::Bobyqa,  // ADD THIS
    _              => nlopt::Algorithm::Slsqp,
};
```

**Critical:** BOBYQA is derivative-free. The NLopt objective callback must NOT
receive a gradient function when using it. Add a guard:

```rust
let uses_gradient = !matches!(algorithm, nlopt::Algorithm::Bobyqa);

if uses_gradient {
    opt.set_min_objective(objective_with_grad)?;
} else {
    opt.set_min_objective(objective_no_grad)?;
}
```

The `objective_no_grad` closure is the same as the existing one but ignores the
gradient output slot (pass zeros or leave it untouched — check the nlopt crate's
callback signature).

### Cargo.toml — no change needed

`nlopt = "0.8"` is already there. BOBYQA is included in NLopt.

### Verify

```bash
cargo check
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv
# Then test explicitly with BOBYQA:
# Add optimizer = bobyqa to [fit_options] in examples/warfarin.ferx temporarily
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv
```

OFV should be similar to SLSQP. BOBYQA may take more iterations but should converge.

---

## Task 2 — Newton Trust Region (ferx-nlme)

### Cargo.toml additions

```toml
argmin     = { version = "0.10", features = ["serde1"] }
argmin-math = { version = "0.4", features = ["nalgebra_latest"] }
```

Use `nalgebra_latest` because the codebase already uses `nalgebra` for Cholesky
factorization of omega — the types will mesh without conversion.

### New file: `src/estimation/trust_region.rs`

```rust
use argmin::core::{CostFunction, Gradient, Hessian, Executor, Error};
use argmin::solver::trustregion::{TrustRegion, Steihaug};
use nalgebra::{DVector, DMatrix};
use crate::types::{CompiledModel, Population, FitOptions};

// Wire in whatever the actual NLL and gradient function names are —
// check stats/likelihood.rs and ad/ad_gradients.rs for the real signatures.

pub struct FoceiProblem<'a> {
    pub model:      &'a CompiledModel,
    pub population: &'a Population,
    pub options:    &'a FitOptions,
}

impl CostFunction for FoceiProblem<'_> {
    type Param  = DVector<f64>;
    type Output = f64;

    fn cost(&self, p: &DVector<f64>) -> Result<f64, Error> {
        // Replace focei_nll with the actual function name from stats/likelihood.rs
        Ok(focei_nll(p.as_slice(), self.model, self.population, self.options))
    }
}

impl Gradient for FoceiProblem<'_> {
    type Param    = DVector<f64>;
    type Gradient = DVector<f64>;

    fn gradient(&self, p: &DVector<f64>) -> Result<DVector<f64>, Error> {
        // Replace focei_gradient with the actual function from ad/ad_gradients.rs
        let g = focei_gradient(p.as_slice(), self.model, self.population, self.options);
        Ok(DVector::from_vec(g))
    }
}

impl Hessian for FoceiProblem<'_> {
    type Param   = DVector<f64>;
    type Hessian = DMatrix<f64>;

    fn hessian(&self, p: &DVector<f64>) -> Result<DMatrix<f64>, Error> {
        // First pass: finite differences of the gradient.
        // This is still far better than NONMEM (which also finite-diffs the
        // gradient, not just the Hessian).
        // Replace with exact Enzyme Hessian later.
        let n = p.len();
        let eps = 1e-5_f64;
        let g0 = focei_gradient(p.as_slice(), self.model, self.population, self.options);
        let mut h = DMatrix::zeros(n, n);
        for i in 0..n {
            let mut p_plus = p.clone();
            p_plus[i] += eps;
            let g1 = focei_gradient(
                p_plus.as_slice(), self.model, self.population, self.options
            );
            for j in 0..n {
                h[(i, j)] = (g1[j] - g0[j]) / eps;
            }
        }
        // Symmetrize to guard against floating-point asymmetry
        Ok((&h + h.transpose()) / 2.0)
    }
}

pub fn run_trust_region(
    init:       &[f64],
    model:      &CompiledModel,
    population: &Population,
    options:    &FitOptions,
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let problem     = FoceiProblem { model, population, options };
    let init_param  = DVector::from_column_slice(init);

    // Steihaug conjugate-gradient inner solver — handles indefinite Hessians
    // gracefully, which matters early in optimization when curvature is poor.
    let subproblem  = Steihaug::new().with_max_iters(20);
    let solver      = TrustRegion::new(subproblem)
        .with_radius(1.0)
        .with_max_radius(10.0);

    let result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(init_param)
                .max_iters(options.maxiter as u64)
        })
        .run()?;

    Ok(result.state.best_param.unwrap().as_slice().to_vec())
}
```

### Wire into `src/estimation/outer_optimizer.rs`

Add the trust region branch before the NLopt block:

```rust
pub fn run_outer_optimizer(
    init:       &[f64],
    model:      &CompiledModel,
    population: &Population,
    options:    &FitOptions,
) -> Result<Vec<f64>, FerxError> {

    match options.optimizer.as_deref() {
        Some("trust_region") | Some("newton_tr") => {
            crate::estimation::trust_region::run_trust_region(
                init, model, population, options
            )
            .map_err(|e| FerxError::OptimizationFailed(e.to_string()))
        },
        _ => run_nlopt(init, model, population, options),
    }
}
```

Add `mod trust_region;` to `src/estimation/mod.rs` (or wherever the module
declarations live).

### Verify

```bash
cargo check
# Then test:
# Add optimizer = trust_region to [fit_options] in examples/warfarin.ferx
cargo run --release -- examples/warfarin.ferx --data data/warfarin.csv
```

---

## Task 3 — Expose in the R package (ferx)

Do this after both Rust tasks compile and produce sensible OFVs.

### `src/rust/src/lib.rs` (the extendr glue, ~250 lines)

Find the `#[extendr]` function that calls `fit()`. Add `optimizer: &str` as a
parameter if it is not already there, and pass it into `FitOptions`:

```rust
#[extendr]
fn r_fit(model_path: &str, data_path: &str, method: &str, optimizer: &str) -> List {
    let options = FitOptions {
        optimizer: Some(optimizer.to_string()),
        // ... other fields unchanged
    };
    // ...
}
```

After changing the Rust signature, regenerate the R wrapper:

```r
rextendr::document()
```

This regenerates `R/extendr-wrappers.R` — do not edit that file by hand.

### `R/ferx_fit.R`

Add `optimizer` as a formal argument:

```r
ferx_fit <- function(
    model,
    data,
    method    = "focei",
    optimizer = "slsqp",
    ...
) {
    valid_optimizers <- c("slsqp", "lbfgs", "mma", "bobyqa", "trust_region")
    if (!optimizer %in% valid_optimizers) {
        stop(sprintf(
            "`optimizer` must be one of: %s",
            paste(valid_optimizers, collapse = ", ")
        ))
    }
    result <- .Call(wrap__r_fit, model, data, method, optimizer)
    structure(result, class = "ferx_fit")
}
```

Rebuild:

```bash
R CMD INSTALL .
```

### Test from R

```r
library(ferx)
ex <- ferx_example("warfarin")

# Should work as before
fit_slsqp <- ferx_fit(ex$model, ex$data, optimizer = "slsqp")

# New: derivative-free, robust
fit_bobyqa <- ferx_fit(ex$model, ex$data, optimizer = "bobyqa")

# New: second-order
fit_tr <- ferx_fit(ex$model, ex$data, optimizer = "trust_region")

# OFVs should be similar across all three
fit_slsqp$ofv
fit_bobyqa$ofv
fit_tr$ofv
```

---

## Order of attack

1. BOBYQA in ferx-nlme — compile and test
2. Trust region in ferx-nlme — compile and test
3. R layer in ferx — expose optimizer argument, rebuild, test

Do each step before moving to the next. If cargo check fails on trust_region,
check the actual function names in stats/likelihood.rs and ad/ad_gradients.rs —
the names used in this file are illustrative and may differ slightly.

## Notes for Claude Code

- Do NOT touch the autodiff feature flag or any `#[autodiff_forward]` /
  `#[autodiff_reverse]` annotated functions. The new optimizer code does not
  need Enzyme — it calls the existing gradient functions as black boxes.
- Warnings should go into FitResult.warnings (Vec<String>), not stderr.
- The inner optimizer (ETA estimation per subject) is separate and unchanged.
- Run `cargo clippy` after `cargo check` to catch style issues before committing.
