# Introduction

FeRx-NLME is a high-performance Nonlinear Mixed Effects (NLME) modeling engine written in Rust. It is designed for population pharmacokinetic (PopPK) analysis, implementing the same statistical methodology as established tools like NONMEM, Monolix, and Pumas.

## Key Features

- **FOCE/FOCEI estimation** -- First-Order Conditional Estimation with optional interaction, the gold standard for PopPK
- **SAEM estimation** -- Stochastic Approximation EM for robust convergence on complex models
- **Analytical PK solutions** -- Built-in one- and two-compartment models (IV bolus, oral, infusion) with numerical stability guarantees
- **ODE solver** -- Dormand-Prince RK45 adaptive integrator for custom kinetic models (e.g. Michaelis-Menten)
- **Automatic differentiation** -- Enzyme-based reverse-mode AD for fast, exact gradients
- **Simple model DSL** -- Declarative `.ferx` model files that read like equations
- **NONMEM-compatible data** -- Reads standard NONMEM CSV datasets directly
- **Covariate support** -- Time-constant and time-varying covariates with automatic detection
- **Parallel estimation** -- Per-subject computations parallelized via Rayon

## How It Compares

| Feature | FeRx | NONMEM | Monolix | Pumas |
|---------|------|--------|---------|-------|
| Language | Rust | Fortran | C++ | Julia |
| FOCE/FOCEI | Yes | Yes | No | Yes |
| SAEM | Yes | No | Yes | Yes |
| Analytical PK | Yes | Yes | Yes | Yes |
| ODE models | Yes | Yes | Yes | Yes |
| Auto-diff | Yes (Enzyme) | No | No | Yes (ForwardDiff) |
| Open source | Yes | No | No | No |

## Architecture Overview

FeRx uses a two-level optimization structure:

- **Outer loop**: Optimizes population parameters (theta, omega, sigma) using NLopt or built-in BFGS
- **Inner loop**: For each subject, finds empirical Bayes estimates (EBEs) of random effects by minimizing individual negative log-likelihood

Parameters are internally transformed for unconstrained optimization: theta and sigma are log-transformed, and omega uses Cholesky factorization to guarantee positive-definiteness.

## Project Structure

```
src/
  types.rs          -- Core data structures
  api.rs            -- Public API (fit, simulate, predict)
  parser/           -- .ferx model file parser
  pk/               -- Analytical PK solutions
  ode/              -- ODE solver and predictions
  estimation/       -- FOCE, FOCEI, SAEM algorithms
  stats/            -- Likelihood and residual computations
  io/               -- Data reading and output writing
  ad/               -- Automatic differentiation
  bin/run_model.rs  -- CLI binary
```

## License

FeRx-NLME is released under the [MIT License](https://github.com/InsightRX/ferx-nlme/blob/main/LICENSE). Copyright © 2026 InsightRX, Inc.
