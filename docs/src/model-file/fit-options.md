# Fit Options

The optional `[fit_options]` block configures the estimation method and optimizer settings.

## Syntax

```
[fit_options]
  key = value
```

## General Options

| Key | Values | Default | Description |
|-----|--------|---------|-------------|
| `method` | `foce`, `focei`, `saem` | `foce` | Estimation method |
| `maxiter` | integer | `500` | Maximum outer loop iterations |
| `covariance` | `true`, `false` | `true` | Compute covariance matrix and standard errors |
| `optimizer` | `slsqp`, `lbfgs`, `nlopt_lbfgs`, `mma`, `bfgs`, `bobyqa`, `trust_region` | `slsqp` | Optimization algorithm |
| `inner_maxiter` | integer | `200` | Max iterations for the inner (per-subject EBE) optimizer |
| `inner_tol` | float | `1e-8` | Gradient-norm convergence tolerance for the inner optimizer |
| `steihaug_max_iters` | integer | `50` | Max CG iterations for the Steihaug subproblem (only used when `optimizer = trust_region`) |
| `global_search` | `true`, `false` | `false` | Run gradient-free pre-search before local optimization |
| `global_maxeval` | integer | auto | Max evaluations for global search |
| `bloq_method` | `drop`, `m3` | `drop` | How to handle rows with `CENS=1`. `m3` enables Beal's M3 likelihood (see [BLOQ example](../examples/bloq.md)). |
| `mu_referencing` | `true`, `false` | `true` | Re-centre inner-loop ETA estimates on the current population mean (auto-detected from `[individual_parameters]`). See the [FAQ entry](../faq.md#do-i-need-to-use-mu-referencing-in-my-model-definitions-like-in-nonmem--nlmixr2) for details. Set `false` to reproduce pre-automatic-mu behaviour. |

## Estimation Methods

### FOCE (default)
```
method = foce
```
First-Order Conditional Estimation. Linearizes the model around the empirical Bayes estimates. Fast and reliable for most models.

### FOCEI
```
method = focei
```
FOCE with Interaction. Includes the dependence of the residual error on random effects. More accurate than FOCE when the error model depends on individual predictions, but slightly slower.

### SAEM
```
method = saem
```
Stochastic Approximation EM. Uses Metropolis-Hastings sampling instead of MAP optimization for random effects. More robust to local minima; recommended for complex models with many random effects.

## SAEM-Specific Options

| Key | Default | Description |
|-----|---------|-------------|
| `n_exploration` | `150` | Phase 1 iterations (step size = 1) |
| `n_convergence` | `250` | Phase 2 iterations (step size = 1/k) |
| `n_mh_steps` | `3` | Metropolis-Hastings steps per subject per iteration |
| `adapt_interval` | `50` | Iterations between MH step-size adaptation |
| `seed` | `12345` | RNG seed for reproducibility |

## SIR (Sampling Importance Resampling)

SIR provides non-parametric parameter uncertainty estimates as an optional post-estimation step. Requires `covariance = true`.

| Key | Default | Description |
|-----|---------|-------------|
| `sir` | `false` | Enable SIR uncertainty estimation |
| `sir_samples` | `1000` | Number of proposal samples (M) |
| `sir_resamples` | `250` | Number of resampled vectors (m) |
| `sir_seed` | `12345` | RNG seed for reproducibility |

See [SIR documentation](../estimation/sir.md) for details.

## Optimizer Choices

| Optimizer | Description | Recommended For |
|-----------|-------------|-----------------|
| `slsqp` | Sequential Least Squares Programming (NLopt) | General use (default) |
| `bfgs` | Built-in BFGS quasi-Newton | When NLopt is unavailable |
| `lbfgs` | Limited-memory BFGS | Large parameter spaces |
| `nlopt_lbfgs` | NLopt L-BFGS | Alternative L-BFGS |
| `mma` | Method of Moving Asymptotes (NLopt) | Constrained problems |
| `bobyqa` | NLopt BOBYQA — derivative-free quadratic interpolation | Noisy or non-smooth objectives where FD gradients are unreliable |
| `trust_region` | Newton trust-region with Steihaug CG subproblem (argmin) | Well-conditioned problems where second-order curvature helps convergence; tune `steihaug_max_iters` |

Notes:
- `bobyqa` does not use gradients, so it is robust to small discontinuities in
  the FOCE surface caused by EBE re-estimation, but it converges more slowly
  than gradient-based methods on smooth problems.
- `trust_region` uses a finite-difference Hessian of the OFV-at-fixed-EBEs.
  Each Hessian costs O(n²) OFV evaluations, so it is fastest when the number
  of packed parameters is small. Increase `steihaug_max_iters` when the
  parameter count exceeds the default of 50.

## Global Search

Setting `global_search = true` runs a gradient-free pre-search (NLopt CRS2-LM) before the local optimizer. This helps escape local minima on challenging datasets.

The number of global evaluations is auto-scaled based on the number of parameters and observations, or can be set explicitly with `global_maxeval`.

## Examples

Standard FOCE with defaults:
```
[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

FOCEI with global search:
```
[fit_options]
  method        = focei
  maxiter       = 500
  covariance    = true
  global_search = true
```

SAEM with custom settings:
```
[fit_options]
  method        = saem
  n_exploration = 200
  n_convergence = 300
  n_mh_steps    = 5
  seed          = 42
  covariance    = true
```

FOCEI with SIR uncertainty:
```
[fit_options]
  method        = focei
  covariance    = true
  sir           = true
  sir_samples   = 1000
  sir_resamples = 250
  sir_seed      = 42
```

Derivative-free BOBYQA fit:
```
[fit_options]
  method        = foce
  optimizer     = bobyqa
  maxiter       = 300
  inner_maxiter = 100
  inner_tol     = 1e-6
```

Trust-region with tuned CG subproblem:
```
[fit_options]
  method             = foce
  optimizer          = trust_region
  maxiter            = 200
  steihaug_max_iters = 30
```
