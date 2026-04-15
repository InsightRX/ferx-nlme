# Estimation Methods

FeRx implements two families of estimation methods for nonlinear mixed effects models:

- **[FOCE / FOCEI](foce.md)** -- First-Order Conditional Estimation (with or without interaction). The workhorse of population PK, using nested optimization to find maximum likelihood estimates.

- **[Gauss-Newton (BHHH)](gauss-newton.md)** -- A fast alternative that exploits the nonlinear-least-squares structure of the FOCE objective. Converges in 10-30 iterations using the outer product of per-subject gradients as an approximate Hessian. Available as pure GN or a GN+FOCEI hybrid.

- **[SAEM](saem.md)** -- Stochastic Approximation Expectation-Maximization. Uses MCMC sampling for random effects, providing more robust convergence on complex models.

- **[SIR](sir.md)** -- Sampling Importance Resampling. An optional post-estimation step that provides non-parametric parameter uncertainty estimates (95% CIs), more robust than the asymptotic covariance matrix.

## Quick Comparison

| Feature | FOCE/FOCEI | Gauss-Newton | SAEM |
|---------|-----------|-------------|------|
| Random effect estimation | MAP (optimization) | MAP (optimization) | MCMC (sampling) |
| Convergence speed | Medium (~50-100 evals) | Fast (~10-30 iterations) | Slower (~400 iterations) |
| Local minima robustness | Can get stuck | Can get stuck | More robust |
| Gradient required | Yes (AD or FD) | Yes (FD, per-subject) | No (for E-step) |
| Stochastic | No | No | Yes |
| Best for | General use | Fast iteration, well-conditioned models | Complex models, many random effects |

## Choosing a Method

**Start with FOCE** for standard 1- or 2-compartment models with 2-4 random effects. It is deterministic and well-understood.

**Try Gauss-Newton (`gn` or `gn_hybrid`)** when you want faster convergence during model development, or when FOCE is slow to converge.

**Switch to SAEM** when:
- FOCE fails to converge or produces implausible estimates
- The model has many random effects (>4)
- You suspect the FOCE solution is a local minimum
- The model has complex nonlinear random effect structure

All methods produce comparable results on well-behaved models. The final OFV from SAEM and GN is computed using the FOCE approximation, so AIC/BIC values are directly comparable.
