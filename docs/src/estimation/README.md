# Estimation Methods

FeRx implements two families of estimation methods for nonlinear mixed effects models:

- **[FOCE / FOCEI](foce.md)** -- First-Order Conditional Estimation (with or without interaction). The workhorse of population PK, using nested optimization to find maximum likelihood estimates.

- **[SAEM](saem.md)** -- Stochastic Approximation Expectation-Maximization. Uses MCMC sampling for random effects, providing more robust convergence on complex models.

## Quick Comparison

| Feature | FOCE/FOCEI | SAEM |
|---------|-----------|------|
| Random effect estimation | MAP (optimization) | MCMC (sampling) |
| Convergence speed | Fast (~50-100 iterations) | Slower (~400 iterations) |
| Local minima robustness | Can get stuck | More robust |
| Gradient required | Yes (AD or FD) | No (for E-step) |
| Stochastic | No | Yes |
| Best for | Simple models, quick iteration | Complex models, many random effects |

## Choosing a Method

**Start with FOCE** for standard 1- or 2-compartment models with 2-4 random effects. It is fast and deterministic.

**Switch to SAEM** when:
- FOCE fails to converge or produces implausible estimates
- The model has many random effects (>4)
- You suspect the FOCE solution is a local minimum
- The model has complex nonlinear random effect structure

Both methods produce comparable results on well-behaved models. The final OFV from SAEM is computed using the FOCE approximation, so AIC/BIC values are directly comparable.
