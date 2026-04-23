# SAEM

Stochastic Approximation Expectation-Maximization (SAEM) is an alternative estimation method that uses MCMC sampling for random effects instead of MAP optimization. It is more robust to local minima and can handle complex random effect structures.

## Algorithm Overview

SAEM replaces the deterministic inner loop of FOCE with stochastic sampling, following the Monolix convention with a two-phase step-size schedule.

### References

- Delyon, Lavielle, Moulines (1999). *Convergence of a stochastic approximation version of the EM algorithm.* Annals of Statistics, 94--128.
- Kuhn & Lavielle (2004). *Coupling a stochastic approximation version of EM with an MCMC procedure.* ESAIM: Probability and Statistics 8:115--131.

## Two-Phase Schedule

### Phase 1: Exploration (iterations 1 to K1)

Step size \\( \gamma_k = 1 \\). The algorithm explores the parameter space rapidly, with the sufficient statistics being fully replaced each iteration. This allows fast movement toward the basin of the MLE.

Default: 150 iterations.

### Phase 2: Convergence (iterations K1+1 to K1+K2)

Step size \\( \gamma_k = 1/(k - K_1) \\). The algorithm performs a decreasing-weight average, which guarantees almost-sure convergence to the MLE under regularity conditions.

Default: 250 iterations.

## Per-Iteration Steps

Each SAEM iteration consists of:

### 1. E-Step: Metropolis-Hastings Sampling

For each subject, run `n_mh_steps` Metropolis-Hastings iterations to sample from the conditional distribution of random effects:

\\[ p(\eta_i | y_i, \theta, \Omega, \sigma) \\]

**Proposal**:

- *Exploration phase* (with `mu_referencing = true`): \\( \eta_{\text{prop}} = \mu_k + \delta_i \cdot L \cdot z \\), an independence proposal centred on the current population mean \\( \mu_k \\) — auto-detected from `[individual_parameters]`. This helps the chain escape the \\( \eta = 0 \\) basin when `THETA` is far from the truth.
- *Convergence phase*, and whenever `mu_referencing = false`: \\( \eta_{\text{prop}} = \eta_{\text{current}} + \delta_i \cdot L \cdot z \\), a symmetric random walk that preserves detailed balance during stochastic approximation.

In both cases \\( z \sim N(0, I) \\) and \\( L = \text{chol}(\Omega) \\).

**Acceptance**: Accept with probability \\( \min(1, \exp(\text{NLL}_{\text{current}} - \text{NLL}_{\text{prop}})) \\).

The MH sampling is parallelized across subjects using Rayon.

### 2. Stochastic Approximation Update

Update the sufficient statistic for \\( \Omega \\):

\\[ S_2 \leftarrow (1 - \gamma_k) \cdot S_2 + \gamma_k \cdot \frac{1}{N} \sum_{i=1}^{N} \eta_i \eta_i^T \\]

### 3. M-Step for Omega (Closed Form)

\\[ \Omega_k = S_2 \\]

### 4. M-Step for Theta and Sigma (Optimization)

Minimize the conditional observation negative log-likelihood with ETAs held fixed:

\\[ \sum_{i=1}^{N} \sum_{j=1}^{n_i} \left[ \frac{1}{2} \log V_{ij} + \frac{1}{2} \frac{(y_{ij} - f_{ij})^2}{V_{ij}} \right] \\]

This is optimized over \\( \theta \\) and \\( \sigma \\) in log-space using NLopt SLSQP with finite-difference gradients.

### 5. Adaptive MH Step Sizes

Every `adapt_interval` iterations, the per-subject step sizes \\( \delta_i \\) are adjusted:
- If acceptance rate > 40%: increase \\( \delta_i \\) by 10% (up to 5.0)
- If acceptance rate < 40%: decrease \\( \delta_i \\) by 10% (down to 0.01)

This targets an acceptance rate around 40%, balancing exploration and mixing.

## Post-SAEM Finalization

After the SAEM iterations complete:

1. **EBE Refinement**: Run the standard FOCE inner loop (BFGS optimization) warm-started from the SAEM ETAs to obtain final empirical Bayes estimates
2. **FOCE OFV**: Compute the objective function using the FOCE/Laplace approximation, so AIC and BIC are directly comparable with FOCE results
3. **Covariance Step**: Optionally compute standard errors via finite-difference Hessian (same method as FOCE)
4. **Diagnostics**: Compute PRED, IPRED, CWRES, IWRES for each subject

## Configuration

```
[fit_options]
  method        = saem
  n_exploration = 150      # Phase 1 iterations
  n_convergence = 250      # Phase 2 iterations
  n_mh_steps    = 3        # MH steps per subject per iteration
  adapt_interval = 50      # Step-size adaptation frequency
  seed          = 12345    # RNG seed for reproducibility
  covariance    = true     # Compute standard errors
```

## Tuning Guide

### Not Converging

- Increase `n_exploration` (e.g., 300) to give more time for basin finding
- Increase `n_convergence` (e.g., 500) for a longer averaging window
- Increase `n_mh_steps` (e.g., 5-10) for better mixing in the E-step

### Slow Convergence

- Decrease `n_exploration` and `n_convergence` if parameters stabilize early
- Use `adapt_interval = 25` for faster step-size adaptation

### Reproducibility

- Always set `seed` for reproducible results
- Different seeds will produce slightly different estimates due to the stochastic nature of the algorithm

## Output

The SAEM iteration progress is printed to stderr:

```
SAEM: 10 subjects, 3 ETAs, 400 total iter (150 explore + 250 converge)
  SAEM iter    1/400 [explore] gamma=1.000  condNLL=95.244
  SAEM iter   50/400 [explore] gamma=1.000  condNLL=56.705
  SAEM iter  150/400 [explore] gamma=1.000  condNLL=46.071
  SAEM iter  200/400 [converge] gamma=0.020  condNLL=36.799
  SAEM iter  400/400 [converge] gamma=0.004  condNLL=38.096
SAEM iterations complete. Computing final EBEs and OFV...
```

The `condNLL` is the conditional observation negative log-likelihood (not the final OFV). It should generally decrease during the exploration phase and stabilize during convergence.
