# Sampling Importance Resampling (SIR)

SIR is an optional post-estimation step that provides non-parametric parameter uncertainty estimates. It produces 95% confidence intervals that are more robust than the asymptotic covariance matrix, particularly for models with:

- Non-normal parameter distributions
- Boundary estimates (parameters near constraints)
- Small datasets where asymptotic assumptions may not hold

## How It Works

SIR uses the maximum likelihood estimates and their covariance matrix as a proposal distribution, then reweights samples based on the actual likelihood:

1. **Sample**: Draw M parameter vectors from a multivariate normal distribution centered on the ML estimates, using the estimation covariance matrix
2. **Importance weighting**: For each sample, compute the objective function value (OFV) and calculate an importance weight based on the ratio of the true likelihood to the proposal density
3. **Resample**: Draw m vectors (with replacement) proportional to the importance weights

The resampled vectors approximate the true parameter uncertainty distribution. Confidence intervals are derived from their empirical percentiles.

## Enabling SIR

Add `sir = true` to the `[fit_options]` block. The covariance step must also be enabled (it provides the proposal distribution):

```
[fit_options]
  method     = focei
  covariance = true
  sir        = true
```

## Options

| Key | Default | Description |
|-----|---------|-------------|
| `sir` | `false` | Enable/disable SIR |
| `sir_samples` | `1000` | Number of proposal samples (M). Higher values give more reliable weights but take longer |
| `sir_resamples` | `250` | Number of resampled vectors (m). Must be less than `sir_samples` |
| `sir_seed` | `12345` | RNG seed for reproducibility |

## Output

SIR adds the following to the estimation output:

- **95% CI** for each theta, omega, and sigma parameter (2.5th and 97.5th percentiles)
- **Effective sample size (ESS)**: a diagnostic indicating how well the proposal distribution matches the true uncertainty. ESS close to M indicates a good match; ESS much less than m suggests the proposal is a poor fit

## Diagnostics

The effective sample size (ESS) is the primary diagnostic:

- **ESS > m** (resamples): excellent — the proposal distribution is well-matched
- **ESS between 100 and m**: adequate for most purposes
- **ESS < 100**: the proposal may be a poor fit; consider a different estimation method or increasing `sir_samples`

## Computational Cost

SIR evaluates the inner loop (EBE optimization) for each of the M proposal samples. With the default M=1000, this is roughly 3-10x the cost of the estimation step itself. The computation is parallelized across samples and warm-started from the ML EBEs to minimize runtime.

The resampling step itself is negligible.

## Example

```
[fit_options]
  method        = focei
  covariance    = true
  sir           = true
  sir_samples   = 1000
  sir_resamples = 250
  sir_seed      = 42
```

## Reference

Dosne A-G, Bergstrand M, Karlsson MO. "Improving the estimation of parameter uncertainty distributions in nonlinear mixed effects models using sampling importance resampling." *J Pharmacokinet Pharmacodyn*. 2017;44(6):539-562. doi:10.1007/s10928-017-9542-0
