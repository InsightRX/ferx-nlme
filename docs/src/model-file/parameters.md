# Parameters

The `[parameters]` block defines all model parameters: fixed effects (theta), between-subject variability (omega), and residual error (sigma).

## Theta (Fixed Effects)

```
theta NAME(initial_value, lower_bound, upper_bound)
```

- **NAME**: Parameter name (used in `[individual_parameters]` expressions)
- **initial_value**: Starting value for estimation
- **lower_bound**: Lower bound constraint (must be > 0; parameters are log-transformed internally)
- **upper_bound**: Upper bound constraint

Example:
```
theta TVCL(0.134, 0.001, 10.0)
theta TVV(8.1, 0.1, 500.0)
theta TVKA(1.0, 0.01, 50.0)
```

## Omega (Between-Subject Variability)

### Diagonal omega

```
omega NAME ~ variance
```

- **NAME**: Random effect name (used in `[individual_parameters]` as `ETA_XXX`)
- **variance**: Initial variance estimate (diagonal element of the omega matrix)

Example:
```
omega ETA_CL ~ 0.07
omega ETA_V  ~ 0.02
omega ETA_KA ~ 0.40
```

Each variance represents the between-subject variability for that parameter. The coefficient of variation (CV%) is approximately `sqrt(variance) * 100` for log-normally distributed parameters. For example, `omega ETA_CL ~ 0.09` corresponds to ~30% CV.

### Block omega (correlated random effects)

To estimate correlations between random effects, use `block_omega`:

```
block_omega (NAME1, NAME2, ...) = [lower_triangle_values]
```

The values are the lower triangle of the covariance matrix, specified row-wise. For a 2x2 block:

```
block_omega (ETA_CL, ETA_V) = [var_CL, cov_CL_V, var_V]
```

For a 3x3 block:

```
block_omega (ETA_CL, ETA_V, ETA_KA) = [var_CL, cov_CL_V, var_V, cov_CL_KA, cov_V_KA, var_KA]
```

You can mix diagonal and block omega specifications. Diagonal omegas specify uncorrelated random effects, while block omegas estimate the full covariance sub-matrix:

```
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
omega ETA_KA ~ 0.40
```

This estimates a 3x3 omega where ETA_CL and ETA_V are correlated (2x2 block), but ETA_KA is uncorrelated with both.

### Declaration order

The order of `omega` and `block_omega` lines in the `[parameters]` block determines the ETA indexing throughout the model: in the omega matrix and all output. For example:

```
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
omega ETA_KA ~ 0.40
```

produces ETA order `[ETA_CL, ETA_V, ETA_KA]` (indices 1, 2, 3), while:

```
omega ETA_KA ~ 0.40
block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
```

produces `[ETA_KA, ETA_CL, ETA_V]` (indices 1, 2, 3). The `[individual_parameters]` block should list assignments in the same order for clarity, though the parameter mapping is by name, not position.

## Kappa (Inter-Occasion Variability)

Inter-Occasion Variability (IOV) is declared with `kappa` (independent per-parameter) or `block_kappa` (correlated across parameters). Kappa parameters must be paired with `iov_column` in `[fit_options]` and an occasion column in the dataset.

### Diagonal kappa — Option A

```
kappa NAME ~ variance
kappa NAME ~ variance FIX
```

Each `kappa` line adds one diagonal element to the IOV omega matrix. Occasions are independent.

Example:
```
kappa KAPPA_CL ~ 0.05
kappa KAPPA_V  ~ 0.03
kappa KAPPA_CL ~ 0.02 FIX
```

### Block kappa — Option B (correlated IOV)

```
block_kappa (NAME1, NAME2, ...) = [lower_triangle_values]
block_kappa (NAME1, NAME2, ...) = [lower_triangle_values] FIX
```

Mirrors the `block_omega` syntax — values are the lower triangle of the IOV covariance matrix, row-wise. Use this when IOV effects are expected to be correlated (e.g. correlated occasion shifts in CL and V).

Example (2×2 block):
```
block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
```

where `0.05` = Var(KAPPA_CL), `0.01` = Cov(KAPPA_CL, KAPPA_V), `0.03` = Var(KAPPA_V).

### Using kappas in individual parameters

Reference kappa names exactly like BSV etas in `[individual_parameters]`:

```
[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V  + KAPPA_V)
  KA = TVKA * exp(ETA_KA)           # no IOV on absorption
```

Kappas can be combined freely — a parameter can carry BSV only, IOV only, or both.

### Mixed diagonal and block kappa

You can mix `kappa` (uncorrelated) and `block_kappa` (correlated) declarations in the same model:

```
block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
kappa KAPPA_KA ~ 0.10
```

A name may not appear in both `kappa` and `block_kappa` — this is a parse error.

### Declaration order

Kappa declaration order determines the IOV omega matrix layout and the kappa column order in output. Kappas follow all BSV etas in the internal indexing: if a model has 3 BSV etas and 2 kappas, the kappas sit at indices 4 and 5.

## Sigma (Residual Error)

```
sigma NAME ~ value
```

- **NAME**: Residual error parameter name (referenced in `[error_model]`)
- **value**: Initial value

Example:
```
sigma PROP_ERR ~ 0.01
sigma ADD_ERR  ~ 1.0
```

The interpretation of sigma depends on the error model:

| Error Model | Sigma Meaning |
|-------------|---------------|
| Additive | Standard deviation of additive error |
| Proportional | Coefficient of proportional error |
| Combined | First sigma = proportional coefficient, second = additive SD |

## Complete Examples

Diagonal omega (no correlations):
```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01
```

Block omega (correlated CL and V):
```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01
```
