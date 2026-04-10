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

The omega matrix is currently estimated as a diagonal matrix (no covariances between random effects). Each variance represents the between-subject variability for that parameter.

The coefficient of variation (CV%) is approximately `sqrt(variance) * 100` for log-normally distributed parameters. For example, `omega ETA_CL ~ 0.09` corresponds to ~30% CV.

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

## Complete Example

```
[parameters]
  # Fixed effects
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  # Between-subject variability
  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  # Residual error
  sigma PROP_ERR ~ 0.01
```
