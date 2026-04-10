# Initial Values

The optional `[initial_values]` block overrides the default starting values for the optimizer. If omitted, the values from the `[parameters]` block are used.

## Syntax

```
[initial_values]
  theta = [val1, val2, ...]
  omega = [var1, var2, ...]
  sigma = [val1, ...]
```

Values are listed in the same order as parameters are declared in `[parameters]`.

## Example

Given parameters:
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

Override initial values:
```
[initial_values]
  theta = [0.2, 10.0, 1.5]
  omega = [0.09, 0.04, 0.30]
  sigma = [0.02]
```

This sets TVCL=0.2, TVV=10.0, TVKA=1.5, etc., regardless of the values in `[parameters]`.

## When to Use

- **Starting from known estimates**: If you have prior estimates from a previous run or the literature, provide them as initial values for faster convergence
- **Troubleshooting convergence**: Poor initial values can lead to convergence failure. Try starting closer to expected parameter values
- **Sensitivity analysis**: Run the model with different starting points to check for local minima

## Notes

- Theta initial values must be within the bounds specified in `[parameters]`
- Omega values are the diagonal variances (not standard deviations or Cholesky factors)
- The number of values must match the number of parameters declared
