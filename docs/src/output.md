# Output Files

Each model run produces three output files.

## sdtab CSV (`{model}-sdtab.csv`)

A CSV file with per-observation diagnostics, one row per observation per subject.

### Columns

| Column | Description |
|--------|-------------|
| `ID` | Subject identifier |
| `TIME` | Observation time |
| `DV` | Observed value |
| `PRED` | Population prediction (eta = 0) |
| `IPRED` | Individual prediction (eta = EBE) |
| `CWRES` | Conditional weighted residual |
| `IWRES` | Individual weighted residual |
| `ETA1`, `ETA2`, ... | Empirical Bayes estimates of random effects |

### Residual Definitions

**IWRES** (Individual Weighted Residual):
\\[ \text{IWRES}_j = \frac{y_j - \text{IPRED}_j}{\sqrt{V_j}} \\]
where \\( V_j \\) is the residual variance evaluated at the individual prediction.

**CWRES** (Conditional Weighted Residual):
\\[ \text{CWRES}_j = \frac{y_j - f_{0,j}}{\sqrt{\tilde{R}_{jj}}} \\]
where \\( f_0 = f(\hat{\eta}) - H\hat{\eta} \\) is the linearized population prediction and \\( \tilde{R} = H\Omega H^T + R \\) is the conditional variance.

### Example

```csv
ID,TIME,DV,PRED,IPRED,CWRES,IWRES,ETA1,ETA2,ETA3
1,0.5,9.49,10.12,9.55,-0.23,-0.06,0.15,-0.08,0.32
1,1.0,14.42,14.87,14.35,0.18,0.05,0.15,-0.08,0.32
```

## Fit YAML (`{model}-fit.yaml`)

A YAML file containing parameter estimates, standard errors, and model diagnostics.

### Structure

```yaml
model:
  converged: true
  method: FOCE
objective_function:
  ofv: -280.1838
  aic: -266.1838
  bic: -247.2804
data:
  n_subjects: 10
  n_observations: 110
  n_parameters: 7
theta:
  TVCL:
    estimate: 0.132735
    se: 0.014549
    rse_pct: 11.0
  TVV:
    estimate: 7.694842
    se: 0.293028
    rse_pct: 3.8
omega:
  omega_11:
    variance: 0.028584
    cv_pct: 16.9
    se: 0.006394
  omega_22:
    variance: 0.009613
    cv_pct: 9.8
    se: 0.002165
sigma:
  sigma_1:
    estimate: 0.010638
    se: 0.000788
```

### Key Fields

- **ofv**: Objective Function Value (-2 log-likelihood)
- **aic**: Akaike Information Criterion (OFV + 2p)
- **bic**: Bayesian Information Criterion (OFV + p * ln(n))
- **se**: Standard error from the covariance step
- **rse_pct**: Relative standard error as percentage (SE/estimate * 100)
- **cv_pct**: Coefficient of variation for omega (sqrt(variance) * 100)

## Timing File (`{model}-timing.txt`)

A single-line text file with the wall-clock estimation time:

```
elapsed_seconds=0.496000
```

This measures only the estimation step (not parsing or data reading).
