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
| `EBE_OFV` | Each subject's contribution to the total OFV |
| `N_OBS` | Number of observations for the subject |

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

## FitResult Fields (Rust API / console)

The following fields are populated on the `FitResult` struct returned by `fit()` and printed by `print_results()`. They are **not** currently written to the fit YAML — read them programmatically or from the console summary.

### Shrinkage

Two shrinkage metrics are reported after every fit:

**ETA shrinkage** (per random effect — `shrinkage_eta: Vec<f64>`):
\\[ \text{shrinkage}_k = 1 - \frac{\text{SD}(\hat{\eta}_k)}{\sqrt{\omega_{kk}}} \\]

A value near 1 means individual EBEs are all pulled toward zero — the data are not informative about that random effect. A value near 0 means the ETAs are spread consistent with the prior omega.

**EPS shrinkage** (scalar — `shrinkage_eps: f64`):
\\[ \text{shrinkage}_\varepsilon = 1 - \text{SD}(\text{IWRES}) \\]

Values of `NaN` indicate a zero-variance omega component (ETA) or fewer than two valid residuals (EPS).

### Covariance Status

`covariance_status: CovarianceStatus` takes one of three values:

| Value | Meaning |
|-------|---------|
| `Computed` | Covariance step succeeded; SE values are valid |
| `Failed` | Hessian was singular or inversion failed; SE fields are `None` |
| `NotRequested` | `covariance = false` was set; SE fields are `None` |

### Run Record Fields

| Field | Description |
|-------|-------------|
| `model_name` | Name from the `.ferx` file (or `"Unnamed"`) |
| `ferx_version` | Version of ferx-nlme that produced the result |
| `wall_time_secs` | Wall-clock time for the complete fit (seconds) |
| `gradient_method_inner` | Gradient method used in the inner (EBE) loop, e.g. `FiniteDifference` |
| `gradient_method_outer` | Gradient method used in the outer loop, e.g. `FiniteDifference` or `AutoDiff` |
| `uses_ode_solver` | `true` if the model uses the ODE solver, `false` for analytical PK |
| `n_threads_used` | Number of Rayon threads used during estimation |
| `nlopt_missing_algorithms` | NLopt algorithms that were requested but unavailable in this build (empty when all available) |
| `covariance_n_evals_estimated` | Estimated number of OFV evaluations the covariance step will run, populated only when `run_covariance_step = true` and `n_parameters > 30` |

### EBE Convergence Diagnostics

Counters from the inner-loop (EBE) optimizer, useful for diagnosing problematic fits. Always `0` for SAEM (which uses MH sampling rather than EBE optimization).

| Field | Description |
|-------|-------------|
| `ebe_convergence_warnings` | Number of outer iterations in which at least one subject had an unconverged EBE |
| `max_unconverged_subjects` | Worst-case unconverged-subject count seen in a single outer iteration |
| `total_ebe_fallbacks` | Total number of times the Nelder-Mead fallback was invoked across all subjects and outer iterations |

## Timing File (`{model}-timing.txt`)

A single-line text file with the wall-clock estimation time:

```
elapsed_seconds=0.496000
```

This measures only the estimation step (not parsing or data reading).

## Optimizer Trace CSV

When `optimizer_trace = true` is set in `[fit_options]`, a CSV is written to `/tmp/ferx_trace_<pid>_<ts>.csv`. The path is also stored in `FitResult::trace_path`.

Each row is one outer iteration. See the [fit-options trace table](model-file/fit-options.md#optimizer-trace) for the full column reference.

Example use in R (with the `ferx` package):

```r
fit <- ferx_fit("model.ferx", "data.csv", optimizer_trace = TRUE)
trace <- read.csv(fit$trace_path)
plot(trace$iter, trace$ofv, type = "l", xlab = "Iteration", ylab = "OFV")
```
