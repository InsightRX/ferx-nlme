# Inter-Occasion Variability (IOV)

Inter-Occasion Variability (IOV) models the fact that a subject's PK parameters may shift between occasions (treatment periods, study visits, dosing intervals). Unlike between-subject variability (BSV), which is fixed for a subject across the whole study, IOV is a random effect that is re-drawn for each occasion.

## Concepts

### Occasions

An **occasion** is a distinct time window during which the same kappa value applies. Occasions are identified by an integer column in the dataset (e.g. `OCC`). All rows — dose records and observation records — that share the same occasion index belong to one occasion.

### Kappa parameters

**Kappa** (κ) is the IOV random effect, analogous to eta (η) for BSV. Each kappa is drawn independently per occasion:

\\[ \kappa_{ik} \sim \mathcal{N}(0, \Omega_\text{IOV}) \\]

where *i* indexes subjects and *k* indexes occasions. The IOV omega matrix Ω_IOV is estimated alongside the BSV omega.

### Individual parameters

Kappas enter the individual-parameter expressions in the same way as etas:

```
CL = TVCL * exp(ETA_CL + KAPPA_CL)
```

At each occasion *k*, the effective individual CL is `TVCL * exp(ETA_CL_i + KAPPA_CL_ik)`. The BSV eta captures stable between-subject differences; the kappa captures occasion-to-occasion fluctuation within a subject.

## Option A: Diagonal IOV (independent kappas)

Each kappa is declared independently, giving a **diagonal** Ω_IOV:

```
[parameters]
  ...
  kappa KAPPA_CL ~ 0.05
  kappa KAPPA_V  ~ 0.03
```

The two kappas are uncorrelated across occasions. This is the most common formulation and corresponds to NONMEM's IOV Option A.

Use `FIX` to hold a kappa variance constant during estimation:
```
kappa KAPPA_CL ~ 0.05 FIX
```

## Option B: Block IOV (correlated kappas)

When occasion effects on different parameters are expected to covary, use `block_kappa`:

```
[parameters]
  ...
  block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
```

Values are the lower triangle of Ω_IOV: Var(KAPPA_CL), Cov(KAPPA_CL, KAPPA_V), Var(KAPPA_V). This mirrors the `block_omega` syntax for BSV.

### Mixed Option A + B

`kappa` and `block_kappa` declarations can be combined freely — uncorrelated kappas can coexist with a correlated block:

```
block_kappa (KAPPA_CL, KAPPA_V) = [0.05, 0.01, 0.03]
kappa KAPPA_KA ~ 0.10
```

A name may not appear in both a `kappa` and a `block_kappa` line.

## Model File Setup

```
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09          # BSV
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30

  kappa KAPPA_CL ~ 0.05        # IOV (Option A)

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL + KAPPA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
  method     = foce
  iov_column = OCC
```

See `examples/warfarin_iov.ferx` for a complete runnable example.

## Algorithm Details

### Inner loop

When a model has kappa declarations and the subject has occasion labels, the inner optimizer runs `find_ebe_iov` instead of the standard `find_ebe`. It jointly optimizes over:

\\[ p = [\underbrace{\eta_1, \ldots, \eta_{n_\eta}}_{\text{BSV}},\ \underbrace{\kappa_{1,1}, \ldots, \kappa_{1,n_\kappa}}_{\text{occasion 1}},\ \ldots,\ \underbrace{\kappa_{K,1}, \ldots, \kappa_{K,n_\kappa}}_{\text{occasion K}}] \\]

The joint negative log-posterior is:

\\[
-\log p(p \mid y_i) = \frac{1}{2}\left[
  \eta^T \Omega^{-1} \eta + \log|\Omega|
  + \sum_{k=1}^{K} \kappa_k^T \Omega_\text{IOV}^{-1} \kappa_k
  + K \log|\Omega_\text{IOV}|
  + \sum_{j} \left(\frac{(y_{ij} - f_{ij}^{(k_j)})^2}{V_{ij}} + \log V_{ij}\right)
\right]
\\]

where \\( k_j \\) is the occasion of observation *j* and \\( f_{ij}^{(k)} = f(\theta, \eta_i, \kappa_{ik}) \\).

Optimization uses BFGS; the gradient is computed by finite differences (no AD path for IOV).

### Outer loop

The IOV omega diagonal (Option A) or Cholesky elements (Option B) are packed after the sigma block in the optimizer vector:

\\[
x = [\log\theta,\ \text{chol}(\Omega_\text{BSV}),\ \log\sigma,\ \log\text{diag}(\text{chol}(\Omega_\text{IOV}))]
\\]

The per-subject FOCE objective uses a block-diagonal omega that stacks the BSV block with *K* copies of Ω_IOV:

\\[
\Omega_\text{full} = \text{blockdiag}(\Omega_\text{BSV},\ \Omega_\text{IOV},\ \ldots,\ \Omega_\text{IOV})
\\]

The H-matrix (Jacobian ∂f/∂η) covers BSV etas only — kappa columns are not included in the FOCE linearization.

### Covariance step

When `covariance = true`, standard errors for the IOV omega diagonal (or its Cholesky elements) are computed via the finite-difference Hessian at convergence, using the same procedure as for BSV omega.

## Output

| Field | Description |
|-------|-------------|
| `omega_iov` | Estimated IOV covariance matrix |
| `kappa_names` | Names of kappa parameters in declaration order |
| `se_kappa` | Standard errors for IOV omega elements (requires `covariance = true`) |
| `shrinkage_kappa` | EBE shrinkage per kappa (%) |
| `ebe_kappas` | Per-subject, per-occasion kappa EBEs. `ebe_kappas[i][k]` is the kappa vector for subject *i*, occasion *k* |

Kappa EBEs appear in the sdtab output as `KAPPA_xxx` columns.

## Limitations

- **SAEM is not supported** with IOV models — an error is returned directing you to use `foce` or `focei`.
- **Automatic differentiation (AD) is not used** for IOV inner-loop gradients; finite differences are used instead. For large models this is slower than the AD path used for BSV-only models.
- The occasion column must contain positive integers. Non-integer or negative values are not supported.

## Comparison to NONMEM

| Concept | NONMEM | FeRx |
|---------|--------|------|
| Diagonal IOV | `OMEGA BLOCK(n) SAME` with one kappa per block | `kappa NAME ~ variance` (Option A) |
| Correlated IOV | `OMEGA BLOCK(n)` shared across occasions | `block_kappa (N1, N2) = [...]` (Option B) |
| Occasion column | `OCC` in `$INPUT`; referenced in `$PK` | `iov_column = OCC` in `[fit_options]` |
| Kappa in PK | `ETA(n)` with `SAME` block | `KAPPA_xxx` in `[individual_parameters]` |
