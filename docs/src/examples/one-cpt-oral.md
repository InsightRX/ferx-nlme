# One-Compartment Oral (Warfarin)

This example fits a one-compartment oral pharmacokinetic model to warfarin concentration data.

## Model File (`warfarin.ferx`)

```
# One-compartment oral PK model (warfarin)

[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  omega ETA_CL ~ 0.07
  omega ETA_V  ~ 0.02
  omega ETA_KA ~ 0.40

  sigma PROP_ERR ~ 0.01

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[initial_values]
  theta = [0.2, 10.0, 1.5]
  omega = [0.09, 0.04, 0.30]
  sigma = [0.02]

[fit_options]
  method     = foce
  maxiter    = 300
  covariance = true
```

## Model Description

- **Structure**: One-compartment model with first-order absorption (KA) and first-order elimination (CL/V)
- **Random effects**: Log-normal on CL, V, and KA (exponential eta model)
- **Error model**: Proportional residual error
- **Parameters**:
  - TVCL: Typical value of clearance (L/h)
  - TVV: Typical value of volume of distribution (L)
  - TVKA: Typical value of absorption rate constant (1/h)

## Running

```bash
ferx examples/warfarin.ferx --data data/warfarin.csv
```

## Expected Results

```
--- THETA Estimates ---
TVCL                 0.132735
TVV                  7.694842
TVKA                 0.757498

--- OMEGA Estimates (variances) ---
OMEGA(1,1) = 0.028584  (CV% = 16.9)
OMEGA(2,2) = 0.009613  (CV% = 9.8)
OMEGA(3,3) = 0.340868  (CV% = 58.4)

--- SIGMA Estimates ---
SIGMA(1) = 0.010638
```

## SAEM Variant

The same model can be run with SAEM by changing the `[fit_options]`:

```
[fit_options]
  method        = saem
  n_exploration = 150
  n_convergence = 250
  n_mh_steps    = 3
  seed          = 12345
  covariance    = true
```

```bash
ferx examples/warfarin_saem.ferx --data data/warfarin.csv
```
