# Individual Parameters

The `[individual_parameters]` block defines how population parameters (theta), random effects (eta), and covariates combine to produce individual PK parameters.

## Syntax

```
PARAM = expression
```

Each line assigns a PK parameter using an arithmetic expression that can reference:

- **Theta parameters** -- names defined in `[parameters]` (e.g., `TVCL`, `TVV`)
- **Eta random effects** -- names defined as omega parameters (e.g., `ETA_CL`, `ETA_V`)
- **Covariates** -- column names from the data file (e.g., `WT`, `CRCL`)
- **Constants** -- numeric literals (e.g., `70`, `0.75`)

## Supported Operators and Functions

| Operator/Function | Example |
|-------------------|---------|
| `+`, `-`, `*`, `/` | `TVCL * WT / 70` |
| `^` (power) | `(WT/70)^0.75` |
| `exp()` | `exp(ETA_CL)` |
| `log()`, `ln()` | `log(TVCL)` |
| `sqrt()` | `sqrt(WT)` |
| `abs()` | `abs(ETA_CL)` |
| Parentheses | `TVCL * (WT/70)^0.75` |

## Common Parameterizations

### Exponential (log-normal) random effects

The standard approach for PK parameters that must be positive:

```
[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)
```

### Allometric scaling with covariates

```
[individual_parameters]
  CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)
  V  = TVV  * (WT/70)^1.0  * exp(ETA_V)
```

### Estimated covariate effects

Use additional theta parameters for covariate coefficients:

```
[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta THETA_WT(0.75, 0.01, 2.0)
  theta THETA_CRCL(0.5, 0.01, 2.0)

[individual_parameters]
  CL = TVCL * (WT/70)^THETA_WT * (CRCL/100)^THETA_CRCL * exp(ETA_CL)
```

## Covariate Detection

Any uppercase identifier in the expression that does not match a theta name or eta name is automatically treated as a covariate. The covariate value is read from the corresponding column in the data file.

For example, in `CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)`:
- `TVCL` matches a theta parameter
- `ETA_CL` matches an omega parameter
- `WT` matches neither, so it is treated as a covariate column

## PK Parameter Names

The parameter names on the left side of each assignment must map to recognized PK parameter names:

| Name | PK Parameter |
|------|-------------|
| `CL` | Clearance |
| `V` or `V1` | Volume of distribution (central compartment) |
| `Q` | Intercompartmental clearance |
| `V2` | Peripheral volume |
| `KA` | Absorption rate constant |
| `F` | Bioavailability (default 1.0 if omitted) |

For ODE models, the parameter names are user-defined and passed as a flat vector to the ODE right-hand side function.
