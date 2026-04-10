# CLI Reference

The `ferx` command-line tool runs population PK estimation from model files and data.

## Usage

```bash
ferx <model.ferx> --data <data.csv>
ferx <model.ferx> --simulate
```

## Commands

### Fit with Data

```bash
ferx model.ferx --data data.csv
```

Parses the model file, reads the data, and runs the estimation method specified in `[fit_options]` (defaults to FOCE).

### Simulate and Fit

```bash
ferx model.ferx --simulate
```

Parses the model file, generates simulated data from the `[simulation]` block, and fits the model to the simulated data. Requires a `[simulation]` block in the model file.

## Output Files

Three files are generated, named after the model file:

| File | Contents |
|------|----------|
| `{model}-sdtab.csv` | Per-observation diagnostics |
| `{model}-fit.yaml` | Parameter estimates and standard errors |
| `{model}-timing.txt` | Wall-clock estimation time |

See [Output Files](output.md) for detailed format descriptions.

## Console Output

### Progress

The estimation progress is printed to stderr, including:
- Model and data summary (subjects, observations, parameters)
- Optimizer iterations with OFV values (FOCE) or condNLL values (SAEM)
- Covariance step status
- Final parameter table

### Result Summary

A brief summary is printed to stdout:
```
Fit completed!
OFV: -280.1838
Elapsed: 0.496s
  TVCL = 0.132735
  TVV = 7.694842
  TVKA = 0.757498
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (parse failure, data error, convergence failure) |

## Examples

```bash
# One-compartment oral warfarin model
ferx examples/warfarin.ferx --data data/warfarin.csv

# Two-compartment IV with FOCE
ferx examples/two_cpt_iv.ferx --data data/two_cpt_iv.csv

# SAEM estimation
ferx examples/warfarin_saem.ferx --data data/warfarin.csv

# Simulation-estimation study
ferx examples/warfarin.ferx --simulate
```

## Building

```bash
# Build the ferx binary
cargo build --release --features autodiff

# Run directly via cargo
cargo run --release --features autodiff --bin ferx -- model.ferx --data data.csv
```
