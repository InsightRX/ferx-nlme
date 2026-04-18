# Data Format

FeRx reads data in NONMEM-compatible CSV format. This is the standard format used across population PK tools.

## Required Columns

| Column | Type | Description |
|--------|------|-------------|
| `ID` | string/numeric | Subject identifier |
| `TIME` | numeric | Time relative to first dose |
| `DV` | numeric | Dependent variable (observed concentration) |

## Optional Standard Columns

| Column | Type | Default | Description |
|--------|------|---------|-------------|
| `EVID` | integer | 0 | Event ID: 0 = observation, 1 = dose, 4 = reset + dose |
| `AMT` | numeric | 0 | Dose amount (only for `EVID=1` or `EVID=4`) |
| `CMT` | integer | 1 | Compartment number (1-indexed) |
| `RATE` | numeric | 0 | Infusion rate. 0 = bolus dose |
| `MDV` | integer | 0 | Missing DV flag. 1 = DV should be ignored |
| `II` | numeric | 0 | Interdose interval for repeated dosing |
| `SS` | integer | 0 | Steady-state flag. 1 = assume steady state |
| `CENS` | integer | 0 | Censoring flag. 1 = observation is below LLOQ; `DV` carries the LLOQ value. Paired with `bloq_method = m3` in `[fit_options]` to enable likelihood-based handling — see [BLOQ example](examples/bloq.md). |

## Covariate Columns

Any column not in the standard set above is automatically treated as a covariate. Covariate values are:

- **Time-constant**: The first non-missing value for each subject is used
- **Time-varying**: If values change over time for a subject, Last Observation Carried Forward (LOCF) is applied

Covariate names in the data file are matched case-insensitively to names used in `[individual_parameters]` expressions.

## Event Types (EVID)

| EVID | Meaning |
|------|---------|
| 0 | Observation record. `DV` is used for estimation. |
| 1 | Dosing record. `AMT` is administered to compartment `CMT`. |
| 4 | Reset and dose. All compartment amounts are reset to zero before dosing. |

## Example Dataset

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV,WT,CRCL
1,0,.,1,100,1,0,1,70,95
1,0.5,9.49,0,.,.,.,0,70,95
1,1,14.42,0,.,.,.,0,70,95
1,2,17.56,0,.,.,.,0,70,95
1,4,15.23,0,.,.,.,0,70,95
1,8,10.15,0,.,.,.,0,70,95
2,0,.,1,150,1,0,1,85,110
2,0.5,14.2,0,.,.,.,0,85,110
2,1,21.3,0,.,.,.,0,85,110
```

Key points:
- Dose records (`EVID=1`) have `MDV=1` and `DV=.` (missing)
- Observation records (`EVID=0`) have `MDV=0` and a valid `DV`
- Covariates (`WT`, `CRCL`) are included as extra columns
- Missing values can be represented as `.` or left empty

## Infusion Doses

For IV infusions, set `RATE` to the infusion rate (amount per time unit):

```csv
ID,TIME,DV,EVID,AMT,CMT,RATE,MDV
1,0,.,1,500,1,50,1
```

This administers 500 units at a rate of 50 units/hour (duration = 10 hours).

## Steady-State Dosing

For steady-state simulations, set `SS=1` and `II` to the dosing interval:

```csv
ID,TIME,DV,EVID,AMT,CMT,SS,II,MDV
1,0,.,1,100,1,1,12,1
1,0.5,25.3,0,.,.,.,.,0
```

This assumes the subject has reached steady state with 100 units every 12 hours before the observation at TIME=0.5.

## Multiple Doses

Multiple doses are supported as separate rows:

```csv
ID,TIME,DV,EVID,AMT,CMT,MDV
1,0,.,1,100,1,1
1,0.5,9.49,0,.,.,0
1,12,.,1,100,1,1
1,12.5,15.2,0,.,.,0
1,24,.,1,100,1,1
1,24.5,18.1,0,.,.,0
```

## Column Name Case

Column names are case-insensitive. `ID`, `Id`, and `id` are all recognized. Covariate columns preserve their case as declared in the CSV header.
