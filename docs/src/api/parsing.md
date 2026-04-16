# Parsing

## Model Parsing

### `parse_model_file()`

Parse a `.ferx` file into a `CompiledModel`. Only the core model blocks are processed (`[parameters]`, `[individual_parameters]`, `[structural_model]`, `[error_model]`).

```rust
pub fn parse_model_file(path: &Path) -> Result<CompiledModel, String>
```

### `parse_model_string()`

Parse a model from a string instead of a file.

```rust
pub fn parse_model_string(content: &str) -> Result<CompiledModel, String>
```

**Example:**
```rust
let model_str = r#"
[parameters]
  theta TVCL(0.1, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  omega ETA_CL ~ 0.09
  sigma ADD_ERR ~ 1.0

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV

[structural_model]
  pk one_cpt_iv_bolus(cl=CL, v=V)

[error_model]
  DV ~ additive(ADD_ERR)
"#;

let model = parse_model_string(model_str)?;
```

### `parse_full_model_file()`

Parse a complete model file including `[fit_options]` and `[simulation]` blocks.

```rust
pub fn parse_full_model_file(path: &Path) -> Result<ParsedModel, String>
```

Returns a `ParsedModel` which contains:

```rust
pub struct ParsedModel {
    pub model: CompiledModel,
    pub simulation: Option<SimulationSpec>,
    pub fit_options: FitOptions,
}
```

## Data Parsing

### `read_nonmem_csv()`

Read a NONMEM-format CSV file into a `Population`.

```rust
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
) -> Result<Population, String>
```

**Parameters:**
- `path`: Path to the CSV file
- `covariate_columns`: Optional list of covariate column names. If `None`, all non-standard columns are auto-detected as covariates.

**Example:**
```rust
// Auto-detect covariates
let pop = read_nonmem_csv(Path::new("data.csv"), None)?;

// Explicit covariate list
let pop = read_nonmem_csv(
    Path::new("data.csv"),
    Some(&["WT", "CRCL", "AGE"]),
)?;

println!("{} subjects, {} observations",
         pop.subjects.len(), pop.n_obs());
```

### Data Processing Details

- Column names are matched case-insensitively
- Standard NONMEM columns (`ID`, `TIME`, `DV`, `EVID`, `AMT`, `CMT`, `RATE`, `MDV`, `II`, `SS`) are recognized automatically
- Missing values (`.`, empty string) are handled appropriately
- Rows with `EVID=1` are treated as dose events
- Rows with `EVID=0` and `MDV=0` are treated as observations
- Time-constant covariates use the first non-missing value per subject
- Time-varying covariates use Last Observation Carried Forward (LOCF)
