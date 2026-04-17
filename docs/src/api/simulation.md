# Simulation

## `simulate()`

Generate simulated observations from a model with random effects and residual error.

```rust
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult>
```

**Parameters:**
- `model`: Compiled model
- `population`: Template population (dose events and observation times are used; DV values are ignored)
- `params`: True parameter values for simulation
- `n_sim`: Number of simulation replicates

**Returns:** Vector of `SimulationResult`, one per observation per subject per replicate.

**Example:**
```rust
let model = parse_model_file(Path::new("model.ferx"))?;
let population = read_nonmem_csv(Path::new("data.csv"), None)?;

// Simulate 1000 replicates
let sims = simulate(&model, &population, &model.default_params, 1000);

for sim in &sims[..5] {
    println!("Sim {}, ID {}, TIME {}, IPRED {:.3}, DV {:.3}",
             sim.sim, sim.id, sim.time, sim.ipred, sim.dv_sim);
}
```

## `simulate_with_seed()`

Same as `simulate()` but with a fixed random seed for reproducibility.

```rust
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult>
```

## `simulate_from_fit()` / `simulate_from_fit_with_seed()`

Convenience wrappers that simulate at the estimates contained in a
[`FitResult`]. Metadata that `FitResult` doesn't carry (parameter bounds,
sigma names, omega's diagonal flag) is taken from `model.default_params`,
so callers only need `(model, population, fit)`.

```rust
pub fn simulate_from_fit(
    model: &CompiledModel,
    population: &Population,
    fit: &FitResult,
    n_sim: usize,
) -> Vec<SimulationResult>

pub fn simulate_from_fit_with_seed(
    model: &CompiledModel,
    population: &Population,
    fit: &FitResult,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult>
```

**Example** — fit, then generate VPC samples at the fitted estimates:
```rust
let model = parse_model_file(Path::new("model.ferx"))?;
let population = read_nonmem_csv(Path::new("data.csv"), None)?;

let fit = fit(&model, &population, &model.default_params, &opts)?;
let vpc = simulate_from_fit_with_seed(&model, &population, &fit, 1000, 42);
```

Under the hood these call [`ModelParameters::from_fit_result`] and then delegate
to `simulate` / `simulate_with_seed`.

## `predict()`

Population predictions without random effects (eta = 0). No simulation noise is added.

```rust
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult>
```

**Returns:** Vector of `PredictionResult` with population-level predictions.

**Example:**
```rust
let preds = predict(&model, &population, &model.default_params);
for p in &preds {
    println!("ID {}, TIME {}, PRED {:.3}", p.id, p.time, p.pred);
}
```

## `predict_from_fit()`

Convenience wrapper around [`predict`] that uses the theta contained in a
`FitResult`.

```rust
pub fn predict_from_fit(
    model: &CompiledModel,
    population: &Population,
    fit: &FitResult,
) -> Vec<PredictionResult>
```

## `ModelParameters::from_fit_result()`

Low-level helper used by the `_from_fit` variants. Given a `FitResult` and a
template `ModelParameters` (typically `model.default_params`), reconstructs a
fully-typed `ModelParameters` suitable for direct use with `simulate` /
`predict` / `fit`.

```rust
impl ModelParameters {
    pub fn from_fit_result(fit: &FitResult, template: &ModelParameters) -> Self;
}
```

## Result Types

```rust
pub struct SimulationResult {
    pub sim: usize,     // Replicate number (1-indexed)
    pub id: String,     // Subject ID
    pub time: f64,      // Observation time
    pub ipred: f64,     // Individual prediction (no residual error)
    pub dv_sim: f64,    // Simulated observation (with residual error)
}

pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,      // Population prediction (eta = 0)
}
```

## Simulation Process

For each replicate and each subject:

1. Sample random effects: \\( \eta_i \sim N(0, \Omega) \\) using the Cholesky factor \\( L \\): \\( \eta = L \cdot z \\), where \\( z \sim N(0, I) \\)
2. Compute individual PK parameters via `pk_param_fn(theta, eta, covariates)`
3. Generate predictions using the structural model
4. Add residual error: \\( DV = IPRED + \sqrt{V} \cdot \epsilon \\), where \\( \epsilon \sim N(0, 1) \\) and \\( V \\) is the residual variance from the error model
