use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

/// Parse a model file (.ferx) and return a CompiledModel.
pub fn parse_model_file(path: &Path) -> Result<CompiledModel, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read model file: {}", e))?;
    parse_model_string(&content)
}

/// Parse a full model file including simulation spec, initial values, and fit options.
pub fn parse_full_model_file(path: &Path) -> Result<ParsedModel, String> {
    let content =
        std::fs::read_to_string(path).map_err(|e| format!("Failed to read model file: {}", e))?;
    parse_full_model(&content)
}

/// Parse a model string and return a CompiledModel (backward compatible).
pub fn parse_model_string(content: &str) -> Result<CompiledModel, String> {
    let parsed = parse_full_model(content)?;
    Ok(parsed.model)
}

/// Parse a full model string including all optional blocks.
pub fn parse_full_model(content: &str) -> Result<ParsedModel, String> {
    let blocks = extract_blocks(content)?;
    let name = extract_model_name(content);

    // ── Required blocks ──
    let param_lines = blocks
        .get("parameters")
        .ok_or("Missing [parameters] block")?;
    let (thetas, omegas, block_omegas, sigmas, eta_names) = parse_parameters(param_lines)?;

    let struct_lines = blocks
        .get("structural_model")
        .ok_or("Missing [structural_model] block")?;

    let error_lines = blocks
        .get("error_model")
        .ok_or("Missing [error_model] block")?;
    let (error_model, _) = parse_error_model(error_lines)?;

    let indiv_lines = blocks
        .get("individual_parameters")
        .ok_or("Missing [individual_parameters] block")?;

    let theta_names: Vec<String> = thetas.iter().map(|t| t.name.clone()).collect();
    let sigma_names: Vec<String> = sigmas.iter().map(|s| s.name.clone()).collect();
    let n_theta = theta_names.len();
    let n_eta = eta_names.len();
    let n_epsilon = sigma_names.len();

    // Detect ODE vs analytical model
    let is_ode = struct_lines
        .iter()
        .any(|l| l.starts_with("ode(") || l.starts_with("ode "));

    let (pk_model, pk_param_map, ode_spec) = if is_ode {
        let (state_names, obs_cmt_name) = parse_ode_structural(struct_lines)?;
        let ode_lines = blocks
            .get("odes")
            .ok_or("ODE model requires [odes] block")?;
        // Extract individual parameter names (LHS of assignments)
        let indiv_param_names: Vec<String> = indiv_lines
            .iter()
            .filter_map(|l| l.splitn(2, '=').next().map(|s| s.trim().to_string()))
            .collect();
        let ode_spec = build_ode_spec(
            ode_lines,
            &state_names,
            &obs_cmt_name,
            &theta_names,
            &eta_names,
            &indiv_param_names,
        )?;
        // PK model not used for ODE, but we need a placeholder + empty param map
        (PkModel::OneCptOral, HashMap::new(), Some(ode_spec))
    } else {
        let (pk_model, pk_param_map) = parse_structural_model(struct_lines)?;
        (pk_model, pk_param_map, None)
    };

    let pk_param_fn = build_pk_param_fn(indiv_lines, &theta_names, &eta_names, &pk_param_map)?;

    let theta_values: Vec<f64> = thetas.iter().map(|t| t.init).collect();
    let theta_lower: Vec<f64> = thetas.iter().map(|t| t.lower).collect();
    let theta_upper: Vec<f64> = thetas.iter().map(|t| t.upper).collect();
    let omega = build_omega_matrix(&omegas, &block_omegas, &eta_names)?;
    let sigma_values: Vec<f64> = sigmas.iter().map(|s| s.value).collect();
    let sigma = SigmaVector {
        values: sigma_values,
        names: sigma_names,
    };

    let default_params = ModelParameters {
        theta: theta_values,
        theta_names: theta_names.clone(),
        theta_lower,
        theta_upper,
        omega,
        sigma,
    };

    // Auto-generate tv_fn: evaluate individual parameters with eta=0
    // This gives covariate-adjusted typical values for the AD inner loop.
    let tv_assignments = indiv_lines.clone();
    let tv_theta_names = theta_names.clone();
    let tv_eta_names = eta_names.clone();
    let tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>> =
        if !is_ode {
            let assignments: Vec<(String, Expression)> = tv_assignments
                .iter()
                .filter_map(|line| {
                    let parts: Vec<&str> = line.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        let var_name = parts[0].trim().to_string();
                        let expr_str = parts[1].trim();
                        parse_expression(expr_str, &tv_theta_names, &tv_eta_names)
                            .ok()
                            .map(|expr| (var_name, expr))
                    } else {
                        None
                    }
                })
                .collect();

            Some(Box::new(
                move |theta: &[f64], covariates: &HashMap<String, f64>| {
                    let zero_eta = vec![0.0; tv_eta_names.len()];
                    let mut vars: HashMap<String, f64> = HashMap::new();
                    let mut tv_values = Vec::new();

                    for (var_name, expr) in &assignments {
                        let val = eval_expression(expr, theta, &zero_eta, covariates, &vars);
                        vars.insert(var_name.clone(), val);
                        tv_values.push(val);
                    }

                    tv_values
                },
            ))
        } else {
            None
        };

    // Build pk_indices: maps each individual parameter (by declaration order)
    // to its PK parameter index. Needed for AD to place values in correct slots.
    let pk_indices: Vec<usize> = if !pk_param_map.is_empty() {
        // Reverse the pk_param_map: variable_name → pk_param_name
        let var_to_pk: HashMap<String, String> = pk_param_map
            .iter()
            .map(|(pk_name, var_name)| (var_name.to_uppercase(), pk_name.clone()))
            .collect();
        indiv_lines
            .iter()
            .filter_map(|line| line.splitn(2, '=').next().map(|s| s.trim().to_uppercase()))
            .map(|var_name| {
                var_to_pk
                    .get(&var_name)
                    .and_then(|pk_name| PkParams::name_to_index(pk_name))
                    .unwrap_or(0)
            })
            .collect()
    } else {
        // ODE model: sequential indices
        (0..n_eta).collect()
    };

    let model = CompiledModel {
        name,
        pk_model,
        error_model,
        pk_param_fn,
        n_theta,
        n_eta,
        n_epsilon,
        theta_names,
        eta_names,
        default_params,
        tv_fn,
        pk_indices,
        ode_spec,
        bloq_method: BloqMethod::Drop,
    };

    // ── Optional blocks ──
    let simulation = blocks
        .get("simulation")
        .map(|lines| parse_simulation_block(lines))
        .transpose()?;
    let fit_options = if let Some(lines) = blocks.get("fit_options") {
        parse_fit_options(lines)?
    } else {
        FitOptions::default()
    };

    // Mirror fit-level BLOQ method onto the compiled model so the likelihood
    // functions can branch without threading bloq_method through every call.
    let mut model = model;
    model.bloq_method = fit_options.bloq_method;

    Ok(ParsedModel {
        model,
        simulation,
        fit_options,
    })
}

// ── [simulation] block parser ───────────────────────────────────────────────

fn parse_simulation_block(lines: &[String]) -> Result<SimulationSpec, String> {
    let mut n_subjects = 10;
    let mut dose_amt = 100.0;
    let mut dose_cmt = 1;
    let mut obs_times = Vec::new();
    let mut seed = 42u64;

    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        match parts[0] {
            "subjects" => {
                n_subjects = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad subjects: {}", line))?
            }
            "dose" => {
                dose_amt = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad dose: {}", line))?
            }
            "cmt" => dose_cmt = parts[1].parse().map_err(|_| format!("Bad cmt: {}", line))?,
            "seed" => {
                seed = parts[1]
                    .parse()
                    .map_err(|_| format!("Bad seed: {}", line))?
            }
            "times" => obs_times = parse_float_array(parts[1])?,
            _ => {}
        }
    }
    if obs_times.is_empty() {
        return Err("[simulation] block requires 'times = [...]'".to_string());
    }

    Ok(SimulationSpec {
        n_subjects,
        dose_amt,
        dose_cmt,
        obs_times,
        seed,
        covariates: vec![],
    })
}

// ── [fit_options] block parser ──────────────────────────────────────────────

fn parse_fit_options(lines: &[String]) -> Result<FitOptions, String> {
    let mut opts = FitOptions::default();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        // `method` has looser legacy semantics in the .ferx parser (no
        // strict validation, silent fallback to Foce). Keep it inline here
        // so `apply_fit_option` can stay strict for R/settings callers.
        if parts[0] == "method" {
            let val = parts[1].to_lowercase();
            if val.trim() == "saem" {
                opts.method = EstimationMethod::Saem;
            } else if val.contains("hybrid")
                || val.contains("gn_hybrid")
                || val.contains("gn-hybrid")
            {
                opts.method = EstimationMethod::FoceGnHybrid;
            } else if val.contains("gn") || val.contains("gauss") {
                opts.method = EstimationMethod::FoceGn;
            } else if val.contains("focei")
                || val.contains("foce-i")
                || val.contains("interaction")
            {
                opts.method = EstimationMethod::FoceI;
                opts.interaction = true;
            } else {
                opts.method = EstimationMethod::Foce;
            }
            continue;
        }
        // All other keys flow through the shared dispatch so R's
        // `settings` path and the .ferx parser stay in lock-step on which
        // keys are accepted. Malformed values return an error; unknown
        // keys are ignored here (the .ferx parser is tolerant by
        // convention — the R wrapper enforces strictness itself).
        match apply_fit_option(&mut opts, parts[0], parts[1]) {
            Ok(_) => {}
            Err(e) => return Err(format!("[fit_options]: {}", e)),
        }
    }
    Ok(opts)
}

/// Apply a single `key = value` pair to a [`FitOptions`] struct.
///
/// Returns:
/// - `Ok(true)`  — key was recognized and applied.
/// - `Ok(false)` — key is not a known fit option.
/// - `Err(msg)`  — key is recognized but the value is malformed.
///
/// This is the single source of truth for the `[fit_options]` key grammar,
/// shared between `.ferx` parsing and the R wrapper's generic `settings`
/// list. Callers that want strict validation (e.g. the R wrapper) should
/// propagate `Err` and treat `Ok(false)` as "unknown setting". The `.ferx`
/// parser is intentionally tolerant of unknown keys for forward compat.
///
/// Does NOT handle `method` — that stays in the block parser because
/// `.ferx` files use a looser, case-insensitive substring match there
/// which would surprise R-side callers if enforced uniformly.
pub fn apply_fit_option(
    opts: &mut FitOptions,
    key: &str,
    value: &str,
) -> Result<bool, String> {
    let value = value.trim();

    let parse_usize = |name: &str| -> Result<usize, String> {
        value.parse::<usize>().map_err(|_| {
            format!(
                "fit option `{name}`: expected non-negative integer, got `{value}`"
            )
        })
    };
    let parse_bool = |name: &str| -> Result<bool, String> {
        match value.to_lowercase().as_str() {
            "true" | "t" | "yes" | "1" | "on" => Ok(true),
            "false" | "f" | "no" | "0" | "off" => Ok(false),
            _ => Err(format!(
                "fit option `{name}`: expected true/false, got `{value}`"
            )),
        }
    };
    let parse_u64_opt = |name: &str| -> Result<Option<u64>, String> {
        if value.is_empty()
            || value.eq_ignore_ascii_case("null")
            || value.eq_ignore_ascii_case("na")
        {
            Ok(None)
        } else {
            value.parse::<u64>().map(Some).map_err(|_| {
                format!(
                    "fit option `{name}`: expected non-negative integer, got `{value}`"
                )
            })
        }
    };
    let parse_f64 = |name: &str| -> Result<f64, String> {
        value
            .parse::<f64>()
            .map_err(|_| format!("fit option `{name}`: expected number, got `{value}`"))
    };

    match key {
        "maxiter" => opts.outer_maxiter = parse_usize("maxiter")?,
        "inner_maxiter" => opts.inner_maxiter = parse_usize("inner_maxiter")?,
        "inner_tol" => opts.inner_tol = parse_f64("inner_tol")?,
        "covariance" => opts.run_covariance_step = parse_bool("covariance")?,
        "verbose" => opts.verbose = parse_bool("verbose")?,
        "optimizer" => {
            opts.optimizer = match value.to_lowercase().as_str() {
                "slsqp" => Optimizer::Slsqp,
                "lbfgs" | "nlopt_lbfgs" => Optimizer::NloptLbfgs,
                "mma" => Optimizer::Mma,
                "bfgs" => Optimizer::Bfgs,
                "bobyqa" => Optimizer::Bobyqa,
                "trust_region" | "newton_tr" => Optimizer::TrustRegion,
                other => {
                    return Err(format!(
                        "fit option `optimizer`: unknown value `{other}` — expected \
                         slsqp/lbfgs/nlopt_lbfgs/mma/bfgs/bobyqa/trust_region"
                    ));
                }
            };
        }
        "steihaug_max_iters" => {
            opts.steihaug_max_iters = parse_usize("steihaug_max_iters")?
        }
        "global_search" => opts.global_search = parse_bool("global_search")?,
        "global_maxeval" => opts.global_maxeval = parse_usize("global_maxeval")?,
        "n_exploration" => opts.saem_n_exploration = parse_usize("n_exploration")?,
        "n_convergence" => opts.saem_n_convergence = parse_usize("n_convergence")?,
        "n_mh_steps" => opts.saem_n_mh_steps = parse_usize("n_mh_steps")?,
        "adapt_interval" => opts.saem_adapt_interval = parse_usize("adapt_interval")?,
        "seed" | "saem_seed" => opts.saem_seed = parse_u64_opt("seed")?,
        "gn_lambda" => opts.gn_lambda = parse_f64("gn_lambda")?,
        "sir" => opts.sir = parse_bool("sir")?,
        "sir_samples" => opts.sir_samples = parse_usize("sir_samples")?,
        "sir_resamples" => opts.sir_resamples = parse_usize("sir_resamples")?,
        "sir_seed" => opts.sir_seed = parse_u64_opt("sir_seed")?,
        "bloq_method" | "bloq" => {
            opts.bloq_method = match value.to_lowercase().as_str() {
                "m3" => BloqMethod::M3,
                "drop" | "none" | "ignore" => BloqMethod::Drop,
                other => {
                    return Err(format!(
                        "fit option `bloq_method`: unknown value `{other}` — expected 'm3' or 'drop'"
                    ));
                }
            };
        }
        _ => return Ok(false),
    }
    Ok(true)
}

// ── [structural_model] ODE variant parser ───────────────────────────────────

fn parse_ode_structural(lines: &[String]) -> Result<(Vec<String>, String), String> {
    // ode(obs_cmt=central, states=[depot, central])
    let re =
        Regex::new(r"ode\(\s*obs_cmt\s*=\s*(\w+)\s*,\s*states\s*=\s*\[([^\]]+)\]\s*\)").unwrap();
    for line in lines {
        if let Some(caps) = re.captures(line) {
            let obs_cmt = caps[1].to_string();
            let states: Vec<String> = caps[2].split(',').map(|s| s.trim().to_string()).collect();
            return Ok((states, obs_cmt));
        }
    }
    Err(
        "Could not parse ODE structural model. Expected: ode(obs_cmt=NAME, states=[...])"
            .to_string(),
    )
}

// ── [odes] block → OdeSpec ──────────────────────────────────────────────────

fn build_ode_spec(
    lines: &[String],
    state_names: &[String],
    obs_cmt_name: &str,
    _theta_names: &[String],
    _eta_names: &[String],
    indiv_param_names: &[String],
) -> Result<crate::ode::OdeSpec, String> {
    let n_states = state_names.len();
    let obs_cmt_idx = state_names
        .iter()
        .position(|s| s == obs_cmt_name)
        .ok_or_else(|| {
            format!(
                "Observable compartment '{}' not in states {:?}",
                obs_cmt_name, state_names
            )
        })?;

    // Parse each d/dt(state) = expression
    let ddt_re = Regex::new(r"d/dt\((\w+)\)\s*=\s*(.+)").unwrap();
    let mut ode_exprs: Vec<(String, Expression)> = Vec::new();

    // For ODE expressions, pass empty theta/eta names so all identifiers
    // (states + individual params) are treated as Variables, not Theta/Eta/Covariate
    let empty: Vec<String> = vec![];

    for line in lines {
        if let Some(caps) = ddt_re.captures(line) {
            let state = caps[1].to_string();
            let expr_str = caps[2].trim();
            let expr = parse_expression(expr_str, &empty, &empty)?;
            ode_exprs.push((state, expr));
        }
    }

    if ode_exprs.len() != n_states {
        return Err(format!(
            "Expected {} ODE equations (one per state), found {}",
            n_states,
            ode_exprs.len()
        ));
    }

    // Build RHS closure: (u, params_flat, t, du)
    // u[i] = state values, params_flat = PkParams.values
    // The individual parameter names from [individual_parameters] are stored in params_flat
    // via PkParams indexing. We need to map state names and parameter names to indices.
    let state_names_owned = state_names.to_vec();
    let indiv_names_owned = indiv_param_names.to_vec();
    let ode_exprs_owned = ode_exprs;

    let rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync> =
        Box::new(move |u: &[f64], params: &[f64], _t: f64, du: &mut [f64]| {
            let mut vars: HashMap<String, f64> = HashMap::new();

            // Inject state variables: state_name → u[i]
            for (i, name) in state_names_owned.iter().enumerate() {
                vars.insert(name.clone(), u[i]);
                vars.insert(name.to_lowercase(), u[i]);
            }

            // Inject individual parameters by name → params[i]
            // params = PkParams.values, where pk_param_fn stores individual params
            // by position matching the order in [individual_parameters] block
            for (i, name) in indiv_names_owned.iter().enumerate() {
                if i < params.len() {
                    vars.insert(name.clone(), params[i]);
                    vars.insert(name.to_uppercase(), params[i]);
                    vars.insert(name.to_lowercase(), params[i]);
                }
            }

            let empty_theta: [f64; 0] = [];
            let empty_eta: [f64; 0] = [];
            let empty_cov = HashMap::new();

            for (i, (_, expr)) in ode_exprs_owned.iter().enumerate() {
                du[i] = eval_expression(expr, &empty_theta, &empty_eta, &empty_cov, &vars);
            }
        });

    Ok(crate::ode::OdeSpec {
        rhs,
        n_states,
        state_names: state_names.to_vec(),
        obs_cmt_idx,
    })
}

// ── Helper: parse "[1.0, 2.0, 3.0]" → Vec<f64> ────────────────────────────

fn parse_float_array(s: &str) -> Result<Vec<f64>, String> {
    let s = s.trim().trim_start_matches('[').trim_end_matches(']');
    s.split(',')
        .map(|v| {
            v.trim()
                .parse::<f64>()
                .map_err(|_| format!("Bad float in array: '{}'", v.trim()))
        })
        .collect()
}

// --- Internal types ---

struct ThetaSpec {
    name: String,
    init: f64,
    lower: f64,
    upper: f64,
}

struct OmegaSpec {
    name: String,
    variance: f64,
}

/// Specifies a block (correlated) group of omegas.
/// The values are the lower triangle of the covariance matrix, row-wise:
/// e.g. for 2x2: [var1, cov12, var2]; for 3x3: [var1, cov12, var2, cov13, cov23, var3]
struct BlockOmegaSpec {
    names: Vec<String>,
    lower_triangle: Vec<f64>,
}

struct SigmaSpec {
    name: String,
    value: f64,
}

// --- Block extraction ---

fn extract_model_name(content: &str) -> String {
    let re = Regex::new(r"(?m)^\s*model\s+(\w+)").unwrap();
    re.captures(content)
        .and_then(|c| c.get(1))
        .map(|m| m.as_str().to_string())
        .unwrap_or_else(|| "Unnamed".to_string())
}

fn extract_blocks(content: &str) -> Result<HashMap<String, Vec<String>>, String> {
    let mut blocks: HashMap<String, Vec<String>> = HashMap::new();
    let block_re = Regex::new(r"\[(\w+)\]").unwrap();

    let mut current_block: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') || trimmed.starts_with("//") {
            continue;
        }

        if let Some(caps) = block_re.captures(trimmed) {
            current_block = Some(caps[1].to_lowercase());
            continue;
        }

        if trimmed.starts_with("model ") || trimmed == "end" {
            continue;
        }

        if let Some(ref block) = current_block {
            blocks
                .entry(block.clone())
                .or_default()
                .push(trimmed.to_string());
        }
    }

    Ok(blocks)
}

// --- Parameter parsing ---

fn parse_parameters(
    lines: &[String],
) -> Result<
    (
        Vec<ThetaSpec>,
        Vec<OmegaSpec>,
        Vec<BlockOmegaSpec>,
        Vec<SigmaSpec>,
        Vec<String>, // eta names in declaration order
    ),
    String,
> {
    let mut thetas = Vec::new();
    let mut omegas = Vec::new();
    let mut block_omegas = Vec::new();
    let mut sigmas = Vec::new();
    let mut eta_names_ordered = Vec::new();

    // theta NAME(init, lower, upper)  or  theta NAME(init)
    let theta_re = Regex::new(
        r"theta\s+(\w+)\(\s*([0-9eE.+-]+)\s*(?:,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*)?\)",
    )
    .unwrap();

    // omega NAME ~ value
    let omega_re = Regex::new(r"omega\s+(\w+)\s*~\s*([0-9eE.+-]+)").unwrap();

    // block_omega (NAME1, NAME2, ...) = [lower_triangle_values]
    let block_omega_re = Regex::new(r"block_omega\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\]").unwrap();

    // sigma NAME ~ value
    let sigma_re = Regex::new(r"sigma\s+(\w+)\s*~\s*([0-9eE.+-]+)").unwrap();

    for line in lines {
        if let Some(caps) = theta_re.captures(line) {
            let name = caps[1].to_string();
            let init: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad theta init: {}", line))?;
            let lower: f64 = caps
                .get(3)
                .map(|m| m.as_str().parse().unwrap_or(1e-9))
                .unwrap_or(1e-9);
            let upper: f64 = caps
                .get(4)
                .map(|m| m.as_str().parse().unwrap_or(1e9))
                .unwrap_or(1e9);
            thetas.push(ThetaSpec {
                name,
                init,
                lower,
                upper,
            });
        } else if let Some(caps) = block_omega_re.captures(line) {
            let names: Vec<String> = caps[1].split(',').map(|s| s.trim().to_string()).collect();
            let values: Vec<f64> = caps[2]
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|_| format!("Bad block_omega value in: {}", line))
                })
                .collect::<Result<Vec<_>, _>>()?;
            let n = names.len();
            let expected = n * (n + 1) / 2;
            if values.len() != expected {
                return Err(format!(
                    "block_omega with {} etas expects {} lower-triangle values, got {}: {}",
                    n,
                    expected,
                    values.len(),
                    line
                ));
            }
            for n in &names {
                eta_names_ordered.push(n.clone());
            }
            block_omegas.push(BlockOmegaSpec {
                names,
                lower_triangle: values,
            });
        } else if let Some(caps) = omega_re.captures(line) {
            let name = caps[1].to_string();
            let variance: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad omega: {}", line))?;
            eta_names_ordered.push(name.clone());
            omegas.push(OmegaSpec { name, variance });
        } else if let Some(caps) = sigma_re.captures(line) {
            let name = caps[1].to_string();
            let value: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad sigma: {}", line))?;
            sigmas.push(SigmaSpec { name, value });
        }
    }

    Ok((thetas, omegas, block_omegas, sigmas, eta_names_ordered))
}

// --- Build omega matrix from diagonal + block specs ---

/// Construct a full OmegaMatrix from diagonal omega specs and block omega specs.
/// The `eta_names` vector determines the matrix ordering (declaration order from
/// the model file). If any block omega is present, the matrix is non-diagonal.
fn build_omega_matrix(
    diag_omegas: &[OmegaSpec],
    block_omegas: &[BlockOmegaSpec],
    eta_names: &[String],
) -> Result<OmegaMatrix, String> {
    let n = eta_names.len();
    if n == 0 {
        return Err("No omega parameters defined".to_string());
    }

    // If no block omegas, use the simple diagonal path
    if block_omegas.is_empty() {
        let variances: Vec<f64> = diag_omegas.iter().map(|o| o.variance).collect();
        return Ok(OmegaMatrix::from_diagonal(&variances, eta_names.to_vec()));
    }

    // Build a name→index map
    let name_to_idx: std::collections::HashMap<&str, usize> = eta_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    // Start with a zero matrix, fill diagonal entries from diagonal specs
    let mut matrix = nalgebra::DMatrix::zeros(n, n);
    for spec in diag_omegas {
        if let Some(&idx) = name_to_idx.get(spec.name.as_str()) {
            matrix[(idx, idx)] = spec.variance;
        }
    }

    // Fill block entries from block specs (lower triangle, row-wise)
    for block in block_omegas {
        let block_n = block.names.len();
        let mut val_idx = 0;
        for row in 0..block_n {
            let i = *name_to_idx.get(block.names[row].as_str()).ok_or_else(|| {
                format!("block_omega references unknown eta '{}'", block.names[row])
            })?;
            for col in 0..=row {
                let j = *name_to_idx.get(block.names[col].as_str()).ok_or_else(|| {
                    format!("block_omega references unknown eta '{}'", block.names[col])
                })?;
                matrix[(i, j)] = block.lower_triangle[val_idx];
                matrix[(j, i)] = block.lower_triangle[val_idx]; // symmetric
                val_idx += 1;
            }
        }
    }

    Ok(OmegaMatrix::from_matrix(matrix, eta_names.to_vec(), false))
}

// --- Structural model parsing ---

fn parse_structural_model(lines: &[String]) -> Result<(PkModel, HashMap<String, String>), String> {
    // pk model_name(param=VAR, param=VAR, ...)
    let pk_re = Regex::new(r"pk\s+(\w+)\(([^)]+)\)").unwrap();

    for line in lines {
        if let Some(caps) = pk_re.captures(line) {
            let model_name = &caps[1];
            let pk_model = match model_name {
                "one_cpt_iv_bolus" | "one_compartment_iv_bolus" => PkModel::OneCptIvBolus,
                "one_cpt_oral" | "one_compartment_oral" => PkModel::OneCptOral,
                "one_cpt_infusion" | "one_compartment_infusion" => PkModel::OneCptInfusion,
                "two_cpt_iv_bolus" | "two_compartment_iv_bolus" => PkModel::TwoCptIvBolus,
                "two_cpt_oral" | "two_compartment_oral" => PkModel::TwoCptOral,
                "two_cpt_infusion" | "two_compartment_infusion" => PkModel::TwoCptInfusion,
                "three_cpt_iv_bolus" | "three_compartment_iv_bolus" => PkModel::ThreeCptIvBolus,
                "three_cpt_oral" | "three_compartment_oral" => PkModel::ThreeCptOral,
                "three_cpt_infusion" | "three_compartment_infusion" => PkModel::ThreeCptInfusion,
                other => return Err(format!("Unknown PK model: {}", other)),
            };

            let params_str = &caps[2];
            let mut param_map = HashMap::new();
            for pair in params_str.split(',') {
                let parts: Vec<&str> = pair.split('=').map(|s| s.trim()).collect();
                if parts.len() == 2 {
                    param_map.insert(parts[0].to_lowercase(), parts[1].to_string());
                }
            }

            return Ok((pk_model, param_map));
        }
    }

    Err("No PK model found in [structural_model] block".to_string())
}

// --- Error model parsing ---

fn parse_error_model(lines: &[String]) -> Result<(ErrorModel, Vec<String>), String> {
    // DV ~ proportional(SIGMA_NAME)
    // DV ~ additive(SIGMA_NAME)
    // DV ~ combined(SIGMA1, SIGMA2)
    let re = Regex::new(r"(\w+)\s*~\s*(\w+)\(([^)]+)\)").unwrap();

    for line in lines {
        if let Some(caps) = re.captures(line) {
            let error_type = &caps[2];
            let sigma_names: Vec<String> =
                caps[3].split(',').map(|s| s.trim().to_string()).collect();

            let error_model = match error_type.to_lowercase().as_str() {
                "additive" => ErrorModel::Additive,
                "proportional" => ErrorModel::Proportional,
                "combined" => ErrorModel::Combined,
                other => return Err(format!("Unknown error model: {}", other)),
            };

            return Ok((error_model, sigma_names));
        }
    }

    Err("No error model found in [error_model] block".to_string())
}

// --- Individual parameter function builder ---

/// Build the PK parameter function from [individual_parameters] block.
///
/// Each line is an assignment like:
///   CL = TVCL * exp(ETA_CL) * (WT/70)^0.75
///   V  = TVV * exp(ETA_V)
///   KA = TVKA
///
/// We parse these into a closure that maps (theta, eta, covariates) -> PkParams.
fn build_pk_param_fn(
    lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
    pk_param_map: &HashMap<String, String>,
) -> Result<PkParamFn, String> {
    let mut assignments: Vec<(String, Expression)> = Vec::new();

    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid individual parameter line: {}", line));
        }
        let var_name = parts[0].trim().to_string();
        let expr_str = parts[1].trim();
        let expr = parse_expression(expr_str, theta_names, eta_names)?;
        assignments.push((var_name, expr));
    }

    let pk_map: HashMap<String, String> = pk_param_map.clone();
    let assignments_owned = assignments;

    Ok(Box::new(
        move |theta: &[f64], eta: &[f64], covariates: &HashMap<String, f64>| {
            let mut vars: HashMap<String, f64> = HashMap::new();

            for (var_name, expr) in &assignments_owned {
                let val = eval_expression(expr, theta, eta, covariates, &vars);
                vars.insert(var_name.clone(), val);
            }

            let mut p = PkParams::default();

            if pk_map.is_empty() {
                // ODE model or no pk_param_map: store individual params by declaration order
                for (i, (var_name, _)) in assignments_owned.iter().enumerate() {
                    if i < MAX_PK_PARAMS {
                        if let Some(&val) = vars.get(var_name) {
                            p.values[i] = val;
                        }
                    }
                }
            } else {
                // Analytical model: map pk_param_name → value via pk_param_map
                let mut named = HashMap::new();
                for (pk_name, var_name) in &pk_map {
                    if let Some(&val) = vars.get(var_name) {
                        named.insert(pk_name.clone(), val);
                    } else if let Some(&val) = vars.get(&var_name.to_lowercase()) {
                        named.insert(pk_name.clone(), val);
                    }
                }
                p = PkParams::from_hashmap(&named);
            }

            p
        },
    ))
}

// --- Simple expression AST and evaluator ---

#[derive(Debug, Clone)]
enum Expression {
    Literal(f64),
    Theta(usize),
    Eta(usize),
    Covariate(String),
    Variable(String),
    BinOp(Box<Expression>, BinOp, Box<Expression>),
    UnaryFn(String, Box<Expression>),
    Power(Box<Expression>, Box<Expression>),
}

#[derive(Debug, Clone)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}

fn parse_expression(
    s: &str,
    theta_names: &[String],
    eta_names: &[String],
) -> Result<Expression, String> {
    let tokens = tokenize(s)?;
    let (expr, _) = parse_add_sub(&tokens, 0, theta_names, eta_names)?;
    Ok(expr)
}

fn eval_expression(
    expr: &Expression,
    theta: &[f64],
    eta: &[f64],
    covariates: &HashMap<String, f64>,
    vars: &HashMap<String, f64>,
) -> f64 {
    match expr {
        Expression::Literal(v) => *v,
        Expression::Theta(i) => theta[*i],
        Expression::Eta(i) => eta[*i],
        Expression::Covariate(name) => covariates.get(&name.to_lowercase()).copied().unwrap_or(0.0),
        Expression::Variable(name) => vars.get(name).copied().unwrap_or(0.0),
        Expression::BinOp(lhs, op, rhs) => {
            let l = eval_expression(lhs, theta, eta, covariates, vars);
            let r = eval_expression(rhs, theta, eta, covariates, vars);
            match op {
                BinOp::Add => l + r,
                BinOp::Sub => l - r,
                BinOp::Mul => l * r,
                BinOp::Div => {
                    if r.abs() < 1e-30 {
                        0.0
                    } else {
                        l / r
                    }
                }
            }
        }
        Expression::UnaryFn(name, arg) => {
            let v = eval_expression(arg, theta, eta, covariates, vars);
            match name.as_str() {
                "exp" => v.exp(),
                "log" | "ln" => v.max(1e-30).ln(),
                "sqrt" => v.max(0.0).sqrt(),
                "abs" => v.abs(),
                _ => v,
            }
        }
        Expression::Power(base, exp) => {
            let b = eval_expression(base, theta, eta, covariates, vars);
            let e = eval_expression(exp, theta, eta, covariates, vars);
            b.powf(e)
        }
    }
}

// --- Tokenizer ---

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    LParen,
    RParen,
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
}

fn tokenize(s: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        match chars[i] {
            ' ' | '\t' => i += 1,
            '(' => {
                tokens.push(Token::LParen);
                i += 1;
            }
            ')' => {
                tokens.push(Token::RParen);
                i += 1;
            }
            '+' => {
                tokens.push(Token::Plus);
                i += 1;
            }
            '-' => {
                // Check if this is a negative number (after operator or at start)
                let is_unary = tokens.is_empty()
                    || matches!(
                        tokens.last(),
                        Some(
                            Token::LParen
                                | Token::Plus
                                | Token::Minus
                                | Token::Star
                                | Token::Slash
                                | Token::Caret
                        )
                    );
                if is_unary
                    && i + 1 < chars.len()
                    && (chars[i + 1].is_ascii_digit() || chars[i + 1] == '.')
                {
                    let start = i;
                    i += 1;
                    while i < chars.len()
                        && (chars[i].is_ascii_digit()
                            || chars[i] == '.'
                            || chars[i] == 'e'
                            || chars[i] == 'E')
                    {
                        i += 1;
                    }
                    let num_str: String = chars[start..i].iter().collect();
                    let num: f64 = num_str
                        .parse()
                        .map_err(|_| format!("Bad number: {}", num_str))?;
                    tokens.push(Token::Number(num));
                } else {
                    tokens.push(Token::Minus);
                    i += 1;
                }
            }
            '*' => {
                tokens.push(Token::Star);
                i += 1;
            }
            '/' => {
                tokens.push(Token::Slash);
                i += 1;
            }
            '^' => {
                tokens.push(Token::Caret);
                i += 1;
            }
            c if c.is_ascii_digit() || c == '.' => {
                let start = i;
                while i < chars.len()
                    && (chars[i].is_ascii_digit()
                        || chars[i] == '.'
                        || chars[i] == 'e'
                        || chars[i] == 'E')
                {
                    i += 1;
                }
                let num_str: String = chars[start..i].iter().collect();
                let num: f64 = num_str
                    .parse()
                    .map_err(|_| format!("Bad number: {}", num_str))?;
                tokens.push(Token::Number(num));
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let start = i;
                while i < chars.len() && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let ident: String = chars[start..i].iter().collect();
                tokens.push(Token::Ident(ident));
            }
            other => return Err(format!("Unexpected character: {}", other)),
        }
    }

    Ok(tokens)
}

// --- Recursive descent parser ---

fn parse_add_sub(
    tokens: &[Token],
    pos: usize,
    theta_names: &[String],
    eta_names: &[String],
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_mul_div(tokens, pos, theta_names, eta_names)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, theta_names, eta_names)?;
                left = Expression::BinOp(Box::new(left), BinOp::Add, Box::new(right));
                pos = p;
            }
            Token::Minus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, theta_names, eta_names)?;
                left = Expression::BinOp(Box::new(left), BinOp::Sub, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_mul_div(
    tokens: &[Token],
    pos: usize,
    theta_names: &[String],
    eta_names: &[String],
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_power(tokens, pos, theta_names, eta_names)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Star => {
                let (right, p) = parse_power(tokens, pos + 1, theta_names, eta_names)?;
                left = Expression::BinOp(Box::new(left), BinOp::Mul, Box::new(right));
                pos = p;
            }
            Token::Slash => {
                let (right, p) = parse_power(tokens, pos + 1, theta_names, eta_names)?;
                left = Expression::BinOp(Box::new(left), BinOp::Div, Box::new(right));
                pos = p;
            }
            _ => break,
        }
    }

    Ok((left, pos))
}

fn parse_power(
    tokens: &[Token],
    pos: usize,
    theta_names: &[String],
    eta_names: &[String],
) -> Result<(Expression, usize), String> {
    let (base, mut pos) = parse_atom(tokens, pos, theta_names, eta_names)?;

    if pos < tokens.len() && tokens[pos] == Token::Caret {
        let (exp, p) = parse_atom(tokens, pos + 1, theta_names, eta_names)?;
        pos = p;
        return Ok((Expression::Power(Box::new(base), Box::new(exp)), pos));
    }

    Ok((base, pos))
}

fn parse_atom(
    tokens: &[Token],
    pos: usize,
    theta_names: &[String],
    eta_names: &[String],
) -> Result<(Expression, usize), String> {
    if pos >= tokens.len() {
        return Err("Unexpected end of expression".to_string());
    }

    match &tokens[pos] {
        Token::Minus => {
            // Unary minus: -expr → 0 - expr
            let (expr, p) = parse_atom(tokens, pos + 1, theta_names, eta_names)?;
            Ok((
                Expression::BinOp(
                    Box::new(Expression::Literal(0.0)),
                    BinOp::Sub,
                    Box::new(expr),
                ),
                p,
            ))
        }
        Token::Number(n) => Ok((Expression::Literal(*n), pos + 1)),
        Token::LParen => {
            let (expr, p) = parse_add_sub(tokens, pos + 1, theta_names, eta_names)?;
            if p >= tokens.len() || tokens[p] != Token::RParen {
                return Err("Missing closing parenthesis".to_string());
            }
            Ok((expr, p + 1))
        }
        Token::Ident(name) => {
            // Check if it's a function call: name(expr)
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen {
                let func_name = name.to_lowercase();
                let (arg, p) = parse_add_sub(tokens, pos + 2, theta_names, eta_names)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err(format!("Missing closing parenthesis for function {}", name));
                }
                return Ok((Expression::UnaryFn(func_name, Box::new(arg)), p + 1));
            }

            // Check if it's a theta
            if let Some(idx) = theta_names.iter().position(|n| n == name) {
                return Ok((Expression::Theta(idx), pos + 1));
            }

            // Check if it's an eta
            if let Some(idx) = eta_names.iter().position(|n| n == name) {
                return Ok((Expression::Eta(idx), pos + 1));
            }

            // Must be a covariate or previously assigned variable
            // Heuristic: UPPERCASE names with no matching theta/eta are covariates
            // (when theta_names is empty, e.g. ODE context, treat all as Variable)
            if !theta_names.is_empty()
                && name
                    .chars()
                    .all(|c| c.is_uppercase() || c.is_ascii_digit() || c == '_')
            {
                Ok((Expression::Covariate(name.clone()), pos + 1))
            } else {
                Ok((Expression::Variable(name.clone()), pos + 1))
            }
        }
        other => Err(format!("Unexpected token: {:?}", other)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_diagonal_omega() {
        let lines = vec![
            "omega ETA_CL ~ 0.07".to_string(),
            "omega ETA_V  ~ 0.02".to_string(),
        ];
        let (_, omegas, block_omegas, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 2);
        assert_eq!(block_omegas.len(), 0);
        assert_eq!(omegas[0].name, "ETA_CL");
        assert!((omegas[0].variance - 0.07).abs() < 1e-10);
    }

    #[test]
    fn test_parse_block_omega() {
        let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string()];
        let (_, omegas, block_omegas, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 0);
        assert_eq!(block_omegas.len(), 1);
        assert_eq!(block_omegas[0].names, vec!["ETA_CL", "ETA_V"]);
        assert_eq!(block_omegas[0].lower_triangle, vec![0.09, 0.02, 0.04]);
    }

    #[test]
    fn test_parse_block_omega_3x3() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V, ETA_KA) = [0.09, 0.01, 0.04, 0.005, 0.002, 0.16]"
                .to_string(),
        ];
        let (_, _, block_omegas, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(block_omegas[0].names.len(), 3);
        assert_eq!(block_omegas[0].lower_triangle.len(), 6); // 3*(3+1)/2
    }

    #[test]
    fn test_parse_block_omega_wrong_count() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02]".to_string(), // needs 3, got 2
        ];
        let result = parse_parameters(&lines);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_mixed_diagonal_and_block() {
        let lines = vec![
            "omega ETA_KA ~ 0.40".to_string(),
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
        ];
        let (_, omegas, block_omegas, _, eta_names) = parse_parameters(&lines).unwrap();
        assert_eq!(omegas.len(), 1);
        assert_eq!(block_omegas.len(), 1);
        // Declaration order preserved: ETA_KA first, then block (ETA_CL, ETA_V)
        assert_eq!(eta_names, vec!["ETA_KA", "ETA_CL", "ETA_V"]);
    }

    #[test]
    fn test_declaration_order_block_before_diagonal() {
        let lines = vec![
            "block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]".to_string(),
            "omega ETA_KA ~ 0.40".to_string(),
        ];
        let (_, _, _, _, eta_names) = parse_parameters(&lines).unwrap();
        // block_omega declared first, so ETA_CL, ETA_V come before ETA_KA
        assert_eq!(eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
    }

    #[test]
    fn test_build_omega_matrix_diagonal_only() {
        let diag = vec![
            OmegaSpec {
                name: "ETA_CL".into(),
                variance: 0.09,
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
            },
        ];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&diag, &[], &names).unwrap();
        assert!(omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
        assert!((omega.matrix[(0, 1)]).abs() < 1e-10);
    }

    #[test]
    fn test_build_omega_matrix_block() {
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&[], &block, &names).unwrap();
        assert!(!omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10);
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10);
        assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10);
        assert!((omega.matrix[(1, 0)] - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_build_omega_matrix_mixed() {
        let diag = vec![OmegaSpec {
            name: "ETA_KA".into(),
            variance: 0.16,
        }];
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
        }];
        let names = vec!["ETA_KA".into(), "ETA_CL".into(), "ETA_V".into()];
        let omega = build_omega_matrix(&diag, &block, &names).unwrap();
        assert!(!omega.diagonal);
        assert!((omega.matrix[(0, 0)] - 0.16).abs() < 1e-10); // ETA_KA
        assert!((omega.matrix[(1, 1)] - 0.09).abs() < 1e-10); // ETA_CL
        assert!((omega.matrix[(2, 2)] - 0.04).abs() < 1e-10); // ETA_V
        assert!((omega.matrix[(1, 2)] - 0.02).abs() < 1e-10); // cov(CL, V)
        assert!((omega.matrix[(0, 1)]).abs() < 1e-10); // no cov(KA, CL)
    }

    #[test]
    fn test_parse_full_model_with_block_omega() {
        let content = r#"
# Test model with block omega

[parameters]
  theta TVCL(0.134, 0.001, 10.0)
  theta TVV(8.1, 0.1, 500.0)
  theta TVKA(1.0, 0.01, 50.0)

  block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04]
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
"#;
        let parsed = parse_full_model(content).unwrap();
        let omega = &parsed.model.default_params.omega;
        assert_eq!(omega.dim(), 3);
        assert!(!omega.diagonal);
        // Eta names preserve declaration order from the model file
        assert_eq!(omega.eta_names, vec!["ETA_CL", "ETA_V", "ETA_KA"]);
        // ETA_CL = index 0, ETA_V = index 1, ETA_KA = index 2
        assert!((omega.matrix[(0, 0)] - 0.09).abs() < 1e-10); // ETA_CL
        assert!((omega.matrix[(1, 1)] - 0.04).abs() < 1e-10); // ETA_V
        assert!((omega.matrix[(2, 2)] - 0.40).abs() < 1e-10); // ETA_KA
        assert!((omega.matrix[(0, 1)] - 0.02).abs() < 1e-10); // cov(CL, V)
    }

    // ── fit_options parsing: new optimizer choices ──────────────────────────

    fn minimal_model_with_fit_options(fit_opts: &str) -> String {
        format!(
            r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)
  theta TVKA(1.5, 0.01, 50.0)
  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04
  omega ETA_KA ~ 0.30
  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)
  KA = TVKA * exp(ETA_KA)

[structural_model]
  pk one_cpt_oral(cl=CL, v=V, ka=KA)

[error_model]
  DV ~ proportional(PROP_ERR)

[fit_options]
{}
"#,
            fit_opts
        )
    }

    #[test]
    fn test_parse_optimizer_bobyqa() {
        let content = minimal_model_with_fit_options("  optimizer = bobyqa");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_parse_optimizer_trust_region() {
        let content = minimal_model_with_fit_options("  optimizer = trust_region");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_newton_tr_alias() {
        // newton_tr is an accepted alias for trust_region
        let content = minimal_model_with_fit_options("  optimizer = newton_tr");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_case_insensitive() {
        // Parser lowercases the value, so mixed-case should map the same way.
        let content = minimal_model_with_fit_options("  optimizer = BOBYQA");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);

        let content2 = minimal_model_with_fit_options("  optimizer = Trust_Region");
        let parsed2 = parse_full_model(&content2).unwrap();
        assert_eq!(parsed2.fit_options.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_parse_optimizer_defaults_to_slsqp() {
        // No [fit_options] block → default optimizer.
        let content = minimal_model_with_fit_options("  maxiter = 100");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Slsqp);
    }

    #[test]
    fn test_parse_steihaug_max_iters() {
        let content = minimal_model_with_fit_options(
            "  optimizer = trust_region\n  steihaug_max_iters = 30",
        );
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
        assert_eq!(parsed.fit_options.steihaug_max_iters, 30);
    }

    #[test]
    fn test_steihaug_max_iters_default() {
        // Default must match the documented value (50).
        let content = minimal_model_with_fit_options("  optimizer = trust_region");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.steihaug_max_iters, 50);
    }

    #[test]
    fn test_parse_inner_maxiter_and_tol() {
        let content = minimal_model_with_fit_options(
            "  inner_maxiter = 75\n  inner_tol = 1e-5",
        );
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.inner_maxiter, 75);
        assert!((parsed.fit_options.inner_tol - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_fit_options_defaults() {
        // Guard against accidental drift in defaults — documented as:
        //   optimizer = slsqp, inner_maxiter = 200, inner_tol = 1e-8,
        //   steihaug_max_iters = 50.
        let opts = FitOptions::default();
        assert_eq!(opts.optimizer, Optimizer::Slsqp);
        assert_eq!(opts.inner_maxiter, 200);
        assert!((opts.inner_tol - 1e-8).abs() < 1e-20);
        assert_eq!(opts.steihaug_max_iters, 50);
    }

    #[test]
    fn test_parse_example_warfarin_bobyqa_file() {
        // The example file is part of the user-visible surface; parsing it is
        // a lightweight smoke test that the key names match what the docs
        // and examples advertise.
        let content = include_str!("../../examples/warfarin_bobyqa.ferx");
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
        assert_eq!(parsed.fit_options.outer_maxiter, 300);
        assert_eq!(parsed.fit_options.inner_maxiter, 100);
    }

    #[test]
    fn test_parse_example_warfarin_trust_region_file() {
        let content = include_str!("../../examples/warfarin_trust_region.ferx");
        let parsed = parse_full_model(content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::TrustRegion);
        assert_eq!(parsed.fit_options.steihaug_max_iters, 30);
    }

    // ── apply_fit_option: shared dispatch used by the R wrapper's `settings`
    //     argument. Keep these assertions in sync with the documented keys.

    #[test]
    fn test_apply_fit_option_optimizer_bobyqa() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "optimizer", "bobyqa"), Ok(true));
        assert_eq!(opts.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_apply_fit_option_optimizer_trust_region() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "optimizer", "trust_region"),
            Ok(true)
        );
        assert_eq!(opts.optimizer, Optimizer::TrustRegion);
    }

    #[test]
    fn test_apply_fit_option_steihaug_max_iters() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "steihaug_max_iters", "30"),
            Ok(true)
        );
        assert_eq!(opts.steihaug_max_iters, 30);
    }

    #[test]
    fn test_apply_fit_option_inner_maxiter_and_tol() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "inner_maxiter", "75"),
            Ok(true)
        );
        assert_eq!(opts.inner_maxiter, 75);

        assert_eq!(apply_fit_option(&mut opts, "inner_tol", "1e-5"), Ok(true));
        assert!((opts.inner_tol - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_apply_fit_option_unknown_key_returns_false() {
        let mut opts = FitOptions::default();
        // Misspelled key — no error, just not applied. The R wrapper turns
        // this into a user-facing error; the .ferx parser stays tolerant.
        assert_eq!(
            apply_fit_option(&mut opts, "n_exploraton", "200"),
            Ok(false)
        );
        // `method` is deliberately not handled here.
        assert_eq!(apply_fit_option(&mut opts, "method", "focei"), Ok(false));
    }

    #[test]
    fn test_apply_fit_option_malformed_value_errors() {
        let mut opts = FitOptions::default();
        assert!(apply_fit_option(&mut opts, "inner_maxiter", "oops").is_err());
        assert!(apply_fit_option(&mut opts, "inner_tol", "not_a_num").is_err());
        assert!(apply_fit_option(&mut opts, "steihaug_max_iters", "-1").is_err());
        assert!(apply_fit_option(&mut opts, "covariance", "maybe").is_err());
        assert!(apply_fit_option(&mut opts, "optimizer", "does_not_exist").is_err());
        assert!(apply_fit_option(&mut opts, "bloq_method", "nope").is_err());
    }

    #[test]
    fn test_apply_fit_option_bool_variants() {
        let mut opts = FitOptions::default();
        for v in ["true", "TRUE", "t", "yes", "1", "on"] {
            opts.sir = false;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(opts.sir, "value `{v}` should parse as true");
        }
        for v in ["false", "FALSE", "f", "no", "0", "off"] {
            opts.sir = true;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(!opts.sir, "value `{v}` should parse as false");
        }
    }
}
