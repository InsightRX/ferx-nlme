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
    let (thetas, omegas, sigmas) = parse_parameters(param_lines)?;

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
    let eta_names: Vec<String> = omegas.iter().map(|o| o.name.clone()).collect();
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
    let omega_variances: Vec<f64> = omegas.iter().map(|o| o.variance).collect();
    let omega = OmegaMatrix::from_diagonal(&omega_variances, eta_names.clone());
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
    };

    // ── Optional blocks ──
    let simulation = blocks
        .get("simulation")
        .map(|lines| parse_simulation_block(lines))
        .transpose()?;
    let (init_theta, init_omega, init_sigma) = if let Some(lines) = blocks.get("initial_values") {
        parse_initial_values(lines)?
    } else {
        (None, None, None)
    };
    let fit_options = if let Some(lines) = blocks.get("fit_options") {
        parse_fit_options(lines)?
    } else {
        FitOptions::default()
    };

    Ok(ParsedModel {
        model,
        simulation,
        init_theta,
        init_omega,
        init_sigma,
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

// ── [initial_values] block parser ───────────────────────────────────────────

fn parse_initial_values(
    lines: &[String],
) -> Result<(Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>), String> {
    let mut theta = None;
    let mut omega = None;
    let mut sigma = None;

    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        match parts[0] {
            "theta" => theta = Some(parse_float_array(parts[1])?),
            "omega" => omega = Some(parse_float_array(parts[1])?),
            "sigma" => sigma = Some(parse_float_array(parts[1])?),
            _ => {}
        }
    }
    Ok((theta, omega, sigma))
}

// ── [fit_options] block parser ──────────────────────────────────────────────

fn parse_fit_options(lines: &[String]) -> Result<FitOptions, String> {
    let mut opts = FitOptions::default();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        match parts[0] {
            "method" => {
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
            }
            "maxiter" => opts.outer_maxiter = parts[1].parse().unwrap_or(500),
            "covariance" => opts.run_covariance_step = parts[1].trim() == "true",
            "optimizer" => {
                opts.optimizer = match parts[1].to_lowercase().as_str() {
                    "slsqp" => Optimizer::Slsqp,
                    "lbfgs" | "nlopt_lbfgs" => Optimizer::NloptLbfgs,
                    "mma" => Optimizer::Mma,
                    "bfgs" => Optimizer::Bfgs,
                    _ => Optimizer::Slsqp,
                };
            }
            "global_search" => opts.global_search = parts[1].trim() == "true",
            "global_maxeval" => opts.global_maxeval = parts[1].parse().unwrap_or(0),
            "n_exploration" => opts.saem_n_exploration = parts[1].trim().parse().unwrap_or(150),
            "n_convergence" => opts.saem_n_convergence = parts[1].trim().parse().unwrap_or(250),
            "n_mh_steps" => opts.saem_n_mh_steps = parts[1].trim().parse().unwrap_or(3),
            "adapt_interval" => opts.saem_adapt_interval = parts[1].trim().parse().unwrap_or(50),
            "seed" => opts.saem_seed = parts[1].trim().parse().ok(),
            "sir" => opts.sir = parts[1].trim() == "true",
            "sir_samples" => opts.sir_samples = parts[1].trim().parse().unwrap_or(1000),
            "sir_resamples" => opts.sir_resamples = parts[1].trim().parse().unwrap_or(500),
            "sir_seed" => opts.sir_seed = parts[1].trim().parse().ok(),
            _ => {}
        }
    }
    Ok(opts)
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
) -> Result<(Vec<ThetaSpec>, Vec<OmegaSpec>, Vec<SigmaSpec>), String> {
    let mut thetas = Vec::new();
    let mut omegas = Vec::new();
    let mut sigmas = Vec::new();

    // theta NAME(init, lower, upper)  or  theta NAME(init)
    let theta_re = Regex::new(
        r"theta\s+(\w+)\(\s*([0-9eE.+-]+)\s*(?:,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+)\s*)?\)",
    )
    .unwrap();

    // omega NAME ~ value
    let omega_re = Regex::new(r"omega\s+(\w+)\s*~\s*([0-9eE.+-]+)").unwrap();

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
        } else if let Some(caps) = omega_re.captures(line) {
            let name = caps[1].to_string();
            let variance: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad omega: {}", line))?;
            omegas.push(OmegaSpec { name, variance });
        } else if let Some(caps) = sigma_re.captures(line) {
            let name = caps[1].to_string();
            let value: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad sigma: {}", line))?;
            sigmas.push(SigmaSpec { name, value });
        }
    }

    Ok((thetas, omegas, sigmas))
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
