use crate::types::*;
use regex::Regex;
use std::collections::HashMap;
use std::path::Path;

// ── Mu-referencing pattern detection ────────────────────────────────────────

/// Walk a Mul-chain and collect direct Theta indices (not inside any function).
fn collect_mul_thetas(expr: &Expression, out: &mut Vec<usize>) {
    match expr {
        Expression::Theta(i) => out.push(*i),
        Expression::BinOp(l, BinOp::Mul, r) => {
            collect_mul_thetas(l, out);
            collect_mul_thetas(r, out);
        }
        _ => {}
    }
}

/// Walk a Mul-chain and find the first `exp(Eta(j))`, returning the eta index.
fn find_exp_eta_in_mul(expr: &Expression) -> Option<usize> {
    match expr {
        Expression::UnaryFn(name, arg) if name == "exp" => {
            if let Expression::Eta(j) = arg.as_ref() {
                return Some(*j);
            }
            None
        }
        Expression::BinOp(l, BinOp::Mul, r) => {
            find_exp_eta_in_mul(l).or_else(|| find_exp_eta_in_mul(r))
        }
        _ => None,
    }
}

/// Collect all eta indices referenced by an expression (e.g. `Eta(2)` appears
/// inside `TVQ * exp(ETA_V2)` → `[2]`). Used to build the AD path's per-tv
/// eta-index map so parameters without etas (e.g. `Q = TVQ`) are handled
/// correctly — otherwise the AD loop would misalign `eta[i]` with `pk[i]`
/// and either apply the wrong eta or leave a pk slot at 0.
fn extract_eta_indices(expr: &Expression) -> Vec<usize> {
    let mut out = Vec::new();
    fn walk(e: &Expression, out: &mut Vec<usize>) {
        match e {
            Expression::Eta(i) => {
                if !out.contains(i) {
                    out.push(*i);
                }
            }
            Expression::BinOp(l, _, r) => {
                walk(l, out);
                walk(r, out);
            }
            Expression::UnaryFn(_, a) => walk(a, out),
            Expression::Power(b, e) => {
                walk(b, out);
                walk(e, out);
            }
            _ => {}
        }
    }
    walk(expr, &mut out);
    out
}

/// Detect mu-referencing patterns in one assignment expression.
/// Returns `Some((eta_idx, theta_idx, log_transformed))` or `None`.
fn detect_pattern(expr: &Expression) -> Option<(usize, usize, bool)> {
    match expr {
        // Pattern 2: exp(log(THETA) + ETA)
        Expression::UnaryFn(name, inner) if name == "exp" => {
            // inner must be Add with log(Theta) and Eta in either order
            if let Expression::BinOp(lhs, BinOp::Add, rhs) = inner.as_ref() {
                let try_log_theta_eta =
                    |a: &Expression, b: &Expression| -> Option<(usize, usize)> {
                        if let Expression::UnaryFn(fn_name, fn_arg) = a {
                            if fn_name == "log" || fn_name == "ln" {
                                if let Expression::Theta(ti) = fn_arg.as_ref() {
                                    if let Expression::Eta(ei) = b {
                                        return Some((*ei, *ti));
                                    }
                                }
                            }
                        }
                        None
                    };
                if let Some((ei, ti)) =
                    try_log_theta_eta(lhs, rhs).or_else(|| try_log_theta_eta(rhs, lhs))
                {
                    return Some((ei, ti, true));
                }
            }
            None
        }
        // Pattern 3: THETA + ETA or ETA + THETA
        Expression::BinOp(lhs, BinOp::Add, rhs) => match (lhs.as_ref(), rhs.as_ref()) {
            (Expression::Theta(ti), Expression::Eta(ei)) => Some((*ei, *ti, false)),
            (Expression::Eta(ei), Expression::Theta(ti)) => Some((*ei, *ti, false)),
            _ => None,
        },
        // Pattern 1 / 4: product containing Theta and exp(Eta)
        _ => {
            let mut thetas = Vec::new();
            collect_mul_thetas(expr, &mut thetas);
            if thetas.len() == 1 {
                if let Some(ei) = find_exp_eta_in_mul(expr) {
                    return Some((ei, thetas[0], true));
                }
            }
            None
        }
    }
}

/// Analyse [individual_parameters] lines and detect mu-referencing relationships.
fn detect_mu_refs(
    indiv_lines: &[String],
    theta_names: &[String],
    eta_names: &[String],
) -> HashMap<String, MuRef> {
    let mut result = HashMap::new();
    for line in indiv_lines {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            continue;
        }
        let expr_str = parts[1].trim();
        let ctx = ParseCtx::new(theta_names, eta_names, &[]);
        let expr = match parse_expression(expr_str, ctx) {
            Ok(e) => e,
            Err(_) => continue,
        };
        if let Some((eta_idx, theta_idx, log_transformed)) = detect_pattern(&expr) {
            if eta_idx < eta_names.len() && theta_idx < theta_names.len() {
                result.insert(
                    eta_names[eta_idx].clone(),
                    MuRef {
                        theta_name: theta_names[theta_idx].clone(),
                        log_transformed,
                    },
                );
            }
        }
    }
    result
}

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

    let (pk_param_fn, referenced_covariates) =
        build_pk_param_fn(indiv_lines, &theta_names, &eta_names, &pk_param_map)?;

    let theta_values: Vec<f64> = thetas.iter().map(|t| t.init).collect();
    let theta_lower: Vec<f64> = thetas.iter().map(|t| t.lower).collect();
    let theta_upper: Vec<f64> = thetas.iter().map(|t| t.upper).collect();
    let theta_fixed: Vec<bool> = thetas.iter().map(|t| t.fixed).collect();
    let omega = build_omega_matrix(&omegas, &block_omegas, &eta_names)?;
    let omega_fixed = build_omega_fixed(&omegas, &block_omegas, &eta_names)?;
    let sigma_values: Vec<f64> = sigmas.iter().map(|s| s.value).collect();
    let sigma_fixed: Vec<bool> = sigmas.iter().map(|s| s.fixed).collect();
    let sigma = SigmaVector {
        values: sigma_values,
        names: sigma_names,
    };

    let default_params = ModelParameters {
        theta: theta_values,
        theta_names: theta_names.clone(),
        theta_lower,
        theta_upper,
        theta_fixed,
        omega,
        omega_fixed,
        sigma,
        sigma_fixed,
    };

    // Auto-generate tv_fn: evaluate individual parameters with eta=0
    // This gives covariate-adjusted typical values for the AD inner loop.
    let tv_assignments = indiv_lines.clone();
    let tv_theta_names = theta_names.clone();
    let tv_eta_names = eta_names.clone();
    let tv_fn: Option<Box<dyn Fn(&[f64], &HashMap<String, f64>) -> Vec<f64> + Send + Sync>> =
        if !is_ode {
            let mut assignments: Vec<(String, Expression)> = Vec::new();
            let mut tv_defined: Vec<String> = Vec::new();
            for line in tv_assignments.iter() {
                let parts: Vec<&str> = line.splitn(2, '=').collect();
                if parts.len() != 2 {
                    continue;
                }
                let var_name = parts[0].trim().to_string();
                let expr_str = parts[1].trim();
                let ctx = ParseCtx::new(&tv_theta_names, &tv_eta_names, &tv_defined);
                if let Ok(expr) = parse_expression(expr_str, ctx) {
                    assignments.push((var_name.clone(), expr));
                    tv_defined.push(var_name);
                }
            }

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

    // Detect mu-referencing relationships from [individual_parameters]
    let mu_refs = detect_mu_refs(indiv_lines, &theta_names, &eta_names);

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

    // Per-tv eta index: for each [individual_parameters] line, find which eta
    // its expression references (or -1 for none). Used by the AD path so
    // `pk[pk_indices[i]] = tv[i] * exp(eta[eta_map[i]])` stays correct even
    // when some params are eta-free (e.g. `Q = TVQ`). If a line references
    // multiple etas (unusual), pick the first.
    let eta_map: Vec<i32> = indiv_lines
        .iter()
        .map(|line| {
            let parts: Vec<&str> = line.splitn(2, '=').collect();
            if parts.len() != 2 {
                return -1i32;
            }
            let ctx = ParseCtx::new(&theta_names, &eta_names, &[]);
            match parse_expression(parts[1].trim(), ctx) {
                Ok(expr) => extract_eta_indices(&expr)
                    .first()
                    .copied()
                    .map(|i| i as i32)
                    .unwrap_or(-1),
                Err(_) => -1,
            }
        })
        .collect();

    let pk_idx_f64: Vec<f64> = pk_indices.iter().map(|&i| i as f64).collect();
    // sel_flat is n_tv × n_eta (n_tv = eta_map.len() = number of
    // individual_parameters lines). pk_indices.len() can differ from
    // eta_map.len() for ODE models (where pk_indices is synthesized as
    // `(0..n_eta).collect()`), so size from eta_map to avoid an OOB panic
    // when the two disagree. The AD path is only taken when tv_fn is
    // populated (analytical models), where the two lengths match.
    let n_tv = eta_map.len();
    let mut sel_flat = vec![0.0f64; n_tv * n_eta];
    for (i, &em) in eta_map.iter().enumerate() {
        if em >= 0 && (em as usize) < n_eta {
            sel_flat[i * n_eta + em as usize] = 1.0;
        }
    }

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
        eta_map,
        pk_idx_f64,
        sel_flat,
        ode_spec,
        bloq_method: BloqMethod::Drop,
        mu_refs,
        referenced_covariates,
        gradient_method: GradientMethod::default(),
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

fn parse_method_token(token: &str) -> Result<EstimationMethod, String> {
    let val = token
        .trim()
        .trim_matches(|c| c == '"' || c == '\'')
        .to_lowercase();
    if val == "saem" {
        Ok(EstimationMethod::Saem)
    } else if val.contains("hybrid") || val == "gn_hybrid" || val == "gn-hybrid" {
        Ok(EstimationMethod::FoceGnHybrid)
    } else if val == "gn" || val.contains("gauss") {
        Ok(EstimationMethod::FoceGn)
    } else if val == "focei" || val == "foce-i" || val == "foce_i" || val.contains("interaction") {
        Ok(EstimationMethod::FoceI)
    } else if val == "foce" {
        Ok(EstimationMethod::Foce)
    } else {
        Err(format!("unknown estimation method: `{}`", token.trim()))
    }
}

fn parse_fit_options(lines: &[String]) -> Result<FitOptions, String> {
    let mut opts = FitOptions::default();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').map(|s| s.trim()).collect();
        if parts.len() != 2 {
            continue;
        }
        if parts[0] == "method" {
            let raw = parts[1].trim();
            // List form: `method = [a, b, c]` — chain of stages.
            if raw.starts_with('[') {
                let inner = raw.trim_start_matches('[').trim_end_matches(']');
                let chain: Vec<EstimationMethod> = inner
                    .split(',')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(parse_method_token)
                    .collect::<Result<_, _>>()?;
                if chain.is_empty() {
                    return Err("method = [] is empty; provide at least one method".into());
                }
                // Interaction flag follows the final stage of the chain.
                opts.interaction = *chain.last().unwrap() == EstimationMethod::FoceI;
                opts.method = *chain.last().unwrap();
                opts.methods = chain;
            } else {
                let m = parse_method_token(raw)?;
                opts.method = m;
                opts.methods.clear();
                if m == EstimationMethod::FoceI {
                    opts.interaction = true;
                }
            }
            opts.user_set_keys.push("method".to_string());
            continue;
        }
        // All other keys flow through the shared dispatch. Both `.ferx`
        // parsing and the R `settings` path are strict: unknown keys and
        // malformed values raise an error rather than silently defaulting.
        // A previous iteration of this parser used `.unwrap_or(default)` /
        // `== "true"` coercions that could silently flip behavior (e.g.
        // `covariance = TRUE` set `false`; `bloq_method = foo` landed on
        // the default `Drop`). Those traps are gone.
        match apply_fit_option(&mut opts, parts[0], parts[1]) {
            Ok(true) => {}
            Ok(false) => {
                return Err(format!("[fit_options]: unknown key `{}`", parts[0]));
            }
            Err(e) => return Err(format!("[fit_options]: {}", e)),
        }
    }
    Ok(opts)
}

/// Apply a single `key = value` pair to `FitOptions`.
///
/// Returns:
/// - `Ok(true)`  — key was recognized and applied.
/// - `Ok(false)` — key is not a known fit option.
/// - `Err(msg)`  — key is recognized but the value is malformed.
///
/// This is the single source of truth for the `[fit_options]` key grammar,
/// shared between `.ferx` parsing and the R wrapper's generic `settings`
/// list. Callers that want strict validation (e.g. the R wrapper) should
/// propagate `Err` and treat `Ok(false)` as "unknown setting".
///
/// Does NOT handle `method` (which has list-chain syntax) — that stays in
/// the block parser.
pub fn apply_fit_option(opts: &mut FitOptions, key: &str, value: &str) -> Result<bool, String> {
    let value = value.trim();

    let parse_usize = |name: &str| -> Result<usize, String> {
        value.parse::<usize>().map_err(|_| {
            format!("fit option `{name}`: expected non-negative integer, got `{value}`")
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
                format!("fit option `{name}`: expected non-negative integer, got `{value}`")
            })
        }
    };
    let parse_f64 = |name: &str| -> Result<f64, String> {
        value
            .parse::<f64>()
            .map_err(|_| format!("fit option `{name}`: expected number, got `{value}`"))
    };

    // Dispatch first, then record the key on success so we can later warn
    // when a key is set that the selected method does not consume. Malformed
    // values still return `Err` and don't get recorded.
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
        "steihaug_max_iters" => opts.steihaug_max_iters = parse_usize("steihaug_max_iters")?,
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
        "mu_referencing" => opts.mu_referencing = parse_bool("mu_referencing")?,
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
        "gradient" | "gradient_method" => {
            opts.gradient_method = match value.to_lowercase().as_str() {
                "auto" => GradientMethod::Auto,
                "ad" | "autodiff" => GradientMethod::Ad,
                "fd" | "finite" | "finite_difference" | "finite-difference" => GradientMethod::Fd,
                other => {
                    return Err(format!(
                        "fit option `gradient`: unknown value `{other}` — expected 'auto', 'ad', or 'fd'"
                    ));
                }
            };
        }
        "threads" => {
            if value.eq_ignore_ascii_case("auto") || value == "0" {
                opts.threads = None;
            } else {
                match value.parse::<usize>() {
                    Ok(n) if n > 0 => opts.threads = Some(n),
                    _ => {
                        return Err(format!(
                            "fit option `threads`: expected 'auto', 0, or a positive integer, got `{value}`"
                        ));
                    }
                }
            }
        }
        "optimizer_trace" => opts.optimizer_trace = parse_bool("optimizer_trace")?,
        "scale_params" => opts.scale_params = parse_bool("scale_params")?,
        _ => return Ok(false),
    }
    opts.user_set_keys.push(key.to_string());
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

    // For ODE RHS expressions, states + individual params get injected into the
    // `vars` map at eval time, so every bare identifier should resolve to a
    // Variable (not a Covariate). ParseCtx::ode() flips the fallback accordingly.
    let ode_defined: Vec<String> = state_names
        .iter()
        .cloned()
        .chain(indiv_param_names.iter().cloned())
        .collect();
    let ode_ctx = ParseCtx::ode(&ode_defined);

    for line in lines {
        if let Some(caps) = ddt_re.captures(line) {
            let state = caps[1].to_string();
            let expr_str = caps[2].trim();
            let expr = parse_expression(expr_str, ode_ctx)?;
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
    fixed: bool,
}

struct OmegaSpec {
    name: String,
    variance: f64,
    fixed: bool,
}

/// Specifies a block (correlated) group of omegas.
/// The values are the lower triangle of the covariance matrix, row-wise:
/// e.g. for 2x2: [var1, cov12, var2]; for 3x3: [var1, cov12, var2, cov13, cov23, var3]
struct BlockOmegaSpec {
    names: Vec<String>,
    lower_triangle: Vec<f64>,
    fixed: bool,
}

struct SigmaSpec {
    name: String,
    value: f64,
    fixed: bool,
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

    // theta NAME(init)  |  theta NAME(init, FIX)
    // theta NAME(init, lower, upper)  |  theta NAME(init, lower, upper, FIX)
    //
    // The `FIX` keyword is case-insensitive and must be the exact token —
    // the trailing `\b` rejects prefix matches like `FIXED`, which would
    // otherwise silently mark the parameter as fixed.
    let theta_re = Regex::new(
        r"(?i)theta\s+(\w+)\(\s*([0-9eE.+-]+)\s*(?:,\s*([0-9eE.+-]+)\s*,\s*([0-9eE.+-]+))?\s*(?:,\s*(FIX)\b)?\s*\)",
    )
    .unwrap();

    // omega NAME ~ value  |  omega NAME ~ value FIX
    let omega_re = Regex::new(r"(?i)omega\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap();

    // block_omega (NAME1, NAME2, ...) = [lower_triangle_values]  |  ... FIX
    let block_omega_re =
        Regex::new(r"(?i)block_omega\s*\(([^)]+)\)\s*=\s*\[([^\]]+)\](?:\s+(FIX)\b)?").unwrap();

    // sigma NAME ~ value  |  sigma NAME ~ value FIX
    let sigma_re = Regex::new(r"(?i)sigma\s+(\w+)\s*~\s*([0-9eE.+-]+)(?:\s+(FIX)\b)?").unwrap();

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
            let fixed = caps.get(5).is_some();
            thetas.push(ThetaSpec {
                name,
                init,
                lower,
                upper,
                fixed,
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
            let fixed = caps.get(3).is_some();
            block_omegas.push(BlockOmegaSpec {
                names,
                lower_triangle: values,
                fixed,
            });
        } else if let Some(caps) = omega_re.captures(line) {
            let name = caps[1].to_string();
            let variance: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad omega: {}", line))?;
            let fixed = caps.get(3).is_some();
            eta_names_ordered.push(name.clone());
            omegas.push(OmegaSpec {
                name,
                variance,
                fixed,
            });
        } else if let Some(caps) = sigma_re.captures(line) {
            let name = caps[1].to_string();
            let value: f64 = caps[2]
                .parse()
                .map_err(|_| format!("Bad sigma: {}", line))?;
            let fixed = caps.get(3).is_some();
            sigmas.push(SigmaSpec { name, value, fixed });
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

/// Build the per-eta `omega_fixed` flags from parsed diagonal + block specs.
///
/// Rules:
/// - `omega NAME ~ value FIX`: flag that eta as fixed.
/// - `block_omega (...) = [...] FIX`: flag every eta in the block.
/// - A diagonal omega FIX on an eta that is also listed in a (free) block is
///   rejected — you must fix the whole block instead.
fn build_omega_fixed(
    diag_omegas: &[OmegaSpec],
    block_omegas: &[BlockOmegaSpec],
    eta_names: &[String],
) -> Result<Vec<bool>, String> {
    let name_to_idx: std::collections::HashMap<&str, usize> = eta_names
        .iter()
        .enumerate()
        .map(|(i, n)| (n.as_str(), i))
        .collect();

    let mut fixed = vec![false; eta_names.len()];

    for spec in diag_omegas {
        if spec.fixed {
            if let Some(&idx) = name_to_idx.get(spec.name.as_str()) {
                fixed[idx] = true;
            }
        }
    }

    for block in block_omegas {
        for name in &block.names {
            let idx = *name_to_idx
                .get(name.as_str())
                .ok_or_else(|| format!("block_omega references unknown eta '{}'", name))?;
            // If the eta was already marked FIX via a diagonal spec but the
            // block is not fully fixed, that's ambiguous.
            if fixed[idx] && !block.fixed {
                return Err(format!(
                    "'{}' is marked FIX but belongs to a non-FIX block_omega; \
                     fix the whole block instead",
                    name
                ));
            }
            if block.fixed {
                fixed[idx] = true;
            }
        }
    }

    Ok(fixed)
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
) -> Result<(PkParamFn, Vec<String>), String> {
    let mut assignments: Vec<(String, Expression)> = Vec::new();

    let mut defined_vars: Vec<String> = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.splitn(2, '=').collect();
        if parts.len() != 2 {
            return Err(format!("Invalid individual parameter line: {}", line));
        }
        let var_name = parts[0].trim().to_string();
        let expr_str = parts[1].trim();
        let ctx = ParseCtx::new(theta_names, eta_names, &defined_vars);
        let expr = parse_expression(expr_str, ctx)?;
        assignments.push((var_name.clone(), expr));
        defined_vars.push(var_name);
    }

    // Covariate names referenced by any individual_parameters expression.
    // Sorted for deterministic error messages downstream.
    let mut cov_set: std::collections::HashSet<String> = std::collections::HashSet::new();
    for (_, expr) in &assignments {
        collect_covariates(expr, &mut cov_set);
    }
    let mut referenced_covariates: Vec<String> = cov_set.into_iter().collect();
    referenced_covariates.sort();

    let pk_map: HashMap<String, String> = pk_param_map.clone();
    let assignments_owned = assignments;

    let pk_param_fn: PkParamFn = Box::new(
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
    );
    Ok((pk_param_fn, referenced_covariates))
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

/// Context threaded through the recursive-descent parser so that every bare
/// identifier can be classified as Theta / Eta / Variable / Covariate without
/// relying on casing heuristics.
#[derive(Clone, Copy)]
struct ParseCtx<'a> {
    theta_names: &'a [String],
    eta_names: &'a [String],
    /// Names previously assigned in the surrounding block (e.g. earlier lines
    /// of [individual_parameters]). These resolve to `Variable`.
    defined_vars: &'a [String],
    /// When `true` (the usual case), an unknown identifier is a covariate.
    /// Set to `false` for the ODE RHS parser, where state names and individual
    /// parameters are injected into the `vars` map at eval time instead.
    fallback_covariate: bool,
}

impl<'a> ParseCtx<'a> {
    fn new(theta_names: &'a [String], eta_names: &'a [String], defined_vars: &'a [String]) -> Self {
        Self {
            theta_names,
            eta_names,
            defined_vars,
            fallback_covariate: true,
        }
    }

    fn ode(defined_vars: &'a [String]) -> Self {
        const EMPTY: &[String] = &[];
        Self {
            theta_names: EMPTY,
            eta_names: EMPTY,
            defined_vars,
            fallback_covariate: false,
        }
    }
}

fn parse_expression(s: &str, ctx: ParseCtx<'_>) -> Result<Expression, String> {
    let tokens = tokenize(s)?;
    let (expr, _) = parse_add_sub(&tokens, 0, ctx)?;
    Ok(expr)
}

/// Walk an expression tree and accumulate every covariate name it references.
fn collect_covariates(expr: &Expression, out: &mut std::collections::HashSet<String>) {
    match expr {
        Expression::Covariate(name) => {
            out.insert(name.clone());
        }
        Expression::BinOp(lhs, _, rhs) => {
            collect_covariates(lhs, out);
            collect_covariates(rhs, out);
        }
        Expression::UnaryFn(_, arg) => collect_covariates(arg, out),
        Expression::Power(base, exp) => {
            collect_covariates(base, out);
            collect_covariates(exp, out);
        }
        _ => {}
    }
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
        Expression::Covariate(name) => covariates.get(name).copied().unwrap_or(0.0),
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
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_mul_div(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Plus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Add, Box::new(right));
                pos = p;
            }
            Token::Minus => {
                let (right, p) = parse_mul_div(tokens, pos + 1, ctx)?;
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
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (mut left, mut pos) = parse_power(tokens, pos, ctx)?;

    while pos < tokens.len() {
        match &tokens[pos] {
            Token::Star => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
                left = Expression::BinOp(Box::new(left), BinOp::Mul, Box::new(right));
                pos = p;
            }
            Token::Slash => {
                let (right, p) = parse_power(tokens, pos + 1, ctx)?;
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
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    let (base, mut pos) = parse_atom(tokens, pos, ctx)?;

    if pos < tokens.len() && tokens[pos] == Token::Caret {
        let (exp, p) = parse_atom(tokens, pos + 1, ctx)?;
        pos = p;
        return Ok((Expression::Power(Box::new(base), Box::new(exp)), pos));
    }

    Ok((base, pos))
}

fn parse_atom(
    tokens: &[Token],
    pos: usize,
    ctx: ParseCtx<'_>,
) -> Result<(Expression, usize), String> {
    if pos >= tokens.len() {
        return Err("Unexpected end of expression".to_string());
    }

    match &tokens[pos] {
        Token::Minus => {
            // Unary minus: -expr → 0 - expr
            let (expr, p) = parse_atom(tokens, pos + 1, ctx)?;
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
            let (expr, p) = parse_add_sub(tokens, pos + 1, ctx)?;
            if p >= tokens.len() || tokens[p] != Token::RParen {
                return Err("Missing closing parenthesis".to_string());
            }
            Ok((expr, p + 1))
        }
        Token::Ident(name) => {
            // Check if it's a function call: name(expr)
            if pos + 1 < tokens.len() && tokens[pos + 1] == Token::LParen {
                let func_name = name.to_lowercase();
                let (arg, p) = parse_add_sub(tokens, pos + 2, ctx)?;
                if p >= tokens.len() || tokens[p] != Token::RParen {
                    return Err(format!("Missing closing parenthesis for function {}", name));
                }
                return Ok((Expression::UnaryFn(func_name, Box::new(arg)), p + 1));
            }

            // Check if it's a theta
            if let Some(idx) = ctx.theta_names.iter().position(|n| n == name) {
                return Ok((Expression::Theta(idx), pos + 1));
            }

            // Check if it's an eta
            if let Some(idx) = ctx.eta_names.iter().position(|n| n == name) {
                return Ok((Expression::Eta(idx), pos + 1));
            }

            // Previously-assigned local variable (e.g. earlier lines of
            // [individual_parameters], or a state/param name injected by the
            // ODE RHS harness).
            if ctx.defined_vars.iter().any(|n| n == name) {
                return Ok((Expression::Variable(name.clone()), pos + 1));
            }

            // Anything else is a covariate reference in the regular model
            // context. The ODE RHS context keeps it as a Variable so that the
            // eval-time `vars` map (which carries state + individual params)
            // can resolve it case-sensitively.
            if ctx.fallback_covariate {
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
    fn test_parse_method_single() {
        let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
        assert_eq!(opts.method, EstimationMethod::FoceI);
        assert!(opts.methods.is_empty());
        assert!(opts.interaction);
    }

    #[test]
    fn test_parse_method_chain() {
        let opts = parse_fit_options(&["method = [saem, focei]".to_string()]).unwrap();
        assert_eq!(
            opts.methods,
            vec![EstimationMethod::Saem, EstimationMethod::FoceI]
        );
        assert_eq!(opts.method, EstimationMethod::FoceI);
        assert!(opts.interaction);
    }

    #[test]
    fn test_parse_method_chain_final_foce() {
        let opts = parse_fit_options(&["method = [saem, foce]".to_string()]).unwrap();
        assert_eq!(opts.method, EstimationMethod::Foce);
        assert!(!opts.interaction);
    }

    #[test]
    fn test_parse_method_chain_empty_rejected() {
        assert!(parse_fit_options(&["method = []".to_string()]).is_err());
    }

    #[test]
    fn test_parse_method_unknown_rejected() {
        assert!(parse_fit_options(&["method = [foce, wibble]".to_string()]).is_err());
    }

    #[test]
    fn test_method_chain_helper_default() {
        let opts = FitOptions::default();
        assert_eq!(opts.method_chain(), vec![EstimationMethod::Foce]);
    }

    #[test]
    fn test_method_chain_helper_populated() {
        let mut opts = FitOptions::default();
        opts.methods = vec![EstimationMethod::Saem, EstimationMethod::FoceI];
        assert_eq!(
            opts.method_chain(),
            vec![EstimationMethod::Saem, EstimationMethod::FoceI]
        );
    }

    #[test]
    fn test_parse_threads_positive() {
        let opts = parse_fit_options(&["threads = 4".to_string()]).unwrap();
        assert_eq!(opts.threads, Some(4));
    }

    #[test]
    fn test_parse_threads_auto() {
        let opts = parse_fit_options(&["threads = auto".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
        // Case-insensitive.
        let opts = parse_fit_options(&["threads = AUTO".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_parse_threads_zero_means_auto() {
        // `threads = 0` is treated as "leave rayon default alone",
        // matching the R binding's `threads <= 0` sentinel.
        let opts = parse_fit_options(&["threads = 0".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_parse_threads_invalid_errors() {
        // Strict parsing: malformed threads values raise a parse error
        // rather than silently falling back to `None` (the pre-refactor
        // `.parse().ok().filter(...)` behavior was a typo trap).
        assert!(parse_fit_options(&["threads = -1".to_string()]).is_err());
        assert!(parse_fit_options(&["threads = wibble".to_string()]).is_err());
    }

    #[test]
    fn test_parse_threads_default_is_none() {
        // No `threads` line → None (rayon global pool, one worker per logical CPU).
        let opts = parse_fit_options(&["method = focei".to_string()]).unwrap();
        assert_eq!(opts.threads, None);
    }

    // ── mu_referencing fit option ────────────────────────────────────────

    #[test]
    fn test_parse_mu_referencing_default_true() {
        let opts = parse_fit_options(&["method = foce".to_string()]).unwrap();
        assert!(opts.mu_referencing);
    }

    #[test]
    fn test_parse_mu_referencing_false() {
        let opts = parse_fit_options(&["mu_referencing = false".to_string()]).unwrap();
        assert!(!opts.mu_referencing);
    }

    #[test]
    fn test_parse_mu_referencing_accepts_synonyms() {
        for raw in &["true", "TRUE", "1", "yes", "on"] {
            let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
            assert!(opts.mu_referencing, "{} should enable", raw);
        }
        for raw in &["false", "FALSE", "0", "no", "off"] {
            let opts = parse_fit_options(&[format!("mu_referencing = {}", raw)]).unwrap();
            assert!(!opts.mu_referencing, "{} should disable", raw);
        }
    }

    #[test]
    fn test_parse_mu_referencing_invalid_rejected() {
        assert!(parse_fit_options(&["mu_referencing = wibble".to_string()]).is_err());
    }

    // ── apply_fit_option (shared dispatch used by the R wrapper's `settings`
    //    argument and by parse_fit_options) ────────────────────────────────

    #[test]
    fn test_apply_fit_option_known_applies() {
        let mut opts = FitOptions::default();
        assert_eq!(
            apply_fit_option(&mut opts, "n_exploration", "200"),
            Ok(true)
        );
        assert_eq!(opts.saem_n_exploration, 200);

        assert_eq!(
            apply_fit_option(&mut opts, "n_convergence", "400"),
            Ok(true)
        );
        assert_eq!(opts.saem_n_convergence, 400);
    }

    #[test]
    fn test_apply_fit_option_unknown_key_returns_false() {
        let mut opts = FitOptions::default();
        // Typo / unknown → Ok(false). Caller decides whether to error out.
        assert_eq!(
            apply_fit_option(&mut opts, "n_exploraton", "200"),
            Ok(false)
        );
        // `method` is deliberately excluded (list-chain syntax is handled
        // in the block parser); treat it as unknown here.
        assert_eq!(apply_fit_option(&mut opts, "method", "focei"), Ok(false));
    }

    #[test]
    fn test_apply_fit_option_malformed_value_errors() {
        let mut opts = FitOptions::default();
        assert!(apply_fit_option(&mut opts, "n_exploration", "oops").is_err());
        assert!(apply_fit_option(&mut opts, "covariance", "maybe").is_err());
        assert!(apply_fit_option(&mut opts, "gn_lambda", "x").is_err());
        assert!(apply_fit_option(&mut opts, "optimizer", "does_not_exist").is_err());
        assert!(apply_fit_option(&mut opts, "bloq_method", "nope").is_err());
        assert!(apply_fit_option(&mut opts, "threads", "-1").is_err());
        // Failed apply must not mutate — default preserved.
        assert_eq!(opts.saem_n_exploration, 150);
    }

    #[test]
    fn test_apply_fit_option_bool_variants() {
        let mut opts = FitOptions::default();
        for v in ["true", "True", "TRUE", "yes", "1", "t"] {
            opts.sir = false;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(opts.sir, "value `{v}` should parse as true");
        }
        for v in ["false", "False", "no", "0", "f"] {
            opts.sir = true;
            assert_eq!(apply_fit_option(&mut opts, "sir", v), Ok(true));
            assert!(!opts.sir, "value `{v}` should parse as false");
        }
    }

    #[test]
    fn test_apply_fit_option_seed_null_clears() {
        let mut opts = FitOptions::default();
        opts.saem_seed = Some(7);
        // R sends NULL/NA through as the literal "null" / "na".
        assert_eq!(apply_fit_option(&mut opts, "seed", "null"), Ok(true));
        assert_eq!(opts.saem_seed, None);

        assert_eq!(apply_fit_option(&mut opts, "seed", "42"), Ok(true));
        assert_eq!(opts.saem_seed, Some(42));

        // `saem_seed` is accepted as an alias so R users can use either spelling.
        assert_eq!(apply_fit_option(&mut opts, "saem_seed", "99"), Ok(true));
        assert_eq!(opts.saem_seed, Some(99));
    }

    #[test]
    fn test_apply_fit_option_threads_variants() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "threads", "4"), Ok(true));
        assert_eq!(opts.threads, Some(4));

        assert_eq!(apply_fit_option(&mut opts, "threads", "auto"), Ok(true));
        assert_eq!(opts.threads, None);

        opts.threads = Some(4);
        assert_eq!(apply_fit_option(&mut opts, "threads", "0"), Ok(true));
        assert_eq!(opts.threads, None);
    }

    #[test]
    fn test_apply_fit_option_optimizer_and_bloq() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "optimizer", "lbfgs"), Ok(true));
        assert_eq!(opts.optimizer, Optimizer::NloptLbfgs);

        assert_eq!(apply_fit_option(&mut opts, "bloq", "m3"), Ok(true));
        assert_eq!(opts.bloq_method, BloqMethod::M3);
    }

    // ── Warn on options that don't apply to the selected estimation method.
    //    These fire from inside fit() via FitOptions::unsupported_keys_warnings,
    //    so we check the raw mechanism here without running a full fit. ────

    #[test]
    fn test_unsupported_saem_key_under_focei_warns() {
        let opts = parse_fit_options(&[
            "method = focei".to_string(),
            "n_convergence = 300".to_string(),
        ])
        .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        let w = &warnings[0];
        assert!(w.contains("n_convergence"), "got: {w}");
        assert!(w.contains("FOCEI"), "got: {w}");
        assert!(w.contains("will be ignored"), "got: {w}");
        // Mentions a FOCE-applicable key so the user can see what's available.
        assert!(w.contains("optimizer"), "got: {w}");
        // Does NOT suggest SAEM-specific keys as available.
        assert!(!w.contains("n_mh_steps"), "got: {w}");
    }

    #[test]
    fn test_unsupported_focei_key_under_saem_warns() {
        let opts =
            parse_fit_options(&["method = saem".to_string(), "optimizer = lbfgs".to_string()])
                .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        let w = &warnings[0];
        assert!(w.contains("optimizer"), "got: {w}");
        assert!(w.contains("SAEM"), "got: {w}");
        assert!(w.contains("n_exploration"), "got: {w}");
    }

    #[test]
    fn test_applicable_key_in_chain_no_warning() {
        // methods = [saem, focei]: n_convergence applies to SAEM, optimizer
        // applies to FOCEI, so neither should warn.
        let opts = parse_fit_options(&[
            "method = [saem, focei]".to_string(),
            "n_convergence = 300".to_string(),
            "optimizer = lbfgs".to_string(),
        ])
        .unwrap();
        assert!(opts.unsupported_keys_warnings().is_empty());
    }

    #[test]
    fn test_common_keys_never_warn() {
        // Covariance/verbose/sir/bloq/threads/mu_referencing apply to every
        // method — they must not produce a warning regardless of method.
        for method in ["foce", "focei", "gn", "gn_hybrid", "saem"] {
            let opts = parse_fit_options(&[
                format!("method = {method}"),
                "covariance = false".to_string(),
                "verbose = false".to_string(),
                "sir = true".to_string(),
                "bloq_method = m3".to_string(),
                "threads = 2".to_string(),
                "mu_referencing = false".to_string(),
            ])
            .unwrap();
            let w = opts.unsupported_keys_warnings();
            assert!(
                w.is_empty(),
                "method={method} produced unexpected warnings: {:?}",
                w
            );
        }
    }

    #[test]
    fn test_unsupported_warning_omits_framework_keys() {
        // Framework-wide keys (covariance/verbose/sir/bloq/threads/mu_referencing)
        // are exposed as top-level wrapper args, not as method-specific settings.
        // The warning's "Method-specific options" list must not include them —
        // listing `covariance` next to `optimizer` would conflate the layers.
        let opts = parse_fit_options(&[
            "method = focei".to_string(),
            "n_convergence = 300".to_string(),
        ])
        .unwrap();
        let w = &opts.unsupported_keys_warnings()[0];
        for framework in [
            "covariance",
            "verbose",
            "sir",
            "sir_samples",
            "sir_resamples",
            "sir_seed",
            "bloq_method",
            "bloq",
            "threads",
            "mu_referencing",
        ] {
            assert!(
                !w.contains(framework),
                "framework key `{framework}` leaked into method-specific list: {w}"
            );
        }
        // And it uses the new phrasing, not the old "Available options".
        assert!(w.contains("Method-specific options"), "got: {w}");
    }

    #[test]
    fn test_gn_lambda_under_focei_warns() {
        let opts =
            parse_fit_options(&["method = focei".to_string(), "gn_lambda = 0.05".to_string()])
                .unwrap();
        let warnings = opts.unsupported_keys_warnings();
        assert_eq!(warnings.len(), 1, "got: {:?}", warnings);
        assert!(warnings[0].contains("gn_lambda"));
    }

    #[test]
    fn test_no_warning_when_no_keys_set() {
        // Bare default FitOptions (no parser path) must not conjure warnings.
        let opts = FitOptions::default();
        assert!(opts.unsupported_keys_warnings().is_empty());
    }

    // ── parse_fit_options: strict parsing at the .ferx layer. Unknown
    //    keys and malformed values both raise an error — a typo like
    //    `covariance = maybe` or `bloq_method = nope` now fails loudly
    //    instead of silently landing on an unexpected default. ───────────

    #[test]
    fn test_parse_fit_options_unknown_key_errors() {
        let err = parse_fit_options(&["n_exploraton = 200".to_string()]).unwrap_err();
        assert!(err.contains("unknown key"), "got: {err}");
        assert!(err.contains("n_exploraton"), "got: {err}");
    }

    #[test]
    fn test_parse_fit_options_malformed_numeric_errors() {
        assert!(parse_fit_options(&["n_exploration = oops".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_malformed_bool_errors() {
        // Pre-refactor, `covariance = maybe` silently coerced to `false`
        // via `== "true"`, flipping the default. Now it errors.
        assert!(parse_fit_options(&["covariance = maybe".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_uppercase_bool_accepted() {
        // Pre-refactor, `covariance = TRUE` silently became `false`
        // because the inline check only matched lowercase "true". The
        // strict parser accepts common casing variants.
        let opts = parse_fit_options(&["covariance = TRUE".to_string()]).unwrap();
        assert!(opts.run_covariance_step);
    }

    #[test]
    fn test_parse_fit_options_bloq_method_typo_errors() {
        // `bloq_method` was already strict in the old inline parser; the
        // new strict dispatch must preserve that (not silently default).
        assert!(parse_fit_options(&["bloq_method = nope".to_string()]).is_err());
    }

    #[test]
    fn test_parse_fit_options_gradient_method() {
        // Accepted aliases resolve to the expected GradientMethod variant.
        for (input, expected) in [
            ("gradient = auto", GradientMethod::Auto),
            ("gradient = ad", GradientMethod::Ad),
            ("gradient = autodiff", GradientMethod::Ad),
            ("gradient = fd", GradientMethod::Fd),
            ("gradient = finite", GradientMethod::Fd),
            ("gradient_method = ad", GradientMethod::Ad),
        ] {
            let opts = parse_fit_options(&[input.to_string()]).unwrap();
            assert_eq!(opts.gradient_method, expected, "input: {input}");
        }

        // Unknown values must fail loudly — silently defaulting would hide
        // typos like `gradient = auo` that a user probably intended as `auto`.
        assert!(parse_fit_options(&["gradient = nope".to_string()]).is_err());
    }

    #[test]
    fn test_parse_all_example_ferx_files() {
        // Smoke test: every checked-in example must parse under the strict
        // [fit_options] rules. Guards against accidentally tightening a key
        // in apply_fit_option in a way that breaks a shipped example.
        let examples_dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("examples");
        let mut seen = 0;
        for entry in std::fs::read_dir(&examples_dir).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().and_then(|s| s.to_str()) != Some("ferx") {
                continue;
            }
            seen += 1;
            if let Err(e) = parse_full_model_file(&path) {
                panic!("failed to parse {}: {}", path.display(), e);
            }
        }
        assert!(
            seen > 0,
            "no .ferx files found in {}",
            examples_dir.display()
        );
    }

    #[test]
    fn test_parse_fit_options_applies_known_keys() {
        let lines = vec![
            "method = saem".to_string(),
            "n_exploration = 200".to_string(),
            "n_convergence = 400".to_string(),
            "sir = true".to_string(),
            "sir_samples = 2000".to_string(),
        ];
        let opts = parse_fit_options(&lines).unwrap();
        assert_eq!(opts.method, EstimationMethod::Saem);
        assert_eq!(opts.saem_n_exploration, 200);
        assert_eq!(opts.saem_n_convergence, 400);
        assert!(opts.sir);
        assert_eq!(opts.sir_samples, 2000);
    }

    // ── mu-referencing pattern detection ─────────────────────────────────

    fn detect_one(line: &str, theta_names: &[&str], eta_names: &[&str]) -> Option<MuRef> {
        let tn: Vec<String> = theta_names.iter().map(|s| s.to_string()).collect();
        let en: Vec<String> = eta_names.iter().map(|s| s.to_string()).collect();
        let refs = detect_mu_refs(&[line.to_string()], &tn, &en);
        // Return the one detected mu-ref (if any). Tests assume a single line.
        refs.into_iter().next().map(|(_, v)| v)
    }

    #[test]
    fn test_detect_mu_ref_multiplicative_exp() {
        // Classic NONMEM pattern: CL = TVCL * exp(ETA_CL)
        let m = detect_one("CL = TVCL * exp(ETA_CL)", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_exp_of_log_sum() {
        // Canonical mu-reference form: exp(log(THETA) + ETA)
        let m = detect_one("CL = exp(log(TVCL) + ETA_CL)", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_exp_of_log_sum_reversed() {
        // ETA on the left: exp(ETA + log(THETA))
        let m = detect_one("CL = exp(ETA_CL + log(TVCL))", &["TVCL"], &["ETA_CL"])
            .expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_additive() {
        // Additive eta: CL = TVCL + ETA_CL → mu = TVCL (not log-transformed)
        let m =
            detect_one("CL = TVCL + ETA_CL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(!m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_additive_reversed() {
        // ETA first: CL = ETA_CL + TVCL
        let m =
            detect_one("CL = ETA_CL + TVCL", &["TVCL"], &["ETA_CL"]).expect("should detect mu-ref");
        assert_eq!(m.theta_name, "TVCL");
        assert!(!m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_product_chain_with_covariate() {
        // Real covariate model: CL = TVCL * (WT/70)^0.75 * exp(ETA_CL).
        // The detector walks the Mul chain for the anchor theta and the
        // exp(eta) factor; the Power sub-expression is opaque (neither a
        // Theta nor an exp(Eta)), so it is simply skipped. As long as there
        // is exactly one bare Theta factor, detection still succeeds.
        let m = detect_one(
            "CL = TVCL * (WT/70)^0.75 * exp(ETA_CL)",
            &["TVCL"],
            &["ETA_CL"],
        )
        .expect("should still detect mu-ref through opaque covariate term");
        assert_eq!(m.theta_name, "TVCL");
        assert!(m.log_transformed);
    }

    #[test]
    fn test_detect_mu_ref_rejects_two_thetas() {
        // Two thetas in the product → ambiguous anchor, pattern rejected.
        let m = detect_one(
            "CL = TVCL * TVCL2 * exp(ETA_CL)",
            &["TVCL", "TVCL2"],
            &["ETA_CL"],
        );
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_rejects_constant_only() {
        // No theta in the product → not a mu-ref.
        let m = detect_one("CL = 2.0 * exp(ETA_CL)", &["TVCL"], &["ETA_CL"]);
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_rejects_compound_eta_expression() {
        // exp(ETA_CL + ETA_OCC) is not a bare exp(Eta) — rejected.
        let m = detect_one(
            "CL = TVCL * exp(ETA_CL + ETA_OCC)",
            &["TVCL"],
            &["ETA_CL", "ETA_OCC"],
        );
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_rejects_no_eta() {
        // KM = TVKM — no eta, no mu-ref recorded.
        let m = detect_one("KM = TVKM", &["TVKM"], &[]);
        assert!(m.is_none());
    }

    #[test]
    fn test_detect_mu_ref_multiple_parameters() {
        // Detect across several lines; each eta maps to its own theta.
        let lines = vec![
            "CL = TVCL * exp(ETA_CL)".to_string(),
            "V  = TVV  * exp(ETA_V)".to_string(),
            "KA = TVKA * exp(ETA_KA)".to_string(),
        ];
        let tn = vec!["TVCL".to_string(), "TVV".to_string(), "TVKA".to_string()];
        let en = vec![
            "ETA_CL".to_string(),
            "ETA_V".to_string(),
            "ETA_KA".to_string(),
        ];
        let refs = detect_mu_refs(&lines, &tn, &en);
        assert_eq!(refs.len(), 3);
        assert_eq!(refs["ETA_CL"].theta_name, "TVCL");
        assert_eq!(refs["ETA_V"].theta_name, "TVV");
        assert_eq!(refs["ETA_KA"].theta_name, "TVKA");
        assert!(refs.values().all(|m| m.log_transformed));
    }

    #[test]
    fn test_detect_mu_ref_full_model_parse() {
        // End-to-end: parse a minimal .ferx and verify mu_refs is populated.
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, 0.1, 500.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04

  sigma PROP_ERR ~ 0.02

[individual_parameters]
  CL = TVCL * exp(ETA_CL)
  V  = TVV  * exp(ETA_V)

[structural_model]
  pk one_cpt_iv_bolus(cl=CL, v=V)

[error_model]
  DV ~ proportional(PROP_ERR)
"#;
        let parsed = parse_full_model(content).expect("model should parse");
        assert_eq!(parsed.model.mu_refs.len(), 2);
        let cl = parsed.model.mu_refs.get("ETA_CL").unwrap();
        assert_eq!(cl.theta_name, "TVCL");
        assert!(cl.log_transformed);
    }

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
                fixed: false,
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
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
            fixed: false,
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
            fixed: false,
        }];
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: false,
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

    // ── FIX keyword ────────────────────────────────────────────────────────

    #[test]
    fn test_parse_theta_fix_without_bounds() {
        let lines = vec!["theta TVCL(0.1, FIX)".to_string()];
        let (thetas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert_eq!(thetas.len(), 1);
        assert!(thetas[0].fixed);
        assert!((thetas[0].init - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_fix_with_bounds() {
        let lines = vec!["theta TVCL(0.1, 0.01, 1.0, FIX)".to_string()];
        let (thetas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(thetas[0].fixed);
        assert!((thetas[0].lower - 0.01).abs() < 1e-12);
        assert!((thetas[0].upper - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_parse_theta_unfixed_by_default() {
        let lines = vec!["theta TVCL(0.1, 0.01, 1.0)".to_string()];
        let (thetas, _, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(!thetas[0].fixed);
    }

    #[test]
    fn test_parse_omega_fix() {
        let lines = vec!["omega ETA_CL ~ 0.09 FIX".to_string()];
        let (_, omegas, _, _, _) = parse_parameters(&lines).unwrap();
        assert!(omegas[0].fixed);
    }

    #[test]
    fn test_parse_sigma_fix() {
        let lines = vec!["sigma PROP ~ 0.05 FIX".to_string()];
        let (_, _, _, sigmas, _) = parse_parameters(&lines).unwrap();
        assert!(sigmas[0].fixed);
    }

    #[test]
    fn test_parse_block_omega_fix() {
        let lines = vec!["block_omega (ETA_CL, ETA_V) = [0.09, 0.02, 0.04] FIX".to_string()];
        let (_, _, blocks, _, _) = parse_parameters(&lines).unwrap();
        assert!(blocks[0].fixed);
    }

    #[test]
    fn test_fix_keyword_case_insensitive() {
        let lines = vec![
            "theta TVCL(0.1, fix)".to_string(),
            "omega ETA ~ 0.05 Fix".to_string(),
            "sigma S ~ 0.02 FIX".to_string(),
        ];
        let (thetas, omegas, _, sigmas, _) = parse_parameters(&lines).unwrap();
        assert!(thetas[0].fixed);
        assert!(omegas[0].fixed);
        assert!(sigmas[0].fixed);
    }

    #[test]
    fn test_fix_keyword_rejects_prefix_match() {
        // `FIXED` must not be silently accepted as `FIX`. Any non-exact token
        // should leave the parameter as free (or fail to parse the line),
        // never flip `fixed = true`.
        let lines = vec![
            "omega ETA_CL ~ 0.09 FIXED".to_string(),
            "sigma PROP ~ 0.02 FIXED".to_string(),
            "block_omega (A, B) = [1.0, 0.0, 1.0] FIXED".to_string(),
        ];
        let (_, omegas, blocks, sigmas, _) = parse_parameters(&lines).unwrap();
        // omega/sigma still parse (trailing `FIXED` is ignored) but must NOT
        // be marked fixed.
        assert!(!omegas[0].fixed);
        assert!(!sigmas[0].fixed);
        assert!(!blocks[0].fixed);
    }

    #[test]
    fn test_build_omega_fixed_diagonal() {
        let diag = vec![
            OmegaSpec {
                name: "ETA_CL".into(),
                variance: 0.09,
                fixed: true,
            },
            OmegaSpec {
                name: "ETA_V".into(),
                variance: 0.04,
                fixed: false,
            },
        ];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let flags = build_omega_fixed(&diag, &[], &names).unwrap();
        assert_eq!(flags, vec![true, false]);
    }

    #[test]
    fn test_build_omega_fixed_block() {
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: true,
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let flags = build_omega_fixed(&[], &block, &names).unwrap();
        assert_eq!(flags, vec![true, true]);
    }

    #[test]
    fn test_build_omega_fixed_rejects_diag_fix_inside_free_block() {
        // ETA_CL is in a non-FIX block but also declared FIX as a diagonal —
        // the parser must reject this as ambiguous.
        let diag = vec![OmegaSpec {
            name: "ETA_CL".into(),
            variance: 0.09,
            fixed: true,
        }];
        let block = vec![BlockOmegaSpec {
            names: vec!["ETA_CL".into(), "ETA_V".into()],
            lower_triangle: vec![0.09, 0.02, 0.04],
            fixed: false,
        }];
        let names = vec!["ETA_CL".into(), "ETA_V".into()];
        let res = build_omega_fixed(&diag, &block, &names);
        assert!(res.is_err());
    }

    #[test]
    fn test_parse_full_model_with_fix() {
        let content = r#"
[parameters]
  theta TVCL(0.2, 0.001, 10.0)
  theta TVV(10.0, FIX)
  theta TVKA(1.5, 0.01, 50.0)

  omega ETA_CL ~ 0.09
  omega ETA_V  ~ 0.04 FIX
  omega ETA_KA ~ 0.30

  sigma PROP_ERR ~ 0.02 FIX

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
        let p = &parsed.model.default_params;
        assert_eq!(p.theta_fixed, vec![false, true, false]);
        assert_eq!(p.omega_fixed, vec![false, true, false]);
        assert_eq!(p.sigma_fixed, vec![true]);
        assert!(p.has_any_fixed());
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
    fn test_parse_optimizer_defaults_to_bobyqa() {
        // No [fit_options] block → default optimizer.
        let content = minimal_model_with_fit_options("  maxiter = 100");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.optimizer, Optimizer::Bobyqa);
    }

    #[test]
    fn test_parse_steihaug_max_iters() {
        let content =
            minimal_model_with_fit_options("  optimizer = trust_region\n  steihaug_max_iters = 30");
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
        let content = minimal_model_with_fit_options("  inner_maxiter = 75\n  inner_tol = 1e-5");
        let parsed = parse_full_model(&content).unwrap();
        assert_eq!(parsed.fit_options.inner_maxiter, 75);
        assert!((parsed.fit_options.inner_tol - 1e-5).abs() < 1e-15);
    }

    #[test]
    fn test_fit_options_defaults() {
        // Guard against accidental drift in defaults — documented as:
        //   optimizer = bobyqa, inner_maxiter = 200, inner_tol = 1e-8,
        //   steihaug_max_iters = 50.
        let opts = FitOptions::default();
        assert_eq!(opts.optimizer, Optimizer::Bobyqa);
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

    // ── apply_fit_option: coverage of the newly-added optimizer keys.
    //    The generic apply_fit_option tests (known/unknown/malformed/bool
    //    variants/threads/seed) live in the earlier test block — these
    //    only add the keys that are new on this branch.

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
        // Reject malformed (e.g. negative) value.
        assert!(apply_fit_option(&mut opts, "steihaug_max_iters", "-1").is_err());
    }

    #[test]
    fn test_apply_fit_option_inner_maxiter_and_tol() {
        let mut opts = FitOptions::default();
        assert_eq!(apply_fit_option(&mut opts, "inner_maxiter", "75"), Ok(true));
        assert_eq!(opts.inner_maxiter, 75);

        assert_eq!(apply_fit_option(&mut opts, "inner_tol", "1e-5"), Ok(true));
        assert!((opts.inner_tol - 1e-5).abs() < 1e-15);

        assert!(apply_fit_option(&mut opts, "inner_maxiter", "oops").is_err());
        assert!(apply_fit_option(&mut opts, "inner_tol", "not_a_num").is_err());
    }
}
