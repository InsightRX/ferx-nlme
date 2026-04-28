use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::saem;
use crate::io::datareader::read_nonmem_csv;
use crate::io::output;
use crate::pk;
use crate::stats::likelihood::{
    compute_cwres, foce_subject_nll, foce_subject_nll_iov, split_obs_by_occasion,
};
use crate::stats::residual_error::compute_iwres;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::path::Path;

/// Route predictions through analytical PK or ODE solver.
fn model_preds(model: &CompiledModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    if let Some(ref ode_spec) = model.ode_spec {
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values)
    } else {
        pk::compute_predictions(model.pk_model, subject, pk_params)
    }
}

/// Run a model file with a NONMEM-format CSV dataset.
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_with_data(
    model_path: &str,
    data_path: &str,
) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    let iov_col = parsed.fit_options.iov_column.as_deref();
    let population = read_nonmem_csv(Path::new(data_path), None, iov_col)?;
    eprintln!(
        "Data:  {} subjects, {} observations from {}",
        population.subjects.len(),
        population.n_obs(),
        data_path
    );

    let init_params = build_init_params(&parsed);
    let result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    Ok((result, population))
}

/// Run a model file with simulated data (from [simulation] block).
/// Returns (FitResult, Population) so caller can write sdtab.
pub fn run_model_simulate(model_path: &str) -> Result<(FitResult, Population), String> {
    use crate::parser::model_parser::parse_full_model_file;
    use std::collections::HashMap;

    let mut parsed = parse_full_model_file(Path::new(model_path))?;
    let sim_spec = parsed
        .simulation
        .clone()
        .ok_or("Model file has no [simulation] block — use --data instead")?;
    set_model_name(&mut parsed.model, model_path);

    eprintln!("Model: {}", parsed.model.name);

    // Build template population
    let subjects: Vec<Subject> = (1..=sim_spec.n_subjects)
        .map(|i| Subject {
            id: format!("{}", i),
            doses: vec![DoseEvent::new(
                0.0,
                sim_spec.dose_amt,
                sim_spec.dose_cmt,
                0.0,
                false,
                0.0,
            )],
            obs_times: sim_spec.obs_times.clone(),
            observations: vec![0.0; sim_spec.obs_times.len()],
            obs_cmts: vec![1; sim_spec.obs_times.len()],
            covariates: HashMap::new(),
            tvcov: HashMap::new(),
            cens: vec![0; sim_spec.obs_times.len()],
            occasions: Vec::new(),
            dose_occasions: Vec::new(),
        })
        .collect();
    let template = Population {
        subjects,
        covariate_names: vec![],
        dv_column: "dv".into(),
    };

    // Simulate
    eprintln!(
        "Simulating {} subjects (seed={})...",
        sim_spec.n_subjects, sim_spec.seed
    );
    let sim_results = simulate_with_seed(
        &parsed.model,
        &template,
        &parsed.model.default_params,
        1,
        sim_spec.seed,
    );

    let mut population = template;
    for subject in &mut population.subjects {
        let sims: Vec<_> = sim_results.iter().filter(|s| s.id == subject.id).collect();
        for (j, s) in sims.iter().enumerate() {
            if j < subject.observations.len() {
                subject.observations[j] = s.dv_sim.max(0.001);
            }
        }
    }

    eprintln!(
        "Loaded {} subjects, {} observations",
        population.subjects.len(),
        population.n_obs()
    );

    let init_params = build_init_params(&parsed);
    let result = fit(
        &parsed.model,
        &population,
        &init_params,
        &parsed.fit_options,
    )?;
    Ok((result, population))
}

/// Legacy alias
pub fn run_from_file(path: &str) -> Result<FitResult, String> {
    run_model_simulate(path).map(|(r, _)| r)
}

fn set_model_name(model: &mut CompiledModel, path: &str) {
    if model.name == "Unnamed" {
        if let Some(stem) = Path::new(path).file_stem().and_then(|s| s.to_str()) {
            model.name = stem.to_string();
        }
    }
}

fn build_init_params(parsed: &ParsedModel) -> ModelParameters {
    parsed.model.default_params.clone()
}

/// Fail early if the model references covariates that the data doesn't carry.
/// Case-sensitive: `CRCL` and `crcl` are distinct names. Historically a missing
/// covariate silently evaluated to zero, which left fits stuck at the initial
/// estimates with no visible diagnostic (see commit introducing this check).
fn validate_covariates(model: &CompiledModel, population: &Population) -> Result<(), String> {
    let missing: Vec<&str> = model
        .referenced_covariates
        .iter()
        .filter(|name| !population.covariate_names.iter().any(|n| n == *name))
        .map(|s| s.as_str())
        .collect();

    if missing.is_empty() {
        return Ok(());
    }

    let available = if population.covariate_names.is_empty() {
        "(none)".to_string()
    } else {
        population.covariate_names.join(", ")
    };
    Err(format!(
        "Model references covariate(s) not found in data (case-sensitive): {}. \
         Available covariate columns: {}.",
        missing.join(", "),
        available
    ))
}

/// High-level fit: model file path + data file path → FitResult
pub fn fit_from_files(
    model_path: &str,
    data_path: &str,
    covariate_columns: Option<&[&str]>,
    options: Option<FitOptions>,
) -> Result<FitResult, String> {
    let mut model = crate::parser::model_parser::parse_model_file(Path::new(model_path))?;
    let population = read_nonmem_csv(Path::new(data_path), covariate_columns, None)?;
    let opts = options.unwrap_or_default();
    model.bloq_method = opts.bloq_method;
    model.gradient_method = opts.gradient_method;
    fit(&model, &population, &model.default_params, &opts)
}

/// Main fit entry point: CompiledModel + Population → FitResult.
///
/// When `options.threads` is `Some(n)`, the fit runs inside a scoped rayon
/// pool of `n` workers, so this setting is per-call (different fits in the
/// same process can use different thread counts). When `None`, rayon's
/// global pool is used (one worker per logical CPU).
pub fn fit(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    validate_covariates(model, population)?;
    match options.threads {
        Some(n) if n > 0 => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n)
                .build()
                .map_err(|e| format!("failed to build rayon pool with {} threads: {}", n, e))?;
            pool.install(|| fit_inner(model, population, init_params, options))
        }
        _ => fit_inner(model, population, init_params, options),
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    let chain = options.method_chain();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let unsupported_warnings = options.unsupported_keys_warnings();

    // Reset gradient timing counters for this fit so FERX_TIME_GRADIENTS
    // readouts are per-call rather than cumulative across a long R session.
    let time_gradients = std::env::var("FERX_TIME_GRADIENTS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if time_gradients {
        crate::estimation::inner_optimizer::GRADIENT_TIMINGS.reset();
    }
    if options.verbose {
        let chain_str: Vec<&str> = chain.iter().map(|m| m.label()).collect();
        // rayon::current_num_threads() reports whichever pool par_iter would use
        // from the current call — the scoped pool when options.threads is Some,
        // otherwise the global pool. So this stays accurate in both paths.
        let n_threads = rayon::current_num_threads();
        let thread_word = if n_threads == 1 { "thread" } else { "threads" };
        if !unsupported_warnings.is_empty() {
            eprintln!("--- Warnings ---");
            for w in &unsupported_warnings {
                eprintln!("  * {}", w);
            }
            eprintln!();
        }
        eprintln!(
            "Starting estimation (chain: {}) on {} {}...",
            chain_str.join(" → "),
            n_threads,
            thread_word
        );
        eprintln!(
            "  {} subjects, {} observations",
            population.subjects.len(),
            population.n_obs()
        );
        eprintln!(
            "  {} thetas, {} etas, {} sigmas",
            model.n_theta, model.n_eta, model.n_epsilon
        );
    }

    // Guard: SAEM does not support IOV (kappas are not sampled in the SAEM
    // stochastic approximation loop).  Fail early with a clear message.
    if model.n_kappa > 0 && chain.iter().any(|&m| m == EstimationMethod::Saem) {
        return Err(
            "method = saem does not support IOV (n_kappa > 0). \
             Use method = foce or method = focei for models with kappa declarations."
                .to_string(),
        );
    }

    // Run each stage in sequence, feeding params forward.
    let n_stages = chain.len();
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = unsupported_warnings;
    let mut total_iterations: usize = 0;

    for (stage_idx, &method) in chain.iter().enumerate() {
        if crate::cancel::is_cancelled(&options.cancel) {
            return Err("cancelled by user".to_string());
        }
        let is_last = stage_idx + 1 == n_stages;
        let mut stage_opts = options.clone();
        stage_opts.method = method;
        stage_opts.methods = Vec::new();
        // Per-stage interaction flag: FOCEI=on, FOCE=off, others inherit from user options.
        match method {
            EstimationMethod::FoceI => stage_opts.interaction = true,
            EstimationMethod::Foce => stage_opts.interaction = false,
            _ => {}
        }
        // Only run the covariance step on the final stage to avoid wasted work.
        if !is_last {
            stage_opts.run_covariance_step = false;
            stage_opts.sir = false;
        }

        if options.verbose && n_stages > 1 {
            eprintln!(
                "\n── Stage {}/{}: {} ──",
                stage_idx + 1,
                n_stages,
                method.label()
            );
        }

        let stage_result = match method {
            EstimationMethod::Saem => {
                saem::run_saem(model, population, &stage_params, &stage_opts)?
            }
            EstimationMethod::FoceGn | EstimationMethod::FoceGnHybrid => {
                crate::estimation::gauss_newton::run_foce_gn(
                    model,
                    population,
                    &stage_params,
                    &stage_opts,
                )
            }
            _ => optimize_population(model, population, &stage_params, &stage_opts),
        };

        stage_params = stage_result.params.clone();
        total_iterations += stage_result.n_iterations;
        for w in &stage_result.warnings {
            accumulated_warnings.push(if n_stages > 1 {
                format!("[{}] {}", method.label(), w)
            } else {
                w.clone()
            });
        }
        result = Some(stage_result);
    }

    if crate::cancel::is_cancelled(&options.cancel) {
        return Err("cancelled by user".to_string());
    }

    let mut result = result.expect("method chain must have at least one stage");
    // Overwrite with chain-aware totals
    result.n_iterations = total_iterations;
    result.warnings = accumulated_warnings;

    // Compute per-subject diagnostics
    let subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        &result.kappas,
        options.interaction,
    );

    let n_obs = population.n_obs();
    // Count only *estimated* parameters for information criteria. NONMEM's
    // convention: a FIX parameter doesn't cost a degree of freedom.
    let fixed_mask = crate::estimation::parameterization::packed_fixed_mask(init_params);
    let n_params = fixed_mask.iter().filter(|&&b| !b).count();

    let ofv = result.ofv;
    let aic = ofv + 2.0 * n_params as f64;
    let bic = ofv + n_params as f64 * (n_obs as f64).ln();

    // Extract SEs from covariance matrix using converged parameter values
    let (se_theta, se_omega, se_sigma) =
        extract_standard_errors(&result.covariance_matrix, &result.params);

    // Optional SIR step
    let mut warnings = result.warnings;

    // Report detected mu-referencing relationships (only when feature is enabled)
    if options.mu_referencing && !model.mu_refs.is_empty() {
        let mut names: Vec<&String> = model.mu_refs.keys().collect();
        names.sort();
        warnings.push(format!(
            "mu-ref: {}",
            names
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    // When M3 BLOQ is combined with non-interaction FOCE, mixing linearized
    // Gaussian residuals with non-linearized log Φ terms gives inconsistent
    // OFVs near the LLOQ boundary. The FOCE dispatcher routes affected
    // subjects through FOCEI internally — surface the promotion to the user.
    if matches!(model.bloq_method, BloqMethod::M3)
        && matches!(
            options.method,
            EstimationMethod::Foce | EstimationMethod::FoceGn
        )
        && !options.interaction
        && population.subjects.iter().any(|s| s.has_bloq())
    {
        warnings.push(
            "M3 BLOQ handling requires FOCEI semantics; subjects with CENS=1 \
             rows were evaluated with η-interaction. Set method=focei explicitly \
             to silence this notice."
                .to_string(),
        );
    }
    let sir_result = if options.sir && !crate::cancel::is_cancelled(&options.cancel) {
        if let Some(ref cov) = result.covariance_matrix {
            if options.verbose {
                eprintln!("\nRunning SIR...");
            }
            match crate::estimation::sir::run_sir(
                model,
                population,
                &result.params,
                &result.eta_hats,
                cov,
                result.ofv,
                options,
            ) {
                Ok(sir) => Some(sir),
                Err(e) => {
                    warnings.push(format!("SIR failed: {}", e));
                    None
                }
            }
        } else {
            warnings.push(
                "SIR requested but covariance matrix is not available. \
                 Enable covariance = true in [fit_options]."
                    .to_string(),
            );
            None
        }
    } else {
        None
    };

    let fit_result = FitResult {
        method: *chain.last().expect("chain non-empty"),
        method_chain: chain.clone(),
        converged: result.converged,
        ofv,
        aic,
        bic,
        theta: result.params.theta.clone(),
        theta_names: result.params.theta_names.clone(),
        omega: result.params.omega.matrix.clone(),
        sigma: result.params.sigma.values.clone(),
        covariance_matrix: result.covariance_matrix,
        se_theta,
        se_omega,
        se_sigma,
        theta_fixed: result.params.theta_fixed.clone(),
        omega_fixed: result.params.omega_fixed.clone(),
        sigma_fixed: result.params.sigma_fixed.clone(),
        subjects,
        n_obs,
        n_subjects: population.subjects.len(),
        n_parameters: n_params,
        n_iterations: result.n_iterations,
        interaction: options.interaction,
        warnings,
        sir_ci_theta: sir_result.as_ref().map(|s| s.ci_theta.clone()),
        sir_ci_omega: sir_result.as_ref().map(|s| s.ci_omega.clone()),
        sir_ci_sigma: sir_result.as_ref().map(|s| s.ci_sigma.clone()),
        sir_ess: sir_result.as_ref().map(|s| s.effective_sample_size),
        omega_iov: result.params.omega_iov.as_ref().map(|m| m.matrix.clone()),
        kappa_names: model.kappa_names.clone(),
        kappa_fixed: result.params.kappa_fixed.clone(),
        se_kappa: None,
        shrinkage_kappa: Vec::new(),
        ebe_kappas: result.kappas.clone(),
    };

    if options.verbose {
        output::print_results(&fit_result);
    }

    if time_gradients {
        let (ad_c, ad_n, fd_c, fd_n, jac_ad_c, jac_ad_n, jac_fd_c, jac_fd_n) =
            crate::estimation::inner_optimizer::GRADIENT_TIMINGS.snapshot();
        let ms = |n: u64| (n as f64) / 1_000_000.0;
        let avg_us = |n: u64, c: u64| {
            if c == 0 {
                0.0
            } else {
                (n as f64) / (c as f64) / 1_000.0
            }
        };
        eprintln!("--- Gradient timings (FERX_TIME_GRADIENTS=1) ---");
        eprintln!(
            "  BFGS (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            ad_c,
            ms(ad_n),
            avg_us(ad_n, ad_c)
        );
        eprintln!(
            "  BFGS (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            fd_c,
            ms(fd_n),
            avg_us(fd_n, fd_c)
        );
        eprintln!(
            "  Jac  (AD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_ad_c,
            ms(jac_ad_n),
            avg_us(jac_ad_n, jac_ad_c)
        );
        eprintln!(
            "  Jac  (FD):  {:>8} calls, {:>10.2} ms total, {:>8.2} µs/call",
            jac_fd_c,
            ms(jac_fd_n),
            avg_us(jac_fd_n, jac_fd_c)
        );
    }

    Ok(fit_result)
}

/// Compute per-subject diagnostics (IPRED, PRED, IWRES, CWRES)
fn compute_subject_results(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas_per_subject: &[Vec<DVector<f64>>],
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];
            let kappas: &[DVector<f64>] = if i < kappas_per_subject.len() {
                kappas_per_subject[i].as_slice()
            } else {
                &[]
            };

            // Individual predictions: f(eta_hat), with occasion-specific kappas for IOV.
            let ipred = if !kappas.is_empty() {
                let occ_groups = split_obs_by_occasion(subject);
                let mut ipreds = vec![0.0; subject.obs_times.len()];
                for (k, (_, obs_indices)) in occ_groups.iter().enumerate() {
                    let kap: &[f64] =
                        if k < kappas.len() { kappas[k].as_slice() } else { &[] };
                    let combined: Vec<f64> =
                        eta.iter().copied().chain(kap.iter().copied()).collect();
                    let pk = (model.pk_param_fn)(&params.theta, &combined, &subject.covariates);
                    let all_preds = model_preds(model, subject, &pk);
                    for &j in obs_indices {
                        ipreds[j] = all_preds[j];
                    }
                }
                ipreds
            } else {
                let pk_params_ind =
                    (model.pk_param_fn)(&params.theta, eta.as_slice(), &subject.covariates);
                model_preds(model, subject, &pk_params_ind)
            };

            // Population predictions: f(eta = 0, kappa = 0).
            let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
            let pk_params_pop =
                (model.pk_param_fn)(&params.theta, &zero_eta, &subject.covariates);
            let pred = model_preds(model, subject, &pk_params_pop);

            // IWRES (NaN on BLOQ rows — see compute_cwres for CWRES handling).
            let mut iwres = compute_iwres(
                &subject.observations,
                &ipred,
                model.error_model,
                &params.sigma.values,
            );
            for (j, c) in subject.cens.iter().enumerate() {
                if *c != 0 {
                    iwres[j] = f64::NAN;
                }
            }

            // CWRES
            let cwres = compute_cwres(
                subject,
                &ipred,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                model.error_model,
            );

            // OFV contribution
            let ofv_i = if !kappas.is_empty() {
                let omega_iov = params
                    .omega_iov
                    .as_ref()
                    .expect("omega_iov present when kappas non-empty");
                foce_subject_nll_iov(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                    kappas,
                    omega_iov,
                )
            } else {
                foce_subject_nll(
                    model,
                    subject,
                    &params.theta,
                    eta,
                    h,
                    &params.omega,
                    &params.sigma.values,
                    interaction,
                )
            };

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
            }
        })
        .collect()
}

/// Extract standard errors from covariance matrix on the packed parameter scale,
/// then transform back to the original scale via delta method.
fn extract_standard_errors(
    cov: &Option<DMatrix<f64>>,
    template: &ModelParameters,
) -> (Option<Vec<f64>>, Option<Vec<f64>>, Option<Vec<f64>>) {
    let cov = match cov {
        Some(c) => c,
        None => return (None, None, None),
    };

    let n = cov.nrows();
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    // SE on packed scale
    let se_packed: Vec<f64> = (0..n)
        .map(|i| {
            let v = cov[(i, i)];
            if v > 0.0 {
                v.sqrt()
            } else {
                0.0
            }
        })
        .collect();

    // Theta: SE on original scale via delta method
    // If x = log(theta), then SE(theta) = theta * SE(x)
    let se_theta: Vec<f64> = (0..n_theta)
        .map(|i| template.theta[i] * se_packed[i])
        .collect();

    // Omega: SE for diagonal variances
    // omega_ii = L_ii^2, so SE(omega_ii) ≈ 2*L_ii * SE(L_ii)
    // L_ii = exp(x_i), SE(L_ii) = L_ii * SE(x_i)
    // SE(omega_ii) = 2 * L_ii^2 * SE(x_i) = 2 * omega_ii * SE(x_i)
    let omega_start = n_theta;
    let se_omega: Vec<f64> = (0..n_eta)
        .map(|i| {
            let idx = if template.omega.diagonal {
                omega_start + i
            } else {
                // Index of diagonal element i in packed lower triangle
                omega_start + i * (i + 1) / 2 + i
            };
            if idx < n {
                2.0 * template.omega.matrix[(i, i)] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    // Sigma: SE via delta method (log-transformed)
    let sigma_start = omega_start
        + if template.omega.diagonal {
            n_eta
        } else {
            n_eta * (n_eta + 1) / 2
        };
    let se_sigma: Vec<f64> = (0..n_sigma)
        .map(|i| {
            let idx = sigma_start + i;
            if idx < n {
                template.sigma.values[i] * se_packed[idx]
            } else {
                0.0
            }
        })
        .collect();

    (Some(se_theta), Some(se_omega), Some(se_sigma))
}

/// Simulate observations from a model with given parameters (random seed).
pub fn simulate(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
) -> Vec<SimulationResult> {
    use rand::prelude::*;
    simulate_inner(model, population, params, n_sim, &mut thread_rng())
}

/// Simulate with a fixed seed for reproducibility.
pub fn simulate_with_seed(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    seed: u64,
) -> Vec<SimulationResult> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    simulate_inner(model, population, params, n_sim, &mut rng)
}

fn simulate_inner<R: rand::Rng>(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    n_sim: usize,
    rng: &mut R,
) -> Vec<SimulationResult> {
    use rand_distr::Normal;

    let normal = Normal::new(0.0, 1.0).unwrap();
    let n_eta = model.n_eta;

    let mut results = Vec::new();

    for sim_idx in 0..n_sim {
        for subject in &population.subjects {
            // Sample eta from N(0, Omega); append zero kappas for IOV models.
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let z_vec = DVector::from_column_slice(&z);
            let eta = &params.omega.chol * z_vec;
            let mut eta_slice: Vec<f64> = eta.iter().copied().collect();
            eta_slice.resize(n_eta + model.n_kappa, 0.0);

            // Compute individual parameters
            let pk_params = (model.pk_param_fn)(&params.theta, &eta_slice, &subject.covariates);

            // Predict concentrations
            let ipreds = model_preds(model, subject, &pk_params);

            // Add residual error
            for (j, &ipred) in ipreds.iter().enumerate() {
                let var = crate::stats::residual_error::residual_variance(
                    model.error_model,
                    ipred,
                    &params.sigma.values,
                );
                let eps: f64 = rng.sample(normal);
                let dv_sim = ipred + var.sqrt() * eps;

                results.push(SimulationResult {
                    sim: sim_idx + 1,
                    id: subject.id.clone(),
                    time: subject.obs_times[j],
                    ipred,
                    dv_sim,
                });
            }
        }
    }

    results
}

/// A single simulated observation
#[derive(Debug, Clone)]
pub struct SimulationResult {
    pub sim: usize,
    pub id: String,
    pub time: f64,
    pub ipred: f64,
    pub dv_sim: f64,
}

/// Predict concentrations for a population using given parameters (no random effects).
pub fn predict(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
) -> Vec<PredictionResult> {
    let zero_eta = vec![0.0_f64; model.n_eta + model.n_kappa];
    let mut results = Vec::new();

    for subject in &population.subjects {
        let pk_params = (model.pk_param_fn)(&params.theta, &zero_eta, &subject.covariates);
        let preds = model_preds(model, subject, &pk_params);

        for (j, &pred) in preds.iter().enumerate() {
            results.push(PredictionResult {
                id: subject.id.clone(),
                time: subject.obs_times[j],
                pred,
            });
        }
    }

    results
}

/// A single prediction
#[derive(Debug, Clone)]
pub struct PredictionResult {
    pub id: String,
    pub time: f64,
    pub pred: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
//  IOV integration tests
//
//  Each test builds a minimal warfarin-like 1-cpt IV model with a single kappa
//  for CL, simulates a small population (4 subjects × 2 occasions × 3 obs),
//  and verifies that `fit()` completes without panicking and returns meaningful
//  IOV estimates.  Tests run under `--features ci` (no autodiff required).
// ─────────────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod iov_integration {
    use super::fit;
    use crate::types::*;

    use std::collections::HashMap;

    // ── Model ────────────────────────────────────────────────────────────────
    fn make_iov_model() -> CompiledModel {
        let omega = OmegaMatrix::from_diagonal(&[0.09], vec!["ETA_CL".into()]);
        let omega_iov = OmegaMatrix::from_diagonal(&[0.04], vec!["KAPPA_CL".into()]);
        let default_params = ModelParameters {
            theta: vec![5.0, 50.0],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            theta_lower: vec![0.1, 5.0],
            theta_upper: vec![50.0, 500.0],
            theta_fixed: vec![false; 2],
            omega,
            omega_fixed: vec![false],
            sigma: SigmaVector { values: vec![0.05], names: vec!["PROP_ERR".into()] },
            sigma_fixed: vec![false],
            omega_iov: Some(omega_iov),
            kappa_fixed: vec![false],
        };
        CompiledModel {
            name: "iov_test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            // pk_param_fn: eta[0]=BSV for CL, eta[1]=KAPPA_CL (appended by IOV path)
            pk_param_fn: Box::new(|theta: &[f64], eta: &[f64], _: &HashMap<String, f64>| {
                let mut p = PkParams::default();
                let kappa = if eta.len() > 1 { eta[1] } else { 0.0 };
                p.values[0] = theta[0] * (eta[0] + kappa).exp(); // CL = TVCL * exp(ETA_CL + KAPPA_CL)
                p.values[1] = theta[1];                           // V
                p
            }),
            n_theta: 2,
            n_eta: 1,
            n_epsilon: 1,
            n_kappa: 1,
            kappa_names: vec!["KAPPA_CL".into()],
            theta_names: vec!["TVCL".into(), "TVV".into()],
            eta_names: vec!["ETA_CL".into()],
            default_params,
            mu_refs: HashMap::new(),
            tv_fn: None,
            pk_indices: vec![0, 1],
            eta_map: vec![0],
            pk_idx_f64: vec![0.0, 1.0],
            sel_flat: vec![1.0, 0.0],
            ode_spec: None,
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
            gradient_method: GradientMethod::Fd,
        }
    }

    // ── Population ───────────────────────────────────────────────────────────
    // 4 subjects, each with 2 occasions.  Times 1–3 = occasion 1, times 4–6 = occ 2.
    // Observations are plausible IV-bolus concentrations (dose=100).
    fn make_iov_population() -> Population {
        let obs_times = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let occasions = vec![1u32, 1, 1, 2, 2, 2];
        let dose_occ = vec![1u32, 2]; // two dose rows: one per occasion
        let subject_data: &[(&str, Vec<f64>)] = &[
            ("S1", vec![36.0, 28.0, 21.0, 34.0, 26.0, 19.0]),
            ("S2", vec![40.0, 32.0, 24.0, 38.0, 29.0, 22.0]),
            ("S3", vec![33.0, 25.0, 19.0, 31.0, 24.0, 18.0]),
            ("S4", vec![42.0, 33.0, 25.0, 39.0, 30.0, 23.0]),
        ];

        let subjects: Vec<Subject> = subject_data
            .iter()
            .map(|(id, obs)| Subject {
                id: id.to_string(),
                doses: vec![
                    DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0),
                    DoseEvent::new(3.5, 100.0, 1, 0.0, false, 0.0),
                ],
                obs_times: obs_times.clone(),
                observations: obs.clone(),
                obs_cmts: vec![1; 6],
                covariates: HashMap::new(),
                tvcov: HashMap::new(),
                cens: vec![0; 6],
                occasions: occasions.clone(),
                dose_occasions: dose_occ.clone(),
            })
            .collect();
        Population {
            subjects,
            covariate_names: Vec::new(),
            dv_column: "DV".to_string(),
        }
    }

    fn fast_opts(method: EstimationMethod, optimizer: Optimizer, mu_referencing: bool) -> FitOptions {
        FitOptions {
            method,
            methods: Vec::new(),
            outer_maxiter: 60,
            outer_gtol: 1e-3,
            inner_maxiter: 50,
            inner_tol: 1e-4,
            run_covariance_step: false,
            interaction: method == EstimationMethod::FoceI,
            mu_referencing,
            optimizer,
            lbfgs_memory: 5,
            verbose: false,
            ..FitOptions::default()
        }
    }

    // ── Helper ───────────────────────────────────────────────────────────────
    fn assert_iov_fit_ok(result: &FitResult) {
        assert!(result.ofv.is_finite(), "OFV must be finite");
        assert!(result.omega_iov.is_some(), "omega_iov must be populated");
        let iov_diag = result.omega_iov.as_ref().unwrap()[(0, 0)];
        assert!(iov_diag > 0.0, "omega_iov diagonal must be positive, got {iov_diag}");
        assert_eq!(result.kappa_names, vec!["KAPPA_CL"], "kappa name mismatch");
        assert_eq!(result.ebe_kappas.len(), 4, "expected kappas for 4 subjects");
        for (i, subj_kappas) in result.ebe_kappas.iter().enumerate() {
            assert_eq!(subj_kappas.len(), 2, "subject {i} should have 2 occasions");
        }
    }

    // ── Tests: FOCE + all outer optimizers ───────────────────────────────────

    #[test]
    fn test_iov_foce_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_slsqp() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Slsqp, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Lbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_nlopt_lbfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::NloptLbfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_mma() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Mma, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_foce_bfgs() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bfgs, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: FOCEI ─────────────────────────────────────────────────────────

    #[test]
    fn test_iov_focei_bobyqa() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: mu-referencing ─────────────────────────────────────────────────

    #[test]
    fn test_iov_foce_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Foce, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_focei_mu_referencing_on() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceI, Optimizer::Bobyqa, true);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("fit with mu_referencing should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Tests: GN and GN_Hybrid ───────────────────────────────────────────────

    #[test]
    fn test_iov_gn() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGn, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("GN fit should succeed");
        assert_iov_fit_ok(&result);
    }

    #[test]
    fn test_iov_gn_hybrid() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::FoceGnHybrid, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts).expect("GN hybrid fit should succeed");
        assert_iov_fit_ok(&result);
    }

    // ── Test: SAEM + IOV must return Err ──────────────────────────────────────

    #[test]
    fn test_iov_saem_returns_err() {
        let model = make_iov_model();
        let pop = make_iov_population();
        let opts = fast_opts(EstimationMethod::Saem, Optimizer::Bobyqa, false);
        let result = fit(&model, &pop, &model.default_params, &opts);
        assert!(result.is_err(), "SAEM with IOV must return an error");
        let msg = result.unwrap_err();
        assert!(
            msg.contains("saem") && msg.contains("IOV"),
            "error message should mention saem and IOV, got: {msg}"
        );
    }
}
