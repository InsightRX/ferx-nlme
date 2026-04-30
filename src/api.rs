use crate::estimation::outer_optimizer::optimize_population;
use crate::estimation::saem;
use crate::io::datareader::read_nonmem_csv;
use crate::io::output;
use crate::pk;
use crate::stats::likelihood::{compute_cwres, foce_subject_nll};
use crate::stats::residual_error::compute_iwres;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::path::Path;
use std::time::Instant;

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

    let population = read_nonmem_csv(Path::new(data_path), None)?;
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
    let population = read_nonmem_csv(Path::new(data_path), covariate_columns)?;
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

/// Probe whether NLopt CRS2-LM (used for global_search) is available.
fn probe_nlopt_algorithms() -> Vec<String> {
    fn dummy_obj(_x: &[f64], _grad: Option<&mut [f64]>, _data: &mut ()) -> f64 {
        0.0
    }
    let available = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _opt = nlopt::Nlopt::new(
            nlopt::Algorithm::Crs2Lm,
            1,
            dummy_obj,
            nlopt::Target::Minimize,
            (),
        );
    }));
    if available.is_err() {
        vec![
            "NLopt CRS2-LM not available in this build — global_search = true will fail. \
             Install a full NLopt build: brew install nlopt / apt install libnlopt-dev"
                .to_string(),
        ]
    } else {
        vec![]
    }
}

fn fit_inner(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<FitResult, String> {
    let fit_start = Instant::now();
    let chain = options.method_chain();
    // Compute up-front so we can both surface the warnings before the fit
    // starts (a long-running fit shouldn't bury a "this option is unused"
    // notice at the end) and carry them through into FitResult.warnings.
    let unsupported_warnings = options.unsupported_keys_warnings();

    // Capture thread count before chain runs (current_num_threads() reports
    // whichever Rayon pool is active — scoped pool when threads=Some, else global).
    let n_threads_used = rayon::current_num_threads();

    // Initialise the per-iteration optimizer trace if requested.
    if options.optimizer_trace {
        let pid = std::process::id();
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let path = format!("/tmp/ferx_trace_{}_{}.csv", pid, ts);
        if let Err(e) = crate::estimation::trace::init(path.clone()) {
            eprintln!("[ferx] warning: could not open trace file {}: {}", path, e);
        } else {
            eprintln!("[ferx] optimizer trace → {}", path);
        }
    }

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

    // Pre-compute n_params (uses init_params, available before chain runs).
    let fixed_mask = crate::estimation::parameterization::packed_fixed_mask(init_params);
    let n_params_pre = fixed_mask.iter().filter(|&&b| !b).count();

    // Probe NLopt algorithm availability only when global_search will actually
    // run — otherwise the CRS2-LM warning is misleading for users who never
    // requested it.
    let nlopt_missing = if options.global_search {
        probe_nlopt_algorithms()
    } else {
        Vec::new()
    };

    // Covariance step cost warning: fire before chain so user sees it
    // immediately. Use checked_mul so an absurd parameter count cannot wrap
    // and produce a bogus estimate; on overflow we still warn but suppress
    // the numeric estimate.
    let covariance_n_evals_estimated = if options.run_covariance_step && n_params_pre > 30 {
        n_params_pre.checked_mul(n_params_pre)
    } else {
        None
    };

    // Run each stage in sequence, feeding params forward.
    let n_stages = chain.len();
    let mut stage_params: ModelParameters = init_params.clone();
    let mut result: Option<crate::estimation::outer_optimizer::OuterResult> = None;
    let mut accumulated_warnings: Vec<String> = unsupported_warnings;

    // Emit NLopt / covariance warnings before any work starts.
    accumulated_warnings.extend(nlopt_missing.iter().cloned());
    if options.run_covariance_step && n_params_pre > 30 {
        if let Some(n_evals) = covariance_n_evals_estimated {
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → {} OFV evaluations \
                 (finite-difference Hessian). This may take several minutes \
                 on complex models.",
                n_params_pre, n_evals
            ));
        } else {
            // n_params² overflowed usize — warn without the (wrapped) number.
            accumulated_warnings.push(format!(
                "Covariance step: {} parameters → n² OFV evaluations \
                 (finite-difference Hessian). Estimate exceeds usize range; \
                 expect this to be very slow.",
                n_params_pre
            ));
        }
    }

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

    // Thread efficiency warnings (post-chain, uses n_threads_used captured above).
    let n_subjects = population.subjects.len();
    if n_subjects > 0 && n_threads_used > n_subjects {
        // `threads = 0` is not a valid Rayon pool size, so for n_subjects = 1
        // we still suggest a 1-thread pool.
        let suggested = n_subjects.max(1);
        result.warnings.push(format!(
            "{} threads configured but only {} subject(s) — consider threads = {} to reduce \
             scheduling overhead (no speed benefit beyond n_subjects)",
            n_threads_used, n_subjects, suggested
        ));
    }
    // SAEM-specific: MH scheduling has higher per-subject overhead than FOCE.
    // Skip when n_subjects < 2 (n_subjects/2 = 0 is meaningless and the prior
    // warning already covers the n_threads > n_subjects case).
    if chain.iter().any(|&m| m == EstimationMethod::Saem) && n_subjects >= 2 {
        let suggested = (n_subjects / 2).max(1);
        if n_threads_used > suggested {
            result.warnings.push(format!(
                "SAEM with more threads than subjects/2 may be slower due to MH scheduling \
                 overhead. Consider threads = {} for SAEM.",
                suggested
            ));
        }
    }

    // Compute per-subject diagnostics
    let subjects = compute_subject_results(
        model,
        population,
        &result.params,
        &result.eta_hats,
        &result.h_matrices,
        options.interaction,
    );

    let n_obs = population.n_obs();
    let n_params = n_params_pre;

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

    let final_method = *chain.last().expect("chain non-empty");
    let grad_inner =
        crate::build_info::gradient_method_inner(&crate::build_info::BUILD_INFO, model);
    let grad_outer = crate::build_info::gradient_method_outer(
        &crate::build_info::BUILD_INFO,
        final_method,
        options.optimizer,
    );

    // Flush and close the trace file; capture path for FitResult.
    let trace_path = crate::estimation::trace::finish();

    // Shrinkage
    let shrinkage_eta = compute_eta_shrinkage(&subjects, &result.params.omega.matrix);
    let shrinkage_eps = compute_eps_shrinkage(&subjects);

    // Covariance status
    let covariance_status = if !options.run_covariance_step {
        CovarianceStatus::NotRequested
    } else if result.covariance_matrix.is_some() {
        CovarianceStatus::Computed
    } else {
        CovarianceStatus::Failed
    };

    let wall_time_secs = fit_start.elapsed().as_secs_f64();

    let fit_result = FitResult {
        method: final_method,
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
        saem_mu_ref_m_step_evals_saved: result.saem_mu_ref_m_step_evals_saved,
        gradient_method_inner: grad_inner.as_str().to_string(),
        gradient_method_outer: grad_outer.as_str().to_string(),
        uses_ode_solver: model.is_ode_based(),
        n_threads_used,
        nlopt_missing_algorithms: nlopt_missing,
        covariance_n_evals_estimated,
        trace_path,
        ebe_convergence_warnings: result.ebe_convergence_warnings,
        max_unconverged_subjects: result.max_unconverged_subjects,
        total_ebe_fallbacks: result.total_ebe_fallbacks,
        covariance_status,
        shrinkage_eta,
        shrinkage_eps,
        wall_time_secs,
        model_name: model.name.clone(),
        ferx_version: env!("CARGO_PKG_VERSION").to_string(),
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
    interaction: bool,
) -> Vec<SubjectResult> {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let eta = &eta_hats[i];
            let h = &h_matrices[i];

            // Individual predictions: f(eta_hat)
            let pk_params_ind =
                (model.pk_param_fn)(&params.theta, eta.as_slice(), &subject.covariates);
            let ipred = model_preds(model, subject, &pk_params_ind);

            // Population predictions: f(eta = 0)
            let zero_eta = vec![0.0; model.n_eta];
            let pk_params_pop = (model.pk_param_fn)(&params.theta, &zero_eta, &subject.covariates);
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
            let ofv_i = foce_subject_nll(
                model,
                subject,
                &params.theta,
                eta,
                h,
                &params.omega,
                &params.sigma.values,
                interaction,
            );

            SubjectResult {
                id: subject.id.clone(),
                eta: eta.clone(),
                ipred,
                pred,
                iwres,
                cwres,
                ofv_contribution: 2.0 * ofv_i,
                cens: subject.cens.clone(),
                n_obs: subject.observations.len(),
            }
        })
        .collect()
}

/// ETA shrinkage: `1 - SD(eta_hat_k) / sqrt(omega_kk)` for each random effect k.
pub(crate) fn compute_eta_shrinkage(subjects: &[SubjectResult], omega: &DMatrix<f64>) -> Vec<f64> {
    let n_eta = omega.nrows();
    let n_subj = subjects.len();
    if n_subj < 2 || n_eta == 0 {
        return vec![f64::NAN; n_eta];
    }
    (0..n_eta)
        .map(|k| {
            let omega_var = omega[(k, k)];
            if omega_var <= 0.0 {
                return f64::NAN;
            }
            let omega_sd = omega_var.sqrt();
            let vals: Vec<f64> = subjects.iter().map(|s| s.eta[k]).collect();
            let mean = vals.iter().sum::<f64>() / n_subj as f64;
            let var =
                vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n_subj - 1) as f64;
            1.0 - var.sqrt() / omega_sd
        })
        .collect()
}

/// EPS shrinkage: `1 - SD(IWRES)` across all valid (non-NaN) residuals.
pub(crate) fn compute_eps_shrinkage(subjects: &[SubjectResult]) -> f64 {
    let vals: Vec<f64> = subjects
        .iter()
        .flat_map(|s| s.iwres.iter().copied())
        .filter(|v| v.is_finite())
        .collect();
    let n = vals.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = vals.iter().sum::<f64>() / n as f64;
    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    1.0 - var.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};

    fn make_subject(eta: Vec<f64>, iwres: Vec<f64>) -> SubjectResult {
        let n = iwres.len();
        SubjectResult {
            id: "1".to_string(),
            eta: DVector::from_vec(eta),
            ipred: vec![0.0; n],
            pred: vec![0.0; n],
            iwres,
            cwres: vec![0.0; n],
            ofv_contribution: 0.0,
            cens: vec![0; n],
            n_obs: n,
        }
    }

    #[test]
    fn test_eta_shrinkage_zero_when_eta_matches_omega_sd() {
        // If SD(eta_hat) == sqrt(omega), shrinkage = 0.
        // With n=2 subjects: eta = [+s, -s] => SD = s * sqrt(2/(n-1)) for n=2 => SD = s*sqrt(2)
        // For shrinkage=0: SD(eta_hat) = sqrt(omega). So pick omega = 2.0, eta = [+1, -1].
        let omega = DMatrix::from_diagonal_element(1, 1, 2.0);
        let subjects = vec![
            make_subject(vec![1.0], vec![0.0]),
            make_subject(vec![-1.0], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert_eq!(sh.len(), 1);
        // SD([1.0, -1.0]) = sqrt(((1-0)^2 + (-1-0)^2) / 1) = sqrt(2) ≈ 1.414
        // shrinkage = 1 - sqrt(2) / sqrt(2) = 0.0
        assert!((sh[0]).abs() < 1e-10, "expected ~0 shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_positive_when_etas_shrunk() {
        // Etas close to zero → shrinkage > 0
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects: Vec<SubjectResult> = (0..10)
            .map(|_| make_subject(vec![0.01], vec![0.0]))
            .collect();
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0] > 0.5, "expected high shrinkage, got {}", sh[0]);
    }

    #[test]
    fn test_eta_shrinkage_nan_when_omega_zero() {
        let omega = DMatrix::zeros(1, 1);
        let subjects = vec![
            make_subject(vec![0.1], vec![0.0]),
            make_subject(vec![-0.1], vec![0.0]),
        ];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN when omega=0");
    }

    #[test]
    fn test_eta_shrinkage_nan_when_fewer_than_2_subjects() {
        let omega = DMatrix::from_diagonal_element(1, 1, 1.0);
        let subjects = vec![make_subject(vec![0.5], vec![0.0])];
        let sh = compute_eta_shrinkage(&subjects, &omega);
        assert!(sh[0].is_nan(), "expected NaN with only 1 subject");
    }

    #[test]
    fn test_eps_shrinkage_near_zero_for_unit_normal_iwres() {
        // IWRES with sample SD = 1 => shrinkage = 0.
        // For n=2: SD([a, -a]) = a*sqrt(2); set a = 1/sqrt(2) so SD = 1.
        let a = 1.0_f64 / 2.0_f64.sqrt();
        let subjects = vec![
            make_subject(vec![0.0], vec![a]),
            make_subject(vec![0.0], vec![-a]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "expected ~0 eps shrinkage, got {}", sh);
    }

    #[test]
    fn test_eps_shrinkage_nan_for_fewer_than_2_residuals() {
        let subjects = vec![make_subject(vec![0.0], vec![0.5])];
        assert!(compute_eps_shrinkage(&subjects).is_nan());
    }

    #[test]
    fn test_eps_shrinkage_ignores_nan_iwres() {
        // BLOQ rows have NaN IWRES — they must be filtered out.
        // After filtering, two valid values with SD=1 remain => shrinkage = 0.
        let a = 1.0_f64 / 2.0_f64.sqrt();
        let subjects = vec![
            make_subject(vec![0.0], vec![a, f64::NAN]),
            make_subject(vec![0.0], vec![-a, f64::NAN]),
        ];
        let sh = compute_eps_shrinkage(&subjects);
        assert!((sh).abs() < 1e-10, "NaN IWRES not filtered, got {}", sh);
    }
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
            // Sample eta from N(0, Omega)
            let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(normal)).collect();
            let z_vec = DVector::from_column_slice(&z);
            let eta = &params.omega.chol * z_vec;
            let eta_slice: Vec<f64> = eta.iter().copied().collect();

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
    let zero_eta = vec![0.0; model.n_eta];
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
