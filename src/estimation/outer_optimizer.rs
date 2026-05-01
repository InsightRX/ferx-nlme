use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{foce_population_nll, foce_population_nll_iov};
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Result of outer optimization
pub struct OuterResult {
    pub params: ModelParameters,
    pub ofv: f64,
    pub converged: bool,
    pub n_iterations: usize,
    pub eta_hats: Vec<DVector<f64>>,
    pub h_matrices: Vec<DMatrix<f64>>,
    /// Per-occasion kappa EBEs for each subject. Empty vecs when `n_kappa == 0`.
    pub kappas: Vec<Vec<DVector<f64>>>,
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub warnings: Vec<String>,
    /// Estimated OFV evaluations saved by the SAEM mu-ref gradient step M-step.
    /// Non-None only when method=saem and mu_referencing=true.
    pub saem_mu_ref_m_step_evals_saved: Option<u64>,
    pub ebe_convergence_warnings: u32,
    pub max_unconverged_subjects: u32,
    pub total_ebe_fallbacks: u32,
}

/// Run the outer optimization loop (population parameter estimation).
pub fn optimize_population(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    match options.optimizer {
        Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma | Optimizer::Bobyqa => {
            optimize_nlopt(model, population, init_params, options)
        }
        Optimizer::Bfgs | Optimizer::Lbfgs => {
            optimize_bfgs(model, population, init_params, options)
        }
        Optimizer::TrustRegion => crate::estimation::trust_region::optimize_trust_region(
            model,
            population,
            init_params,
            options,
        ),
    }
}

/// Warm-started variant: starts from given EBEs and H-matrices instead of zeros.
/// Used by the Gauss-Newton hybrid to polish from the GN result.
pub fn optimize_population_warm(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
    warm_etas: &[DVector<f64>],
    warm_h_mats: &[DMatrix<f64>],
) -> OuterResult {
    // For now, delegate to the standard path — the inner loop warm-starts
    // from the provided EBEs automatically via the NloptState initialization.
    // TODO: pass warm_etas into the NLopt state directly for tighter coupling.
    let _ = (warm_etas, warm_h_mats);
    optimize_population(model, population, init_params, options)
}

// ═══════════════════════════════════════════════════════════════════════════
//  NLopt-based outer optimizer (matches Julia's NLopt path exactly)
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch to the IOV-aware or standard population NLL based on model.n_kappa.
/// `kappas` is ignored (may be empty) when `model.n_kappa == 0`.
pub(crate) fn pop_nll(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    interaction: bool,
) -> f64 {
    if model.n_kappa > 0 {
        if let Some(ref iov) = params.omega_iov {
            return foce_population_nll_iov(
                model, population, &params.theta, eta_hats, h_matrices,
                kappas, &params.omega, iov, &params.sigma.values, interaction,
            );
        }
    }
    foce_population_nll(
        model, population, &params.theta, eta_hats, h_matrices,
        &params.omega, &params.sigma.values, interaction,
    )
}

/// State passed through NLopt's user-data mechanism
struct NloptState {
    cached_etas: Vec<DVector<f64>>,
    cached_h_mats: Vec<DMatrix<f64>>,
    best_ofv: f64,
    n_evals: usize,
    /// Previous parameter vector — used to compute step_norm for the trace.
    prev_x: Vec<f64>,
}

fn optimize_nlopt(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x0 = pack_params(init_params);
    clamp_to_bounds(&mut x0, &bounds);
    let n = x0.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();

    // Per-element scale factors: present O(1) coordinates to NLopt.
    let scale: Vec<f64> = if options.scale_params {
        compute_scale(&x0)
    } else {
        vec![1.0; n]
    };
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    // Scale x0 into optimizer space: xs[i] = x[i] / scale[i].
    for i in 0..n {
        x0[i] /= scale[i];
    }

    let state = NloptState {
        cached_etas: vec![DVector::zeros(n_eta); n_subj],
        cached_h_mats: Vec::new(),
        best_ofv: f64::INFINITY,
        n_evals: 0,
        prev_x: x0.clone(),
    };

    // External counter mirrors state.n_evals — nlopt doesn't hand `state`
    // back after `opt.optimize()`, so we need an Arc to read the final
    // count for reporting. Keep both in sync inside the objective closure.
    let n_evals_outer = Arc::new(AtomicUsize::new(0));
    let n_evals_cl = Arc::clone(&n_evals_outer);

    // EBE stats accumulator: tracks worst unconverged count and total fallbacks.
    #[derive(Default)]
    struct EbeAccum {
        max_unconverged: usize,
        total_fallback: usize,
        n_convergence_warnings: usize,
    }
    let ebe_accum: Arc<Mutex<EbeAccum>> = Arc::new(Mutex::new(EbeAccum::default()));
    let ebe_accum_cl = Arc::clone(&ebe_accum);

    // Select NLopt algorithm
    let algo = match options.optimizer {
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::NloptLbfgs => nlopt::Algorithm::Lbfgs,
        Optimizer::Mma => nlopt::Algorithm::Mma,
        Optimizer::Bobyqa => nlopt::Algorithm::Bobyqa,
        _ => nlopt::Algorithm::Slsqp,
    };

    let verbose = options.verbose;

    // NLopt objective: receives xs (scaled), unscales before running inner loop.
    // Gradient: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i] (chain rule).
    let objective = |xs: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        // Cooperative cancellation: short-circuit cheaply so NLopt burns through
        // its remaining iteration budget in microseconds instead of minutes.
        if crate::cancel::is_cancelled(&options.cancel) {
            if let Some(g) = grad {
                for gi in g.iter_mut() {
                    *gi = 0.0;
                }
            }
            return 1e20;
        }
        // Unscale from optimizer space to real (log/Cholesky) space.
        let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let params = unpack_params(&x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);

        // Run inner loop (warm-started)
        let (ehs, hms, ebe_stats, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&state.cached_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
        );

        // Compute OFV with fixed EBEs
        let nll = pop_nll(model, population, &params, &ehs, &hms, &kappas, options.interaction);
        let raw_ofv = 2.0 * nll;

        // EBE convergence guard: reject step when too many subjects unconverged.
        let unconverged_frac = ebe_stats.n_unconverged as f64 / (n_subj as f64).max(1.0);
        let ebe_guard_triggered = raw_ofv.is_finite()
            && unconverged_frac > options.max_unconverged_frac
            && options.max_unconverged_frac >= 0.0;
        {
            let mut acc = ebe_accum_cl.lock().unwrap();
            if acc.max_unconverged < ebe_stats.n_unconverged {
                acc.max_unconverged = ebe_stats.n_unconverged;
            }
            acc.total_fallback += ebe_stats.n_fallback;
            if ebe_guard_triggered {
                acc.n_convergence_warnings += 1;
            }
        }

        let ofv = if ebe_guard_triggered {
            1e20
        } else if raw_ofv.is_finite() {
            raw_ofv
        } else {
            1e20
        };

        // Compute gradient if requested (central FD with fixed EBEs)
        let mut grad_norm_for_trace: Option<f64> = None;
        if let Some(g) = grad {
            // If OFV is non-finite, gradient is meaningless — use steepest ascent
            // toward center of bounds to nudge optimizer back
            if !raw_ofv.is_finite() {
                for i in 0..g.len() {
                    let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                    g[i] = 100.0 * (xs[i] - center_s);
                }
                state.n_evals += 1;
                n_evals_cl.fetch_add(1, Ordering::Relaxed);
                return ofv;
            }
            let kappas_ref = &kappas;
            let ofv_fn = |xp: &[f64], eh: &[DVector<f64>], hm: &[DMatrix<f64>]| -> f64 {
                let p = unpack_params(xp, init_params);
                2.0 * pop_nll(model, population, &p, eh, hm, kappas_ref, options.interaction)
            };
            // FD gradient in unscaled space; multiply by scale for scaled gradient.
            let grad_vec = gradient_cd(&x, &bounds, &ehs, &hms, &ofv_fn);
            let mut sq = 0.0_f64;
            for i in 0..g.len() {
                let gi = if grad_vec[i].is_finite() { grad_vec[i] * scale[i] } else { 0.0 };
                g[i] = gi;
                sq += gi * gi;
            }
            grad_norm_for_trace = Some(sq.sqrt());
        }

        // Update state
        state.cached_etas = ehs;
        state.cached_h_mats = hms;
        state.n_evals += 1;
        n_evals_cl.fetch_add(1, Ordering::Relaxed);
        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                eprintln!("Eval {:>4}: OFV = {:.6}", state.n_evals, ofv);
            }
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm = {
                let sq: f64 = xs.iter().zip(&state.prev_x).map(|(a, b)| (a - b).powi(2)).sum();
                let n = sq.sqrt();
                if n > 0.0 { Some(n) } else { None }
            };
            let method_str = match options.method {
                EstimationMethod::FoceI => "focei",
                _ => "foce",
            };
            let optimizer_str = match algo {
                nlopt::Algorithm::Bobyqa => "bobyqa",
                nlopt::Algorithm::Mma    => "mma",
                nlopt::Algorithm::Lbfgs  => "nlopt_lbfgs",
                _                        => "slsqp",
            };
            crate::estimation::trace::write_foce(
                state.n_evals,
                method_str,
                ofv,
                grad_norm_for_trace,
                step_norm,
                optimizer_str,
                Some(ebe_stats.n_unconverged),
                Some(ebe_stats.n_fallback),
            );
        }
        state.prev_x = xs.to_vec();

        ofv
    };

    // Create NLopt optimizer with state (operates in scaled xs space)
    let mut opt = nlopt::Nlopt::new(algo, n, objective, nlopt::Target::Minimize, state);
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    opt.set_maxeval(options.outer_maxiter as u32 * (n as u32 + 1))
        .unwrap();
    // Use very loose tolerances — FOCE objective is noisy from EBE re-estimation.
    // Let maxeval be the primary stopping criterion.
    opt.set_xtol_rel(1e-12).unwrap();
    opt.set_ftol_rel(1e-12).unwrap();

    if options.verbose {
        eprintln!(
            "Starting NLopt {:?} optimization ({} parameters)...",
            algo, n
        );
    }

    // Run optimization
    let result = opt.optimize(&mut x0);

    let (mut converged, first_algo) = match &result {
        Ok((status, _)) => {
            if options.verbose {
                eprintln!("NLopt finished: {:?}", status);
            }
            (
                matches!(
                    status,
                    nlopt::SuccessState::Success
                        | nlopt::SuccessState::FtolReached
                        | nlopt::SuccessState::XtolReached
                        | nlopt::SuccessState::StopValReached
                ),
                algo,
            )
        }
        Err((fail, _)) => {
            if options.verbose {
                eprintln!("NLopt stopped: {:?}", fail);
            }
            (matches!(fail, nlopt::FailState::RoundoffLimited), algo)
        }
    };

    drop(opt);

    // Fallback: if L-BFGS failed, retry with SLSQP from current best point.
    // Skip the fallback if the user cancelled — no point burning more cycles.
    let already_slsqp = matches!(first_algo, nlopt::Algorithm::Slsqp);
    let cancelled = crate::cancel::is_cancelled(&options.cancel);
    if !converged && !already_slsqp && !cancelled {
        if options.verbose {
            eprintln!("Retrying with NLopt SLSQP from current point...");
        }

        let state2 = NloptState {
            cached_etas: vec![DVector::zeros(n_eta); n_subj],
            cached_h_mats: Vec::new(),
            best_ofv: f64::INFINITY,
            n_evals: 0,
            prev_x: x0.clone(),
        };

        let n_evals_cl2 = Arc::clone(&n_evals_outer);
        let ebe_accum_cl2 = Arc::clone(&ebe_accum);
        // SLSQP fallback also operates in scaled xs space (same scale as primary opt).
        let objective2 = |xs: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
            if crate::cancel::is_cancelled(&options.cancel) {
                if let Some(g) = grad {
                    for gi in g.iter_mut() {
                        *gi = 0.0;
                    }
                }
                return 1e20;
            }
            let x: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
            let params = unpack_params(&x, init_params);
            let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
            let (ehs, hms, ebe_stats2, kappas) = run_inner_loop_warm(
                model,
                population,
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&state.cached_etas),
                Some(&mu_k),
                options.min_obs_for_convergence_check as usize,
            );
            let nll = pop_nll(model, population, &params, &ehs, &hms, &kappas, options.interaction);
            let raw_ofv = 2.0 * nll;

            let unconverged_frac2 = ebe_stats2.n_unconverged as f64 / (n_subj as f64).max(1.0);
            let ebe_guard2 = raw_ofv.is_finite()
                && unconverged_frac2 > options.max_unconverged_frac
                && options.max_unconverged_frac >= 0.0;
            {
                let mut acc = ebe_accum_cl2.lock().unwrap();
                if acc.max_unconverged < ebe_stats2.n_unconverged {
                    acc.max_unconverged = ebe_stats2.n_unconverged;
                }
                acc.total_fallback += ebe_stats2.n_fallback;
                if ebe_guard2 {
                    acc.n_convergence_warnings += 1;
                }
            }
            let ofv = if ebe_guard2 {
                1e20
            } else if raw_ofv.is_finite() {
                raw_ofv
            } else {
                1e20
            };

            let mut grad_norm_for_trace: Option<f64> = None;
            if let Some(g) = grad {
                if !raw_ofv.is_finite() {
                    for i in 0..g.len() {
                        let center_s = (lower_s[i] + upper_s[i]) / 2.0;
                        g[i] = 100.0 * (xs[i] - center_s);
                    }
                    state.n_evals += 1;
                    n_evals_cl2.fetch_add(1, Ordering::Relaxed);
                    return ofv;
                }
                let kappas_ref = &kappas;
                let ofv_fn = |xp: &[f64], eh: &[DVector<f64>], hm: &[DMatrix<f64>]| -> f64 {
                    let p = unpack_params(xp, init_params);
                    2.0 * pop_nll(model, population, &p, eh, hm, kappas_ref, options.interaction)
                };
                let grad_vec = gradient_cd(&x, &bounds, &ehs, &hms, &ofv_fn);
                let mut sq = 0.0_f64;
                for i in 0..g.len() {
                    let gi = if grad_vec[i].is_finite() { grad_vec[i] * scale[i] } else { 0.0 };
                    g[i] = gi;
                    sq += gi * gi;
                }
                grad_norm_for_trace = Some(sq.sqrt());
            }

            state.cached_etas = ehs;
            state.cached_h_mats = hms;
            state.n_evals += 1;
            n_evals_cl2.fetch_add(1, Ordering::Relaxed);
            if ofv < state.best_ofv {
                state.best_ofv = ofv;
                if verbose {
                    eprintln!("Eval {:>4}: OFV = {:.6} (SLSQP)", state.n_evals, ofv);
                }
            }

            // Optimizer trace (SLSQP fallback, step_norm in scaled space)
            if crate::estimation::trace::is_active() {
                let step_norm = {
                    let sq: f64 = xs.iter().zip(&state.prev_x).map(|(a, b)| (a - b).powi(2)).sum();
                    let n = sq.sqrt();
                    if n > 0.0 { Some(n) } else { None }
                };
                let method_str = match options.method {
                    EstimationMethod::FoceI => "focei",
                    _ => "foce",
                };
                crate::estimation::trace::write_foce(
                    state.n_evals,
                    method_str,
                    ofv,
                    grad_norm_for_trace,
                    step_norm,
                    "slsqp",
                    Some(ebe_stats2.n_unconverged),
                    Some(ebe_stats2.n_fallback),
                );
            }
            state.prev_x = xs.to_vec();

            ofv
        };

        let mut opt2 = nlopt::Nlopt::new(
            nlopt::Algorithm::Slsqp,
            n,
            objective2,
            nlopt::Target::Minimize,
            state2,
        );
        opt2.set_lower_bounds(&lower_s).unwrap();
        opt2.set_upper_bounds(&upper_s).unwrap();
        opt2.set_maxeval(options.outer_maxiter as u32 * (n as u32 + 1))
            .unwrap();
        opt2.set_xtol_rel(1e-12).unwrap();
        opt2.set_ftol_rel(1e-12).unwrap();

        let result2 = opt2.optimize(&mut x0);
        converged = match &result2 {
            Ok((status, _)) => {
                if options.verbose {
                    eprintln!("NLopt SLSQP finished: {:?}", status);
                }
                matches!(
                    status,
                    nlopt::SuccessState::Success
                        | nlopt::SuccessState::FtolReached
                        | nlopt::SuccessState::XtolReached
                        | nlopt::SuccessState::StopValReached
                )
            }
            Err((fail, _)) => {
                if options.verbose {
                    eprintln!("NLopt SLSQP stopped: {:?}", fail);
                }
                matches!(fail, nlopt::FailState::RoundoffLimited)
            }
        };
        drop(opt2);
    }

    // Unscale x0 back from optimizer space to real (log/Cholesky) space.
    for i in 0..n {
        x0[i] *= scale[i];
    }

    let final_params = unpack_params(&x0, init_params);
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);

    // Final inner loop at converged parameters
    let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&final_mu_k),
        options.min_obs_for_convergence_check as usize,
    );

    let final_nll = pop_nll(
        model, population, &final_params, &final_ehs, &final_hms, &final_kappas,
        options.interaction,
    );
    let final_ofv = 2.0 * final_nll;

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // Covariance step (skip if user cancelled — it's expensive and the result
    // will be discarded by the top-level fit() anyway).
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            compute_covariance(
                &x0,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            )
        } else {
            None
        };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }
    if covariance_matrix.is_none() && options.run_covariance_step {
        warnings.push("Covariance step failed".to_string());
    }

    let ebe_final = ebe_accum.lock().unwrap();
    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        // NLopt doesn't expose an "iteration" count (BOBYQA/SLSQP don't have
        // iterations in the textbook sense), so report the number of
        // objective-function evaluations instead — the only monotone
        // progress counter NLopt exposes, and the quantity most users
        // actually care about ("how much work did the fit do").
        n_iterations: n_evals_outer.load(Ordering::Relaxed),
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        ebe_convergence_warnings: ebe_final.n_convergence_warnings as u32,
        max_unconverged_subjects: ebe_final.max_unconverged as u32,
        total_ebe_fallbacks: ebe_final.total_fallback as u32,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Hand-rolled BFGS outer optimizer (legacy fallback)
// ═══════════════════════════════════════════════════════════════════════════

fn optimize_bfgs(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let n = x.len();
    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let mut warnings = Vec::new();
    let mut cached_etas: Vec<DVector<f64>> = vec![DVector::zeros(n_eta); n_subj];

    // Closures operating on unscaled real (log/Cholesky) space.
    let ofv_at_fixed = |x: &[f64],
                        eta_hats: &[DVector<f64>],
                        h_matrices: &[DMatrix<f64>],
                        kappas: &[Vec<DVector<f64>>]|
     -> f64 {
        let params = unpack_params(x, init_params);
        2.0 * pop_nll(model, population, &params, eta_hats, h_matrices, kappas, options.interaction)
    };

    let f_only = |x: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
        );
        let ofv = 2.0 * pop_nll(model, population, &params, &ehs, &hms, &kappas, options.interaction);
        if ofv.is_finite() { ofv } else { 1e20 }
    };

    let fdfg = |x: &[f64],
                prev_etas: &[DVector<f64>]|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _, kappas) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
            options.min_obs_for_convergence_check as usize,
        );
        let kappas_ref = &kappas;
        let ofv_fn_fixed =
            |xp: &[f64], eh: &[DVector<f64>], hm: &[DMatrix<f64>]| -> f64 {
                ofv_at_fixed(xp, eh, hm, kappas_ref)
            };
        let ofv = ofv_at_fixed(x, &ehs, &hms, &kappas);
        let g = gradient_cd(x, &bounds, &ehs, &hms, &ofv_fn_fixed);
        let f = if ofv.is_finite() { ofv } else { 1e20 };
        (f, g, ehs, hms)
    };

    // Per-element scale factors for the BFGS outer loop.
    let scale: Vec<f64> = if options.scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };
    let lower_s: Vec<f64> = (0..n).map(|i| bounds.lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| bounds.upper[i] / scale[i]).collect();
    let bounds_s = PackedBounds { lower: lower_s, upper: upper_s };

    // Wrappers that operate in scaled space; unscale before calling base closures.
    let fdfg_s = |xs: &[f64],
                  prev_etas: &[DVector<f64>]|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        let (f, g_r, ehs, hms) = fdfg(&x_r, prev_etas);
        let g_s: Vec<f64> = (0..n).map(|i| g_r[i] * scale[i]).collect();
        (f, g_s, ehs, hms)
    };

    let f_only_s = |xs: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let x_r: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();
        f_only(&x_r, prev_etas)
    };

    // Scale initial x into optimizer space.
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();

    let (mut f_val, mut g, ehs, _) = fdfg_s(&xs, &cached_etas);
    cached_etas = ehs;

    if options.verbose {
        eprintln!("Iter {:>4}: OFV = {:.6}", 0, f_val);
    }

    let mut h_inv = DMatrix::<f64>::identity(n, n);
    let mut converged = false;
    let mut n_iterations = 0;
    let mut stall_count = 0;

    for iter in 1..=options.outer_maxiter {
        n_iterations = iter;

        if crate::cancel::is_cancelled(&options.cancel) {
            warnings.push("cancelled by user".to_string());
            break;
        }

        let g_norm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if g_norm < options.outer_gtol {
            if options.verbose {
                eprintln!("Converged at iteration {} (|g| = {:.2e})", iter, g_norm);
            }
            converged = true;
            break;
        }

        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let mut d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 || !dg.is_finite() {
            d = g.iter().map(|gi| -gi).collect();
            h_inv = DMatrix::identity(n, n);
        }

        let alpha =
            backtracking_line_search_warm(&xs, &d, &g, f_val, &bounds_s, &cached_etas, &f_only_s);

        if alpha < 1e-18 {
            stall_count += 1;
            if stall_count >= 10 {
                if options.verbose {
                    eprintln!("Stopping: line search stalled at iteration {}", iter);
                }
                break;
            }
            h_inv = DMatrix::identity(n, n);
            continue;
        }
        stall_count = 0;

        let xs_old = xs.clone();
        for i in 0..n {
            xs[i] = (xs[i] + alpha * d[i]).clamp(bounds_s.lower[i], bounds_s.upper[i]);
        }

        let (f_new, g_new, ehs, _) = fdfg_s(&xs, &cached_etas);
        cached_etas = ehs;

        bfgs_update(&mut h_inv, &xs, &xs_old, &g_new, &g, n);

        let prev_ofv = f_val;
        f_val = f_new;
        g = g_new;

        if options.verbose && (iter % 10 == 0 || iter <= 5) {
            eprintln!(
                "Iter {:>4}: OFV = {:.6}  |g| = {:.2e}  alpha = {:.2e}",
                iter, f_val, g_norm, alpha
            );
        }

        // Optimizer trace (step_norm in scaled space)
        if crate::estimation::trace::is_active() {
            let step_norm: f64 = (0..n)
                .map(|i| (xs[i] - xs_old[i]).powi(2))
                .sum::<f64>()
                .sqrt();
            let method_str = match options.method {
                EstimationMethod::FoceI => "focei",
                _ => "foce",
            };
            let optimizer_str = match options.optimizer {
                Optimizer::Lbfgs => "lbfgs",
                _ => "bfgs",
            };
            crate::estimation::trace::write_foce(
                iter,
                method_str,
                f_val,
                Some(g_norm),
                Some(step_norm),
                optimizer_str,
                None,
                None,
            );
        }

        let rel_change = (f_val - prev_ofv).abs() / (f_val.abs() + 1.0);
        if rel_change < 1e-8 && g_norm < 0.1 {
            if options.verbose {
                eprintln!(
                    "Converged at iteration {} (rel OFV change: {:.2e}, |g| = {:.2e})",
                    iter, rel_change, g_norm
                );
            }
            converged = true;
            break;
        }
    }

    // Unscale xs back to real (log/Cholesky) space for unpacking and covariance.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let final_params = unpack_params(&x_final, init_params);
    let bfgs_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (final_ehs, final_hms, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cached_etas),
        Some(&bfgs_final_mu_k),
        options.min_obs_for_convergence_check as usize,
    );
    let final_ofv = ofv_at_fixed(&x_final, &final_ehs, &final_hms, &final_kappas);

    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
            if options.verbose {
                eprintln!("Computing covariance matrix...");
            }
            compute_covariance(
                &x_final,
                init_params,
                model,
                population,
                &final_ehs,
                &final_hms,
                &final_kappas,
                options,
            )
        } else {
            None
        };

    if !converged {
        warnings.push("Outer optimization did not converge".to_string());
    }
    if covariance_matrix.is_none()
        && options.run_covariance_step
        && !crate::cancel::is_cancelled(&options.cancel)
    {
        warnings.push("Covariance step failed".to_string());
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        kappas: final_kappas,
        covariance_matrix,
        warnings,
        saem_mu_ref_m_step_evals_saved: None,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Shared utilities
// ═══════════════════════════════════════════════════════════════════════════

fn bfgs_update(
    h_inv: &mut DMatrix<f64>,
    x_new: &[f64],
    x_old: &[f64],
    g_new: &[f64],
    g_old: &[f64],
    n: usize,
) {
    let s: Vec<f64> = (0..n).map(|i| x_new[i] - x_old[i]).collect();
    let y: Vec<f64> = (0..n).map(|i| g_new[i] - g_old[i]).collect();
    let sy: f64 = s.iter().zip(y.iter()).map(|(si, yi)| si * yi).sum();
    if sy > 1e-12 {
        let rho = 1.0 / sy;
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let eye = DMatrix::<f64>::identity(n, n);
        let rs_yt = rho * &s_vec * y_vec.transpose();
        let ry_st = rho * &y_vec * s_vec.transpose();
        let rss = rho * &s_vec * s_vec.transpose();
        *h_inv = (&eye - &rs_yt) * &*h_inv * (&eye - &ry_st) + rss;
    } else {
        *h_inv = DMatrix::identity(n, n);
    }
}

/// Central finite-difference gradient of FOCE OFV with EBEs held fixed.
///
/// Parameters with `bounds.lower[i] == bounds.upper[i]` (e.g. FIX parameters)
/// are skipped: the perturbed point would be clamped back to `x[i]` so
/// `actual_2h` is zero anyway, but skipping avoids two full OFV evaluations
/// per fixed coordinate.
fn gradient_cd(
    x: &[f64],
    bounds: &PackedBounds,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    ofv: &dyn Fn(&[f64], &[DVector<f64>], &[DMatrix<f64>]) -> f64,
) -> Vec<f64> {
    let n = x.len();
    let eps = 1e-5;
    let mut g = vec![0.0; n];
    let mut x_work = x.to_vec();

    for i in 0..n {
        if (bounds.upper[i] - bounds.lower[i]).abs() < 1e-16 {
            continue; // FIX parameter
        }
        let h = eps * (1.0 + x[i].abs());
        let xi_plus = (x[i] + h).min(bounds.upper[i]);
        let xi_minus = (x[i] - h).max(bounds.lower[i]);
        let actual_2h = xi_plus - xi_minus;
        if actual_2h.abs() < 1e-16 {
            continue;
        }

        x_work[i] = xi_plus;
        let f_plus = ofv(&x_work, eta_hats, h_matrices);
        x_work[i] = xi_minus;
        let f_minus = ofv(&x_work, eta_hats, h_matrices);
        x_work[i] = x[i];

        // If either evaluation is non-finite, use one-sided FD from the base point
        if f_plus.is_finite() && f_minus.is_finite() {
            let gi = (f_plus - f_minus) / actual_2h;
            if gi.is_finite() {
                g[i] = gi;
            }
        } else {
            // Fallback: one-sided from base
            let f0 = ofv(&x, eta_hats, h_matrices);
            if f_plus.is_finite() && f0.is_finite() {
                let gi = (f_plus - f0) / (xi_plus - x[i]);
                if gi.is_finite() {
                    g[i] = gi;
                }
            } else if f_minus.is_finite() && f0.is_finite() {
                let gi = (f0 - f_minus) / (x[i] - xi_minus);
                if gi.is_finite() {
                    g[i] = gi;
                }
            }
        }
    }
    g
}

fn backtracking_line_search_warm(
    x: &[f64],
    d: &[f64],
    g: &[f64],
    f0: f64,
    bounds: &PackedBounds,
    prev_etas: &[DVector<f64>],
    f_only: &dyn Fn(&[f64], &[DVector<f64>]) -> f64,
) -> f64 {
    let c1 = 1e-4;
    let n = x.len();
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
    if dg >= 0.0 {
        return 0.0;
    }

    let mut alpha = 1.0;
    let mut x_new = vec![0.0; n];
    for _ in 0..30 {
        for i in 0..n {
            x_new[i] = (x[i] + alpha * d[i]).clamp(bounds.lower[i], bounds.upper[i]);
        }
        let f_new = f_only(&x_new, prev_etas);
        if f_new <= f0 + c1 * alpha * dg {
            return alpha;
        }
        alpha *= 0.5;
        if alpha < 1e-18 {
            return 0.0;
        }
    }
    0.0
}

/// Compute covariance matrix via finite-difference Hessian at convergence.
pub(crate) fn compute_covariance(
    x_hat: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    kappas: &[Vec<DVector<f64>>],
    options: &FitOptions,
) -> Option<DMatrix<f64>> {
    let n = x_hat.len();
    let eps = 1e-2; // large step for FD Hessian on log-scale parameters

    // OFV for covariance step: includes explicit Omega terms (log|Omega| + eta'*Omega_inv*eta)
    // so the Hessian is sensitive to Omega parameters.
    // This matches Julia's foce_population_nll_diff.
    let ofv_fixed = |x: &[f64]| -> f64 {
        let params = unpack_params(x, template);
        let foce_nll = pop_nll(model, population, &params, eta_hats, h_matrices, kappas, options.interaction);

        // Add explicit Omega prior terms for each subject
        let n_subj = eta_hats.len();
        let n_eta = if n_subj > 0 { eta_hats[0].len() } else { 0 };

        let omega_inv = match params.omega.matrix.clone().cholesky() {
            Some(c) => c.inverse(),
            None => return 1e20,
        };
        let log_det_omega = {
            let mut ld = 0.0;
            for i in 0..n_eta {
                let lii = params.omega.chol[(i, i)];
                if lii > 0.0 {
                    ld += lii.ln();
                } else {
                    return 1e20;
                }
            }
            2.0 * ld
        };

        let mut omega_terms = 0.0;
        for eta in eta_hats {
            omega_terms += eta.dot(&(&omega_inv * eta)) + log_det_omega;
        }

        2.0 * foce_nll + omega_terms
    };

    let base_ofv = ofv_fixed(x_hat);
    if !base_ofv.is_finite() {
        if options.verbose {
            eprintln!("  Covariance failed: base OFV is non-finite");
        }
        return None;
    }

    // FIX parameters contribute no information — skip their FD stencils and,
    // after inverting the Hessian of the free block, leave their covariance
    // rows/cols at zero (→ SE = 0 downstream).
    let fixed_mask = packed_fixed_mask(template);
    let free_idx: Vec<usize> = (0..n).filter(|&i| !fixed_mask[i]).collect();

    let mut hess = DMatrix::zeros(n, n);
    let mut x_ij = x_hat.to_vec();

    let f0 = base_ofv;

    for &i in &free_idx {
        let hi = eps * (1.0 + x_hat[i].abs());

        // Diagonal: 3-point formula  (f(x+h) - 2f(x) + f(x-h)) / h^2
        x_ij[i] = x_hat[i] + hi;
        let fp = ofv_fixed(&x_ij);
        x_ij[i] = x_hat[i] - hi;
        let fm = ofv_fixed(&x_ij);
        x_ij[i] = x_hat[i];

        let h_ii = (fp - 2.0 * f0 + fm) / (hi * hi);
        if h_ii.is_finite() {
            hess[(i, i)] = h_ii;
        }

        // Off-diagonal: 4-point stencil (over free indices only)
        for &j in &free_idx {
            if j <= i {
                continue;
            }
            let hj = eps * (1.0 + x_hat[j].abs());

            x_ij[i] = x_hat[i] + hi;
            x_ij[j] = x_hat[j] + hj;
            let fpp = ofv_fixed(&x_ij);

            x_ij[j] = x_hat[j] - hj;
            let fpm = ofv_fixed(&x_ij);

            x_ij[i] = x_hat[i] - hi;
            let fmm = ofv_fixed(&x_ij);

            x_ij[j] = x_hat[j] + hj;
            let fmp = ofv_fixed(&x_ij);

            x_ij[i] = x_hat[i];
            x_ij[j] = x_hat[j];

            let h_ij = (fpp - fpm - fmp + fmm) / (4.0 * hi * hj);
            if h_ij.is_finite() {
                hess[(i, j)] = h_ij;
                hess[(j, i)] = h_ij;
            }
        }
    }

    // Check for non-finite or zero Hessian entries *in the free block*. Rows
    // and columns of FIX parameters are intentionally zero.
    let mut n_nonfinite = 0;
    let mut n_zero = 0;
    for &i in &free_idx {
        if hess[(i, i)].abs() < 1e-30 {
            n_zero += 1;
        }
        for &j in &free_idx {
            if !hess[(i, j)].is_finite() {
                n_nonfinite += 1;
            }
        }
    }

    if n_nonfinite > 0 || n_zero > 0 {
        if options.verbose {
            eprintln!(
                "  Covariance failed: Hessian has {} non-finite, {} zero-diagonal entries",
                n_nonfinite, n_zero
            );
        }
        return None;
    }

    // Build the reduced Hessian over free indices, invert, then embed back
    // into the full n×n covariance matrix (FIX rows/cols stay zero).
    let n_free = free_idx.len();
    if n_free == 0 {
        // Nothing to estimate — return an all-zero covariance so downstream
        // SE extraction reports zeros (all params FIX).
        return Some(DMatrix::zeros(n, n));
    }
    let mut hess_free = DMatrix::zeros(n_free, n_free);
    for (a, &i) in free_idx.iter().enumerate() {
        for (b, &j) in free_idx.iter().enumerate() {
            hess_free[(a, b)] = hess[(i, j)];
        }
    }
    let hess_free_sym = (&hess_free + hess_free.transpose()) * 0.5;
    match hess_free_sym.try_inverse() {
        Some(cov_free) => {
            let neg_diag: Vec<usize> = (0..n_free).filter(|&a| cov_free[(a, a)] <= 0.0).collect();
            if !neg_diag.is_empty() {
                if options.verbose {
                    eprintln!(
                        "  Covariance failed: negative diagonal in free-block at {:?}",
                        neg_diag
                    );
                }
                return None;
            }
            let mut cov = DMatrix::zeros(n, n);
            for (a, &i) in free_idx.iter().enumerate() {
                for (b, &j) in free_idx.iter().enumerate() {
                    cov[(i, j)] = cov_free[(a, b)];
                }
            }
            if options.verbose {
                eprintln!("  Covariance step successful");
            }
            Some(cov)
        }
        None => {
            if options.verbose {
                eprintln!("  Covariance failed: Hessian not invertible");
            }
            None
        }
    }
}
