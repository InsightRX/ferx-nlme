use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::foce_population_nll;
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Result of outer optimization
pub struct OuterResult {
    pub params: ModelParameters,
    pub ofv: f64,
    pub converged: bool,
    pub n_iterations: usize,
    pub eta_hats: Vec<DVector<f64>>,
    pub h_matrices: Vec<DMatrix<f64>>,
    pub covariance_matrix: Option<DMatrix<f64>>,
    pub warnings: Vec<String>,
}

/// Run the outer optimization loop (population parameter estimation).
pub fn optimize_population(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    match options.optimizer {
        Optimizer::Slsqp | Optimizer::NloptLbfgs | Optimizer::Mma => {
            optimize_nlopt(model, population, init_params, options)
        }
        Optimizer::Bfgs | Optimizer::Lbfgs => {
            optimize_bfgs(model, population, init_params, options)
        }
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

/// State passed through NLopt's user-data mechanism
struct NloptState {
    cached_etas: Vec<DVector<f64>>,
    cached_h_mats: Vec<DMatrix<f64>>,
    best_ofv: f64,
    n_evals: usize,
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

    let state = NloptState {
        cached_etas: vec![DVector::zeros(n_eta); n_subj],
        cached_h_mats: Vec::new(),
        best_ofv: f64::INFINITY,
        n_evals: 0,
    };

    // Select NLopt algorithm
    let algo = match options.optimizer {
        Optimizer::Slsqp => nlopt::Algorithm::Slsqp,
        Optimizer::NloptLbfgs => nlopt::Algorithm::Lbfgs,
        Optimizer::Mma => nlopt::Algorithm::Mma,
        _ => nlopt::Algorithm::Slsqp,
    };

    let verbose = options.verbose;

    // NLopt objective: runs inner loop, computes gradient with fixed EBEs
    let objective = |x: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);

        // Run inner loop (warm-started)
        let (ehs, hms, _) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(&state.cached_etas),
            Some(&mu_k),
        );

        // Compute OFV with fixed EBEs
        let nll = foce_population_nll(
            model,
            population,
            &params.theta,
            &ehs,
            &hms,
            &params.omega,
            &params.sigma.values,
            options.interaction,
        );
        let raw_ofv = 2.0 * nll;
        let ofv = if raw_ofv.is_finite() { raw_ofv } else { 1e20 };

        // Compute gradient if requested (central FD with fixed EBEs)
        if let Some(g) = grad {
            // If OFV is non-finite, gradient is meaningless — use steepest ascent
            // toward center of bounds to nudge optimizer back
            if !raw_ofv.is_finite() {
                for i in 0..g.len() {
                    let center = (bounds.lower[i] + bounds.upper[i]) / 2.0;
                    g[i] = 100.0 * (x[i] - center); // gradient points away from center
                }
                state.n_evals += 1;
                return ofv;
            }
            let ofv_fn = |xp: &[f64], eh: &[DVector<f64>], hm: &[DMatrix<f64>]| -> f64 {
                let p = unpack_params(xp, init_params);
                2.0 * foce_population_nll(
                    model,
                    population,
                    &p.theta,
                    eh,
                    hm,
                    &p.omega,
                    &p.sigma.values,
                    options.interaction,
                )
            };
            let grad_vec = gradient_cd(x, &bounds, &ehs, &hms, &ofv_fn);
            for i in 0..g.len() {
                g[i] = if grad_vec[i].is_finite() {
                    grad_vec[i]
                } else {
                    0.0
                };
            }
        }

        // Update state
        state.cached_etas = ehs;
        state.cached_h_mats = hms;
        state.n_evals += 1;
        if ofv < state.best_ofv {
            state.best_ofv = ofv;
            if verbose {
                eprintln!("Eval {:>4}: OFV = {:.6}", state.n_evals, ofv);
            }
        }

        ofv
    };

    // Create NLopt optimizer with state
    let mut opt = nlopt::Nlopt::new(algo, n, objective, nlopt::Target::Minimize, state);
    opt.set_lower_bounds(&bounds.lower).unwrap();
    opt.set_upper_bounds(&bounds.upper).unwrap();
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

    // Fallback: if L-BFGS failed, retry with SLSQP from current best point
    let already_slsqp = matches!(first_algo, nlopt::Algorithm::Slsqp);
    if !converged && !already_slsqp {
        if options.verbose {
            eprintln!("Retrying with NLopt SLSQP from current point...");
        }

        let state2 = NloptState {
            cached_etas: vec![DVector::zeros(n_eta); n_subj],
            cached_h_mats: Vec::new(),
            best_ofv: f64::INFINITY,
            n_evals: 0,
        };

        let objective2 = |x: &[f64], grad: Option<&mut [f64]>, state: &mut NloptState| -> f64 {
            let params = unpack_params(x, init_params);
            let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
            let (ehs, hms, _) = run_inner_loop_warm(
                model,
                population,
                &params,
                options.inner_maxiter,
                options.inner_tol,
                Some(&state.cached_etas),
                Some(&mu_k),
            );
            let nll = foce_population_nll(
                model,
                population,
                &params.theta,
                &ehs,
                &hms,
                &params.omega,
                &params.sigma.values,
                options.interaction,
            );
            let raw_ofv = 2.0 * nll;
            let ofv = if raw_ofv.is_finite() { raw_ofv } else { 1e20 };

            if let Some(g) = grad {
                if !raw_ofv.is_finite() {
                    for i in 0..g.len() {
                        let center = (bounds.lower[i] + bounds.upper[i]) / 2.0;
                        g[i] = 100.0 * (x[i] - center);
                    }
                    state.n_evals += 1;
                    return ofv;
                }
                let ofv_fn = |xp: &[f64], eh: &[DVector<f64>], hm: &[DMatrix<f64>]| -> f64 {
                    let p = unpack_params(xp, init_params);
                    2.0 * foce_population_nll(
                        model,
                        population,
                        &p.theta,
                        eh,
                        hm,
                        &p.omega,
                        &p.sigma.values,
                        options.interaction,
                    )
                };
                let grad_vec = gradient_cd(x, &bounds, &ehs, &hms, &ofv_fn);
                for i in 0..g.len() {
                    g[i] = if grad_vec[i].is_finite() {
                        grad_vec[i]
                    } else {
                        0.0
                    };
                }
            }

            state.cached_etas = ehs;
            state.cached_h_mats = hms;
            state.n_evals += 1;
            if ofv < state.best_ofv {
                state.best_ofv = ofv;
                if verbose {
                    eprintln!("Eval {:>4}: OFV = {:.6} (SLSQP)", state.n_evals, ofv);
                }
            }
            ofv
        };

        let mut opt2 = nlopt::Nlopt::new(
            nlopt::Algorithm::Slsqp,
            n,
            objective2,
            nlopt::Target::Minimize,
            state2,
        );
        opt2.set_lower_bounds(&bounds.lower).unwrap();
        opt2.set_upper_bounds(&bounds.upper).unwrap();
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

    let final_params = unpack_params(&x0, init_params);
    let final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);

    // Final inner loop at converged parameters
    let (final_ehs, final_hms, _) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&final_mu_k),
    );

    let final_nll = foce_population_nll(
        model,
        population,
        &final_params.theta,
        &final_ehs,
        &final_hms,
        &final_params.omega,
        &final_params.sigma.values,
        options.interaction,
    );
    let final_ofv = 2.0 * final_nll;

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    // Covariance step
    let covariance_matrix = if options.run_covariance_step {
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

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: 0, // NLopt doesn't expose iteration count directly
        eta_hats: final_ehs,
        h_matrices: final_hms,
        covariance_matrix,
        warnings,
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

    let ofv_at_fixed = |x: &[f64], eta_hats: &[DVector<f64>], h_matrices: &[DMatrix<f64>]| -> f64 {
        let params = unpack_params(x, init_params);
        2.0 * foce_population_nll(
            model,
            population,
            &params.theta,
            eta_hats,
            h_matrices,
            &params.omega,
            &params.sigma.values,
            options.interaction,
        )
    };

    let f_only = |x: &[f64], prev_etas: &[DVector<f64>]| -> f64 {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
        );
        let ofv = 2.0
            * foce_population_nll(
                model,
                population,
                &params.theta,
                &ehs,
                &hms,
                &params.omega,
                &params.sigma.values,
                options.interaction,
            );
        if ofv.is_finite() {
            ofv
        } else {
            1e20
        }
    };

    let fdfg = |x: &[f64],
                prev_etas: &[DVector<f64>]|
     -> (f64, Vec<f64>, Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let params = unpack_params(x, init_params);
        let mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
        let (ehs, hms, _) = run_inner_loop_warm(
            model,
            population,
            &params,
            options.inner_maxiter,
            options.inner_tol,
            Some(prev_etas),
            Some(&mu_k),
        );
        let ofv = ofv_at_fixed(x, &ehs, &hms);
        let g = gradient_cd(x, &bounds, &ehs, &hms, &ofv_at_fixed);
        let f = if ofv.is_finite() { ofv } else { 1e20 };
        (f, g, ehs, hms)
    };

    let (mut f_val, mut g, ehs, _) = fdfg(&x, &cached_etas);
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
            backtracking_line_search_warm(&x, &d, &g, f_val, &bounds, &cached_etas, &f_only);

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

        let x_old = x.clone();
        for i in 0..n {
            x[i] = (x[i] + alpha * d[i]).clamp(bounds.lower[i], bounds.upper[i]);
        }

        let (f_new, g_new, ehs, _) = fdfg(&x, &cached_etas);
        cached_etas = ehs;

        bfgs_update(&mut h_inv, &x, &x_old, &g_new, &g, n);

        let prev_ofv = f_val;
        f_val = f_new;
        g = g_new;

        if options.verbose && (iter % 10 == 0 || iter <= 5) {
            eprintln!(
                "Iter {:>4}: OFV = {:.6}  |g| = {:.2e}  alpha = {:.2e}",
                iter, f_val, g_norm, alpha
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

    let final_params = unpack_params(&x, init_params);
    let bfgs_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (final_ehs, final_hms, _) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&cached_etas),
        Some(&bfgs_final_mu_k),
    );
    let final_ofv = ofv_at_fixed(&x, &final_ehs, &final_hms);

    let covariance_matrix = if options.run_covariance_step {
        if options.verbose {
            eprintln!("Computing covariance matrix...");
        }
        compute_covariance(
            &x,
            init_params,
            model,
            population,
            &final_ehs,
            &final_hms,
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

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        covariance_matrix,
        warnings,
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
    options: &FitOptions,
) -> Option<DMatrix<f64>> {
    let n = x_hat.len();
    let eps = 1e-2; // large step for FD Hessian on log-scale parameters

    // OFV for covariance step: includes explicit Omega terms (log|Omega| + eta'*Omega_inv*eta)
    // so the Hessian is sensitive to Omega parameters.
    // This matches Julia's foce_population_nll_diff.
    let ofv_fixed = |x: &[f64]| -> f64 {
        let params = unpack_params(x, template);
        let foce_nll = foce_population_nll(
            model,
            population,
            &params.theta,
            eta_hats,
            h_matrices,
            &params.omega,
            &params.sigma.values,
            options.interaction,
        );

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

    let mut hess = DMatrix::zeros(n, n);
    let mut x_ij = x_hat.to_vec();

    let f0 = base_ofv;

    for i in 0..n {
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

        // Off-diagonal: 4-point stencil
        for j in (i + 1)..n {
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

    // Check for non-finite or zero Hessian entries
    let mut n_nonfinite = 0;
    let mut n_zero = 0;
    for i in 0..n {
        if hess[(i, i)].abs() < 1e-30 {
            n_zero += 1;
        }
        for j in 0..n {
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

    let hess_sym = (&hess + hess.transpose()) * 0.5;
    match hess_sym.try_inverse() {
        Some(cov) => {
            let neg_diag: Vec<usize> = (0..n).filter(|&i| cov[(i, i)] <= 0.0).collect();
            if neg_diag.is_empty() {
                if options.verbose {
                    eprintln!("  Covariance step successful");
                }
                Some(cov)
            } else {
                if options.verbose {
                    eprintln!(
                        "  Covariance failed: negative diagonal at indices {:?}",
                        neg_diag
                    );
                }
                None
            }
        }
        None => {
            if options.verbose {
                eprintln!("  Covariance failed: Hessian not invertible");
            }
            None
        }
    }
}
