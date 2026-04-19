/// Gauss-Newton optimizer for FOCE estimation.
///
/// Instead of the standard approach (separate inner/outer loops with first-order
/// gradient methods), this uses a coupled Gauss-Newton step that exploits the
/// nonlinear-least-squares structure of the FOCE objective:
///
///   OFV = sum_i [ r_i^T W_i^{-1} r_i + log|W_i| ]
///
/// where r_i are the weighted residuals and W_i = R_tilde_i is the linearized
/// covariance for subject i. The Gauss-Newton approximation uses J^T W^{-1} J
/// as the approximate Hessian (dropping second-derivative terms), giving
/// quadratic convergence near the minimum.
///
/// This approach mirrors NONMEM's modified Gauss-Newton algorithm and typically
/// converges in 10-30 iterations vs 100+ for first-order methods.
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::{
    foce_population_nll, foce_subject_nll_interaction, foce_subject_nll_standard,
};
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Run FOCE estimation using a Gauss-Newton optimizer.
///
/// Returns the same `OuterResult` as `optimize_population`.
pub fn run_foce_gn(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let n_subj = population.subjects.len();
    let _n_eta = model.n_eta;
    let verbose = options.verbose;
    let maxiter = options.outer_maxiter;
    let mut lambda = options.gn_lambda; // LM damping factor

    let bounds = compute_bounds(init_params);
    let mut x = pack_params(init_params);
    clamp_to_bounds(&mut x, &bounds);
    let n_packed = x.len();

    let mut warnings = Vec::new();

    // BHHH Information-matrix approximation degrades as the censoring fraction
    // grows — each BLOQ row contributes less Fisher information than its
    // Gaussian counterpart, biasing the outer-product Hessian small-sample.
    if matches!(model.bloq_method, BloqMethod::M3)
        && population.subjects.iter().any(|s| s.has_bloq())
    {
        warnings.push(
            "Gauss-Newton (BHHH) approximation may be inaccurate with M3 BLOQ handling; \
             consider method=foce_i for heavy BLOQ fractions (>20%)."
                .to_string(),
        );
    }

    if verbose {
        eprintln!("Starting FOCE Gauss-Newton estimation...");
        eprintln!("  {} subjects, {} observations", n_subj, population.n_obs());
        eprintln!("  {} packed parameters, lambda={:.4}", n_packed, lambda);
    }

    // Initial inner loop
    let params = unpack_params(&x, init_params);
    let init_mu_k = compute_mu_k(model, &params.theta, options.mu_referencing);
    let (mut eta_hats, mut h_matrices, _) = run_inner_loop_warm(
        model,
        population,
        &params,
        options.inner_maxiter,
        options.inner_tol,
        None,
        Some(&init_mu_k),
    );

    let mut ofv = 2.0
        * foce_population_nll(
            model,
            population,
            &params.theta,
            &eta_hats,
            &h_matrices,
            &params.omega,
            &params.sigma.values,
            options.interaction,
        );

    if verbose {
        eprintln!("  GN iter {:>3}: OFV = {:.6}", 0, ofv);
    }

    let mut converged = false;

    for iter in 1..=maxiter {
        let params = unpack_params(&x, init_params);

        // ---- Build the BHHH system ----
        // Gradient + outer-product Hessian approximation
        let (grad, h_bhhh) = build_gn_system(
            &x,
            init_params,
            model,
            population,
            &eta_hats,
            &h_matrices,
            &bounds,
            options,
        );

        // ---- Levenberg-Marquardt damping ----
        let mut h_lm = h_bhhh.clone();
        for i in 0..n_packed {
            h_lm[(i, i)] += lambda * h_bhhh[(i, i)].max(1e-8);
        }

        // ---- Solve for step: H_lm * delta = -grad ----
        let neg_grad = -&grad;
        let chol = h_lm.clone().cholesky();
        let delta = match chol {
            Some(c) => c.solve(&neg_grad),
            None => {
                // Fall back to regularized pseudo-inverse
                if verbose {
                    eprintln!("  GN iter {:>3}: Hessian singular, increasing lambda", iter);
                }
                lambda *= 10.0;
                continue;
            }
        };

        // ---- Line search with backtracking ----
        let mut alpha = 1.0;
        let mut x_new = x.clone();
        let mut ofv_new = f64::INFINITY;
        let mut eta_new = eta_hats.clone();
        let mut h_new = h_matrices.clone();

        for _ls in 0..15 {
            // Take step
            for i in 0..n_packed {
                x_new[i] = (x[i] + alpha * delta[i]).clamp(bounds.lower[i], bounds.upper[i]);
            }

            let params_try = unpack_params(&x_new, init_params);

            // Re-estimate EBEs at new parameters (warm-started)
            let ls_mu_k = compute_mu_k(model, &params_try.theta, options.mu_referencing);
            let (eh, hm, _) = run_inner_loop_warm(
                model,
                population,
                &params_try,
                options.inner_maxiter,
                options.inner_tol,
                Some(&eta_new),
                Some(&ls_mu_k),
            );

            let nll = foce_population_nll(
                model,
                population,
                &params_try.theta,
                &eh,
                &hm,
                &params_try.omega,
                &params_try.sigma.values,
                options.interaction,
            );
            let ofv_try = 2.0 * nll;

            if ofv_try.is_finite() && ofv_try < ofv {
                ofv_new = ofv_try;
                eta_new = eh;
                h_new = hm;
                break;
            }

            alpha *= 0.5;
        }

        if ofv_new >= ofv {
            // Step failed — increase damping and retry
            lambda *= 10.0;
            if lambda > 1e6 {
                if verbose {
                    eprintln!("  GN iter {:>3}: lambda too large, stopping", iter);
                }
                warnings.push("Gauss-Newton: lambda exceeded threshold".to_string());
                break;
            }
            if verbose {
                eprintln!(
                    "  GN iter {:>3}: step rejected, lambda -> {:.4}",
                    iter, lambda
                );
            }
            continue;
        }

        // ---- Accept step ----
        let ofv_change = (ofv - ofv_new).abs();
        let rel_change = ofv_change / ofv.abs().max(1.0);

        x = x_new;
        ofv = ofv_new;
        eta_hats = eta_new;
        h_matrices = h_new;

        // Decrease damping on success
        lambda = (lambda * 0.3).max(1e-6);

        if verbose {
            eprintln!(
                "  GN iter {:>3}: OFV = {:.6}  (delta={:.2e}, lambda={:.4})",
                iter, ofv, ofv_change, lambda
            );
        }

        // Check convergence
        if rel_change < 1e-6 && iter > 3 {
            converged = true;
            if verbose {
                eprintln!("  Converged: relative OFV change = {:.2e}", rel_change);
            }
            break;
        }
    }

    if !converged {
        warnings.push("Gauss-Newton: max iterations reached without convergence".to_string());
    }

    let gn_ofv = ofv;
    let do_polish = matches!(options.method, EstimationMethod::FoceGnHybrid);

    // ---- Optional hybrid: polish with FOCEI from GN result ----
    if do_polish && verbose {
        eprintln!("GN phase done (OFV={:.4}). Polishing with FOCEI...", ofv);
    }

    let gn_params = unpack_params(&x, init_params);

    if !do_polish {
        // Pure GN — skip FOCEI polish, go directly to covariance step
        let covariance_matrix = if options.run_covariance_step {
            if verbose {
                eprintln!("Running covariance step...");
            }
            let cov = compute_covariance(
                &x,
                &gn_params,
                model,
                population,
                &eta_hats,
                &h_matrices,
                options,
            );
            if cov.is_none() {
                warnings.push("Covariance step failed".to_string());
            }
            cov
        } else {
            None
        };

        if verbose {
            eprintln!("FOCE-GN completed. Final OFV = {:.4}", ofv);
        }

        return OuterResult {
            params: gn_params,
            ofv,
            converged,
            n_iterations: maxiter,
            eta_hats,
            h_matrices,
            covariance_matrix,
            warnings,
        };
    }

    // Build FitOptions for the FOCEI polish: short maxiter, warm-started from GN
    let mut polish_options = options.clone();
    polish_options.method = EstimationMethod::Foce;
    polish_options.outer_maxiter = 100; // short polish
    polish_options.global_search = false;
    polish_options.run_covariance_step = false; // defer to after polish

    let polish_result = crate::estimation::outer_optimizer::optimize_population_warm(
        model,
        population,
        &gn_params,
        &polish_options,
        &eta_hats,
        &h_matrices,
    );

    let final_ofv;
    let final_params;
    let final_etas;
    let final_h_mats;

    if polish_result.ofv < gn_ofv {
        if verbose {
            eprintln!(
                "  FOCEI polish improved OFV: {:.4} -> {:.4}",
                gn_ofv, polish_result.ofv
            );
        }
        final_ofv = polish_result.ofv;
        final_params = polish_result.params;
        final_etas = polish_result.eta_hats;
        final_h_mats = polish_result.h_matrices;
        converged = polish_result.converged || converged;
    } else {
        if verbose {
            eprintln!("  FOCEI polish did not improve (GN result kept)");
        }
        final_ofv = gn_ofv;
        final_params = gn_params;
        final_etas = eta_hats;
        final_h_mats = h_matrices;
    }

    // ---- Covariance step ----
    let covariance_matrix = if options.run_covariance_step {
        if verbose {
            eprintln!("Running covariance step...");
        }
        let packed = pack_params(&final_params);
        let cov = compute_covariance(
            &packed,
            &final_params,
            model,
            population,
            &final_etas,
            &final_h_mats,
            options,
        );
        if cov.is_none() {
            warnings.push("Covariance step failed".to_string());
        }
        cov
    } else {
        None
    };

    if verbose {
        eprintln!("FOCE-GN completed. Final OFV = {:.4}", final_ofv);
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: maxiter,
        eta_hats: final_etas,
        h_matrices: final_h_mats,
        covariance_matrix,
        warnings,
    }
}

/// Build the Gauss-Newton linear system using the gradient and approximate Hessian
/// of the FOCE population objective.
///
/// The gradient is computed via central FD of the total OFV w.r.t. packed params.
/// The approximate Hessian uses the outer product of per-subject gradients (BHHH):
///   H_bhhh = sum_i g_i g_i^T
/// where g_i = d(2*nll_i)/d(x) is the per-subject OFV gradient.
///
/// This is the Berndt-Hall-Hall-Hausman (BHHH) approximation, which is equivalent
/// to Gauss-Newton for the FOCE log-likelihood and is what NONMEM uses internally.
fn build_gn_system(
    x: &[f64],
    template: &ModelParameters,
    model: &CompiledModel,
    population: &Population,
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    bounds: &PackedBounds,
    options: &FitOptions,
) -> (DVector<f64>, DMatrix<f64>) {
    let n = x.len();
    let n_subj = population.subjects.len();

    // Compute per-subject NLL at current point
    let params = unpack_params(x, template);
    let _nll_base: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, _)| {
            subject_nll_at(
                model,
                population,
                i,
                &params,
                &eta_hats[i],
                &h_matrices[i],
                options,
            )
        })
        .collect();

    // Compute per-subject gradient via central FD
    // g_i[j] = d(nll_i)/d(x_j) for each subject i, parameter j
    let eps = 1e-4;
    let mut per_subj_grad: Vec<Vec<f64>> = vec![vec![0.0; n]; n_subj];
    let mut x_work = x.to_vec();

    for j in 0..n {
        let h = eps * (1.0 + x[j].abs());
        let xj_plus = (x[j] + h).min(bounds.upper[j]);
        let xj_minus = (x[j] - h).max(bounds.lower[j]);
        let actual_2h = xj_plus - xj_minus;
        if actual_2h.abs() < 1e-16 {
            continue;
        }

        x_work[j] = xj_plus;
        let params_plus = unpack_params(&x_work, template);
        let nll_plus: Vec<f64> = population
            .subjects
            .iter()
            .enumerate()
            .map(|(i, _)| {
                subject_nll_at(
                    model,
                    population,
                    i,
                    &params_plus,
                    &eta_hats[i],
                    &h_matrices[i],
                    options,
                )
            })
            .collect();

        x_work[j] = xj_minus;
        let params_minus = unpack_params(&x_work, template);
        let nll_minus: Vec<f64> = population
            .subjects
            .iter()
            .enumerate()
            .map(|(i, _)| {
                subject_nll_at(
                    model,
                    population,
                    i,
                    &params_minus,
                    &eta_hats[i],
                    &h_matrices[i],
                    options,
                )
            })
            .collect();

        x_work[j] = x[j];

        for i in 0..n_subj {
            let deriv = (nll_plus[i] - nll_minus[i]) / actual_2h;
            per_subj_grad[i][j] = if deriv.is_finite() { deriv } else { 0.0 };
        }
    }

    // Total gradient: g = sum_i g_i (scaled by 2 for OFV = 2*NLL)
    let mut grad = DVector::zeros(n);
    for i in 0..n_subj {
        for j in 0..n {
            grad[j] += 2.0 * per_subj_grad[i][j];
        }
    }

    // BHHH approximate Hessian: H = sum_i (2*g_i)(2*g_i)^T = 4 * sum_i g_i g_i^T
    // But for the Newton step H*delta = -grad, we can factor out the 4:
    // Use H_bhhh = sum_i g_i g_i^T, and solve (H_bhhh * delta) = -(grad/4)...
    // Actually, let's just use the properly scaled version.
    //
    // For OFV = 2 * sum_i nll_i:
    //   grad(OFV) = 2 * sum_i grad_i
    //   H_bhhh(OFV) ≈ 4 * sum_i grad_i grad_i^T
    //
    // Newton step: delta = -H^{-1} grad = -(4 sum g_i g_i^T)^{-1} (2 sum g_i)
    //            = -0.5 * (sum g_i g_i^T)^{-1} (sum g_i)
    //
    // We return (grad_total, H_total) where grad_total = 2*sum(g_i) and
    // H_total = 4*sum(g_i g_i^T) so the caller solves H*delta = -grad directly.

    let mut h_bhhh = DMatrix::zeros(n, n);
    for i in 0..n_subj {
        let gi = DVector::from_column_slice(&per_subj_grad[i]);
        h_bhhh += 4.0 * &gi * gi.transpose();
    }

    (grad, h_bhhh)
}

/// Compute FOCE NLL for a single subject at given parameters with fixed EBEs.
///
/// Delegates to the canonical `foce_subject_nll_{standard,interaction}` in
/// `stats::likelihood` so M3 BLOQ support is single-sourced — do not re-inline.
fn subject_nll_at(
    model: &CompiledModel,
    population: &Population,
    subj_idx: usize,
    params: &ModelParameters,
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    options: &FitOptions,
) -> f64 {
    let subject = &population.subjects[subj_idx];
    let pk_params = (model.pk_param_fn)(&params.theta, eta_hat.as_slice(), &subject.covariates);
    let ipreds = if let Some(ref ode_spec) = model.ode_spec {
        crate::pk::compute_predictions_ode(ode_spec, subject, &pk_params.values)
    } else {
        crate::pk::compute_predictions(model.pk_model, subject, &pk_params)
    };

    // M3 forces FOCEI for that subject (mirrors the dispatcher in
    // `stats::likelihood::foce_subject_nll`).
    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();

    if options.interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            &params.omega,
            &params.sigma.values,
            model.error_model,
            model.bloq_method,
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            &params.omega,
            &params.sigma.values,
            model.error_model,
            model.bloq_method,
        )
    }
}
