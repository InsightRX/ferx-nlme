/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, OuterResult};
use crate::estimation::parameterization::*;
use crate::stats::likelihood::{foce_population_nll, individual_nll};
use crate::stats::residual_error::residual_variance;
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// SAEM state
// ---------------------------------------------------------------------------

struct SaemState {
    /// Per-subject current ETAs
    etas: Vec<Vec<f64>>,
    /// Cached individual NLL at current ETAs
    nll_cache: Vec<f64>,
    /// Per-subject MH step sizes
    step_scales: Vec<f64>,
    /// Per-subject acceptance counts since last adaptation
    accept_counts: Vec<usize>,
    /// Steps since last adaptation
    steps_since_adapt: usize,
    /// SA sufficient statistic for Omega: running average of (1/N) Σ ηᵢηᵢᵀ
    s2: DMatrix<f64>,
    /// Current theta
    theta: Vec<f64>,
    /// Current omega matrix
    omega_mat: DMatrix<f64>,
    /// Current sigma values
    sigma_vals: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Metropolis-Hastings step for one subject
// ---------------------------------------------------------------------------

/// Run `n_steps` MH iterations for one subject in-place.
/// Returns (n_accepted, updated_nll).
fn mh_steps(
    eta: &mut [f64],
    nll_current: f64,
    subject: &Subject,
    model: &CompiledModel,
    theta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    step_scale: f64,
    rng: &mut impl Rng,
    n_steps: usize,
) -> (usize, f64) {
    let n_eta = eta.len();
    let l = &omega.chol;
    let mut nll = nll_current;
    let mut n_accepted = 0;

    for _ in 0..n_steps {
        // Propose: eta_prop = eta + step_scale * L * z, z ~ N(0,I)
        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let eta_prop: Vec<f64> = (0..n_eta)
            .map(|j| eta[j] + step_scale * perturbation[j])
            .collect();

        let nll_prop = individual_nll(model, subject, theta, &eta_prop, omega, sigma_values);

        // Accept with log-ratio: log(alpha) = nll_current - nll_prop
        let log_u: f64 = rng.gen::<f64>().ln();
        if log_u < nll - nll_prop {
            eta.copy_from_slice(&eta_prop);
            nll = nll_prop;
            n_accepted += 1;
        }
    }

    (n_accepted, nll)
}

// ---------------------------------------------------------------------------
// Gradient of conditional observation NLL w.r.t. log(theta) and log(sigma)
// ---------------------------------------------------------------------------

/// Lightweight M-step: run NLopt SLSQP for a few iterations in log-space,
/// warm-started from the current log-theta/log-sigma.
fn theta_sigma_mstep_light(
    model: &CompiledModel,
    population: &Population,
    etas: &[Vec<f64>],
    log_theta_init: &[f64],
    log_sigma_init: &[f64],
    log_theta_lower: &[f64],
    log_theta_upper: &[f64],
    log_sigma_lower: &[f64],
    log_sigma_upper: &[f64],
    n_theta: usize,
    n_sigma: usize,
    maxiter: u32,
) -> (Vec<f64>, Vec<f64>) {
    let n = n_theta + n_sigma;

    let mut x: Vec<f64> = Vec::with_capacity(n);
    x.extend_from_slice(log_theta_init);
    x.extend_from_slice(log_sigma_init);

    let mut lower: Vec<f64> = Vec::with_capacity(n);
    lower.extend_from_slice(log_theta_lower);
    lower.extend_from_slice(log_sigma_lower);
    let mut upper: Vec<f64> = Vec::with_capacity(n);
    upper.extend_from_slice(log_theta_upper);
    upper.extend_from_slice(log_sigma_upper);

    for i in 0..n {
        x[i] = x[i].clamp(lower[i], upper[i]);
    }

    let obj = |xv: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = xv[..n_theta].iter().map(|&v| v.exp()).collect();
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let val = obs_nll_sum(model, population, &th, &sg, etas);

        if let Some(g) = grad {
            let h = 1e-5;
            let mut xp = xv.to_vec();
            for i in 0..n {
                xp[i] = xv[i] + h;
                let th_p: Vec<f64> = xp[..n_theta].iter().map(|&v| v.exp()).collect();
                let sg_p: Vec<f64> = xp[n_theta..].iter().map(|&v| v.exp()).collect();
                let fp = obs_nll_sum(model, population, &th_p, &sg_p, etas);
                xp[i] = xv[i] - h;
                let th_m: Vec<f64> = xp[..n_theta].iter().map(|&v| v.exp()).collect();
                let sg_m: Vec<f64> = xp[n_theta..].iter().map(|&v| v.exp()).collect();
                let fm = obs_nll_sum(model, population, &th_m, &sg_m, etas);
                g[i] = (fp - fm) / (2.0 * h);
                if !g[i].is_finite() {
                    g[i] = 0.0;
                }
                xp[i] = xv[i];
            }
        }

        if val.is_finite() {
            val
        } else {
            1e20
        }
    };

    let mut opt = nlopt::Nlopt::new(nlopt::Algorithm::Slsqp, n, obj, nlopt::Target::Minimize, ());
    opt.set_lower_bounds(&lower).unwrap();
    opt.set_upper_bounds(&upper).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(1e-4).unwrap();

    match opt.optimize(&mut x) {
        Ok(_) | Err(_) => {}
    }

    let log_theta_new = x[..n_theta].to_vec();
    let log_sigma_new = x[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}

/// Sum of observation log-likelihoods with ETAs held fixed.
///
/// Under M3, CENS=1 rows contribute `-log Φ((LLOQ - f)/√V)` instead of the
/// Gaussian residual term. Without this branch, the SAEM M-step would optimize
/// θ/σ as if censored observations were exact Gaussians at the LLOQ value,
/// producing silently-biased population estimates.
fn obs_nll_sum(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    sigma_values: &[f64],
    etas: &[Vec<f64>],
) -> f64 {
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            let pk_params = (model.pk_param_fn)(theta, &etas[i], &subject.covariates);
            let preds = if let Some(ref ode_spec) = model.ode_spec {
                crate::pk::compute_predictions_ode(ode_spec, subject, &pk_params.values)
            } else {
                crate::pk::compute_predictions(model.pk_model, subject, &pk_params)
            };
            let mut nll = 0.0;
            for (j, (&y, &f)) in subject.observations.iter().zip(preds.iter()).enumerate() {
                let f = f.max(1e-12);
                let v = residual_variance(model.error_model, f, sigma_values).max(1e-12);
                if m3 && subject.cens.get(j).copied().unwrap_or(0) != 0 {
                    let z = (y - f) / v.sqrt();
                    nll += -log_normal_cdf(z);
                } else {
                    nll += 0.5 * (v.ln() + (y - f).powi(2) / v);
                }
            }
            nll
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Main SAEM loop
// ---------------------------------------------------------------------------

pub fn run_saem(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> Result<OuterResult, String> {
    let n_subjects = population.subjects.len();
    let n_eta = model.n_eta;
    let k1 = options.saem_n_exploration;
    let k2 = options.saem_n_convergence;
    let n_iter = k1 + k2;
    let n_mh_steps = options.saem_n_mh_steps;
    let adapt_interval = options.saem_adapt_interval;
    let verbose = options.verbose;

    let n_theta = init_params.theta.len();
    let n_sigma = init_params.sigma.values.len();

    // Master RNG
    let master_seed = options.saem_seed.unwrap_or(12345);

    if verbose {
        eprintln!(
            "SAEM: {} subjects, {} ETAs, {} total iter ({} explore + {} converge)",
            n_subjects, n_eta, n_iter, k1, k2
        );
    }

    // Initialize state
    let theta_cur = init_params.theta.clone();
    let omega_cur = init_params.omega.matrix.clone();
    let sigma_cur = init_params.sigma.values.clone();
    let s2 = omega_cur.clone();

    let etas: Vec<Vec<f64>> = vec![vec![0.0; n_eta]; n_subjects];
    let step_scales = vec![0.3; n_subjects];

    // Initial NLL cache
    let nll_cache: Vec<f64> = population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            individual_nll(
                model,
                subject,
                &theta_cur,
                &etas[i],
                &init_params.omega,
                &sigma_cur,
            )
        })
        .collect();

    // Pack initial log-theta and log-sigma for SA updates
    let mut log_theta: Vec<f64> = theta_cur.iter().map(|&t| t.max(1e-10).ln()).collect();
    let mut log_sigma: Vec<f64> = sigma_cur.iter().map(|&s| s.max(1e-10).ln()).collect();

    // Bounds in log-space
    let log_theta_lower: Vec<f64> = init_params
        .theta_lower
        .iter()
        .map(|&b| b.max(1e-10).ln())
        .collect();
    let log_theta_upper: Vec<f64> = init_params
        .theta_upper
        .iter()
        .map(|&b| b.min(1e9).ln())
        .collect();
    let log_sigma_lower = vec![-8.0f64; n_sigma];
    let log_sigma_upper = vec![5.0f64; n_sigma];

    let mut state = SaemState {
        etas,
        nll_cache,
        step_scales,
        accept_counts: vec![0; n_subjects],
        steps_since_adapt: 0,
        s2,
        theta: theta_cur,
        omega_mat: omega_cur,
        sigma_vals: sigma_cur,
    };

    // Main loop
    for k in 1..=n_iter {
        if crate::cancel::is_cancelled(&options.cancel) {
            if verbose {
                eprintln!("SAEM: cancelled at iteration {}", k);
            }
            break;
        }
        let gamma = if k <= k1 { 1.0 } else { 1.0 / (k - k1) as f64 };

        // Rebuild omega for this iteration
        let omega_k = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );

        // ---- Step 1: MH simulation (parallelized) ----
        {
            use rayon::prelude::*;
            let theta_ref = &state.theta;
            let sigma_ref = &state.sigma_vals;
            let omega_ref = &omega_k;

            let results: Vec<(Vec<f64>, f64, usize)> = state
                .etas
                .par_iter()
                .zip(state.nll_cache.par_iter())
                .zip(state.step_scales.par_iter())
                .enumerate()
                .map(|(i, ((eta, &nll), &scale))| {
                    let subject = &population.subjects[i];
                    let mut rng = StdRng::seed_from_u64(
                        master_seed
                            .wrapping_add(k as u64 * 100_000)
                            .wrapping_add(i as u64),
                    );
                    let mut eta_work = eta.clone();
                    let (n_acc, nll_new) = mh_steps(
                        &mut eta_work,
                        nll,
                        subject,
                        model,
                        theta_ref,
                        omega_ref,
                        sigma_ref,
                        scale,
                        &mut rng,
                        n_mh_steps,
                    );
                    (eta_work, nll_new, n_acc)
                })
                .collect();

            for (i, (eta_new, nll_new, n_acc)) in results.into_iter().enumerate() {
                state.etas[i] = eta_new;
                state.nll_cache[i] = nll_new;
                state.accept_counts[i] += n_acc;
            }
        }
        state.steps_since_adapt += 1;

        // ---- Step 2: SA update of sufficient statistic for Omega ----
        let mut eta_outer = DMatrix::zeros(n_eta, n_eta);
        for eta in &state.etas {
            let ev = DVector::from_column_slice(eta);
            eta_outer += &ev * ev.transpose();
        }
        eta_outer /= n_subjects as f64;

        state.s2 = (1.0 - gamma) * &state.s2 + gamma * &eta_outer;

        // ---- Step 3: M-step Omega (closed form) ----
        state.omega_mat = state.s2.clone();

        // ---- Step 4: M-step theta, sigma via lightweight NLopt (3 iters, warm-started) ----
        // Only run every few iterations during exploration to save time
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        if run_mstep {
            let mstep_maxiter = if k <= k1 { 3 } else { 5 }; // more precise in convergence phase
            let (theta_new, sigma_new) = theta_sigma_mstep_light(
                model,
                population,
                &state.etas,
                &log_theta,
                &log_sigma,
                &log_theta_lower,
                &log_theta_upper,
                &log_sigma_lower,
                &log_sigma_upper,
                n_theta,
                n_sigma,
                mstep_maxiter,
            );
            log_theta = theta_new;
            log_sigma = sigma_new;
            state.theta = log_theta.iter().map(|&v| v.exp()).collect();
            state.sigma_vals = log_sigma.iter().map(|&v| v.exp()).collect();
        }

        // ---- Update NLL cache (parallelized, needed for MH acceptance ratios) ----
        let omega_upd = OmegaMatrix::from_matrix(
            state.omega_mat.clone(),
            init_params.omega.eta_names.clone(),
            init_params.omega.diagonal,
        );
        {
            use rayon::prelude::*;
            let new_nlls: Vec<f64> = state
                .etas
                .par_iter()
                .enumerate()
                .map(|(i, eta)| {
                    individual_nll(
                        model,
                        &population.subjects[i],
                        &state.theta,
                        eta,
                        &omega_upd,
                        &state.sigma_vals,
                    )
                })
                .collect();
            state.nll_cache = new_nlls;
        }

        // ---- Adapt MH step sizes ----
        if state.steps_since_adapt >= adapt_interval {
            for i in 0..n_subjects {
                let rate = state.accept_counts[i] as f64 / (n_mh_steps * adapt_interval) as f64;
                if rate > 0.4 {
                    state.step_scales[i] = (state.step_scales[i] * 1.1).min(5.0);
                } else {
                    state.step_scales[i] = (state.step_scales[i] * 0.9).max(0.01);
                }
                state.accept_counts[i] = 0;
            }
            state.steps_since_adapt = 0;
        }

        // ---- Verbose output (lightweight: use cached NLL sum) ----
        if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
            let phase = if k <= k1 { "explore" } else { "converge" };
            let cond_nll: f64 = state.nll_cache.iter().sum();
            eprintln!(
                "  SAEM iter {:>4}/{} [{}] gamma={:.3}  condNLL={:.3}",
                k, n_iter, phase, gamma, cond_nll
            );
        }
    }

    if verbose {
        eprintln!("SAEM iterations complete. Computing final EBEs and OFV...");
    }

    // ---- Post-SAEM: build final parameters ----
    let final_omega = OmegaMatrix::from_matrix(
        state.omega_mat.clone(),
        init_params.omega.eta_names.clone(),
        init_params.omega.diagonal,
    );
    let final_params = ModelParameters {
        theta: state.theta.clone(),
        theta_names: init_params.theta_names.clone(),
        theta_lower: init_params.theta_lower.clone(),
        theta_upper: init_params.theta_upper.clone(),
        omega: final_omega,
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
    };

    // ---- Final EBEs via inner loop (warm-started from SAEM etas) ----
    let warm_etas: Vec<DVector<f64>> = state
        .etas
        .iter()
        .map(|e| DVector::from_column_slice(e))
        .collect();
    let (eta_hats, h_matrices, _) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm_etas),
    );

    // ---- Final OFV via FOCE approximation (for AIC/BIC comparability) ----
    let foce_nll = foce_population_nll(
        model,
        population,
        &final_params.theta,
        &eta_hats,
        &h_matrices,
        &final_params.omega,
        &final_params.sigma.values,
        options.interaction,
    );
    let ofv = 2.0 * foce_nll;

    // ---- Covariance step ----
    let mut warnings = Vec::new();
    let covariance_matrix = if options.run_covariance_step
        && !crate::cancel::is_cancelled(&options.cancel)
    {
        if verbose {
            eprintln!("Running covariance step...");
        }
        let packed = pack_params(&final_params);
        let cov = compute_covariance(
            &packed,
            &final_params,
            model,
            population,
            &eta_hats,
            &h_matrices,
            options,
        );
        if cov.is_none() {
            warnings.push("Covariance step failed — SEs not available".to_string());
        }
        cov
    } else {
        None
    };

    if verbose {
        eprintln!("SAEM completed. Final OFV = {:.4}", ofv);
    }

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged: ofv.is_finite(),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        covariance_matrix,
        warnings,
    })
}
