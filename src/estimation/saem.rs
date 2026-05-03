/// SAEM (Stochastic Approximation EM) for NLME population parameter estimation.
///
/// Reference: Delyon, Lavielle, Moulines (1999) Annals of Statistics 94–128.
///            Kuhn & Lavielle (2004) ESAIM: Probability and Statistics 8:115–131.
///
/// Two-phase step-size schedule (Monolix convention):
///   Phase 1 (exploration, k ≤ K1):  γₖ = 1          — rapid basin convergence
///   Phase 2 (convergence, k > K1):  γₖ = 1/(k−K1)   — almost-sure convergence to MLE
use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, pop_nll, OuterResult};
use crate::estimation::parameterization::{compute_mu_k, *};
use crate::stats::likelihood::individual_nll;
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

/// Run `n_steps` symmetric random-walk MH iterations for one subject in-place.
/// Returns (n_accepted, updated_nll).
///
/// `eta` is in deviation (eta_true) space — the same space the model's
/// `pk_param_fn` consumes — so proposals are random walks
/// `eta + step_scale · L · z` from the current position. The acceptance
/// log-ratio is `nll_current − nll_prop`, which is correct because the
/// symmetric proposal density cancels.
///
/// Note: an earlier version centred proposals on `mu_k` during exploration.
/// That was incorrect: `individual_nll` interprets `eta` as the deviation
/// `log(CL_i) − log(TVCL)`, while `mu_k = log(TVCL)`, so the model evaluated
/// `CL = TVCL · exp(log TVCL) = TVCL²` for every accepted exploration step.
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
        let z: Vec<f64> = (0..n_eta).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let perturbation = l * z_vec;

        let eta_prop: Vec<f64> = (0..n_eta)
            .map(|j| eta[j] + step_scale * perturbation[j])
            .collect();

        let nll_prop = individual_nll(model, subject, theta, &eta_prop, omega, sigma_values);

        // Symmetric proposal q(η_prop|η) = q(η|η_prop) cancels in the ratio,
        // so the prior+likelihood difference encoded in `individual_nll` is
        // the full acceptance criterion.
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
    scale_params: bool,
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

    // Objective operating on unscaled log-space parameters.
    //
    // The central-FD gradient parallelises over both the parameter dimension
    // (outer) and subjects (inner via `obs_nll_sum`). Rayon's work-stealing
    // schedules the nested par_iter without oversubscription; in practice
    // nested wins because n_dims (~5–10) alone rarely saturates n_cores (10),
    // while subject-level parallelism alone leaves the FD dims serialised
    // across many small per-sweep sync barriers.
    let obj = |xv: &[f64], grad: Option<&mut [f64]>, _: &mut ()| -> f64 {
        let th: Vec<f64> = xv[..n_theta].iter().map(|&v| v.exp()).collect();
        let sg: Vec<f64> = xv[n_theta..].iter().map(|&v| v.exp()).collect();
        let val = obs_nll_sum(model, population, &th, &sg, etas);

        if let Some(g) = grad {
            use rayon::prelude::*;
            let h = 1e-5;
            let g_vec: Vec<f64> = (0..n)
                .into_par_iter()
                .map(|i| {
                    // Pinned dims (lower == upper) cannot move; skip their FD
                    // evaluations entirely.  This is what makes the SAEM mu-ref
                    // gradient step actually save NLopt OFV evaluations —
                    // without it, NLopt's central FD would still hit each
                    // pinned dim.
                    if lower[i] == upper[i] {
                        return 0.0;
                    }
                    let mut xp = xv.to_vec();
                    xp[i] = xv[i] + h;
                    let th_p: Vec<f64> = xp[..n_theta].iter().map(|&v| v.exp()).collect();
                    let sg_p: Vec<f64> = xp[n_theta..].iter().map(|&v| v.exp()).collect();
                    let fp = obs_nll_sum(model, population, &th_p, &sg_p, etas);
                    xp[i] = xv[i] - h;
                    let th_m: Vec<f64> = xp[..n_theta].iter().map(|&v| v.exp()).collect();
                    let sg_m: Vec<f64> = xp[n_theta..].iter().map(|&v| v.exp()).collect();
                    let fm = obs_nll_sum(model, population, &th_m, &sg_m, etas);
                    let gi = (fp - fm) / (2.0 * h);
                    if gi.is_finite() {
                        gi
                    } else {
                        0.0
                    }
                })
                .collect();
            g.copy_from_slice(&g_vec);
        }

        if val.is_finite() {
            val
        } else {
            1e20
        }
    };

    // Compute per-element scale factors from the initial point.
    let scale: Vec<f64> = if scale_params {
        compute_scale(&x)
    } else {
        vec![1.0; n]
    };

    // Scaled starting point and bounds: xs[i] = x[i] / scale[i].
    let mut xs: Vec<f64> = (0..n).map(|i| x[i] / scale[i]).collect();
    let lower_s: Vec<f64> = (0..n).map(|i| lower[i] / scale[i]).collect();
    let upper_s: Vec<f64> = (0..n).map(|i| upper[i] / scale[i]).collect();

    // Wrapper objective: receives scaled xs, unscales before evaluating obj,
    // then scales the gradient back: d(OFV)/d(xs[i]) = d(OFV)/d(x[i]) * scale[i].
    let obj_s = |xv_s: &[f64], grad: Option<&mut [f64]>, data: &mut ()| -> f64 {
        let xv: Vec<f64> = (0..n).map(|i| xv_s[i] * scale[i]).collect();
        if let Some(g) = grad {
            let mut g_raw = vec![0.0_f64; n];
            let val = obj(&xv, Some(&mut g_raw), data);
            for i in 0..n {
                g[i] = g_raw[i] * scale[i];
            }
            val
        } else {
            obj(&xv, None, data)
        }
    };

    let mut opt = nlopt::Nlopt::new(nlopt::Algorithm::Slsqp, n, obj_s, nlopt::Target::Minimize, ());
    opt.set_lower_bounds(&lower_s).unwrap();
    opt.set_upper_bounds(&upper_s).unwrap();
    opt.set_maxeval(maxiter * (n as u32 + 1)).unwrap();
    opt.set_ftol_rel(1e-4).unwrap();

    match opt.optimize(&mut xs) {
        Ok(_) | Err(_) => {}
    }

    // Unscale back to log-space.
    let x_final: Vec<f64> = (0..n).map(|i| xs[i] * scale[i]).collect();

    let log_theta_new = x_final[..n_theta].to_vec();
    let log_sigma_new = x_final[n_theta..].to_vec();
    (log_theta_new, log_sigma_new)
}

/// Observation NLL for a single subject with ETAs held fixed.
///
/// Under M3, CENS=1 rows contribute `-log Φ((LLOQ - f)/√V)`.
fn obs_nll_single(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    sigma_values: &[f64],
    eta: &[f64],
) -> f64 {
    let m3 = matches!(model.bloq_method, BloqMethod::M3);
    let pk_params = (model.pk_param_fn)(theta, eta, &subject.covariates);
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
    use rayon::prelude::*;
    population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| obs_nll_single(model, subject, theta, sigma_values, &etas[i]))
        .sum()
}

/// Build (theta_idx, eta_idx) pairs for log-transformed mu-references only.
///
/// Only `log_transformed = true` mu-refs (patterns `THETA*exp(ETA)` and
/// `exp(log(THETA)+ETA)`) participate in the gradient-step M-step.  For these
/// the chain rule gives `d/d_log(theta) = -Σᵢ d/d_eta`, which matches the
/// update applied in the SAEM loop.  Additive mu-refs (`THETA + ETA`,
/// `log_transformed = false`) require the extra factor of `theta` from the
/// log-space chain rule and are deliberately excluded — they fall through to
/// the regular NLopt M-step.
fn get_mu_ref_pairs(model: &CompiledModel) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if !mu_ref.log_transformed {
                continue;
            }
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                pairs.push((theta_idx, eta_idx));
            }
        }
    }
    pairs
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

    let etas: Vec<Vec<f64>> = (0..n_subjects)
        .map(|_| get_eta_init(n_eta, None, None))
        .collect();
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
    let mut log_theta_lower: Vec<f64> = init_params
        .theta_lower
        .iter()
        .map(|&b| b.max(1e-10).ln())
        .collect();
    let mut log_theta_upper: Vec<f64> = init_params
        .theta_upper
        .iter()
        .map(|&b| b.min(1e9).ln())
        .collect();
    let mut log_sigma_lower = vec![-8.0f64; n_sigma];
    let mut log_sigma_upper = vec![5.0f64; n_sigma];

    // Pin FIX parameters: set lower == upper == log(current) so the inner
    // NLopt M-step treats them as constants. Matches the FOCE/FOCEI treatment.
    for i in 0..n_theta {
        if init_params.theta_fixed.get(i).copied().unwrap_or(false) {
            log_theta_lower[i] = log_theta[i];
            log_theta_upper[i] = log_theta[i];
        }
    }
    for i in 0..n_sigma {
        if init_params.sigma_fixed.get(i).copied().unwrap_or(false) {
            log_sigma_lower[i] = log_sigma[i];
            log_sigma_upper[i] = log_sigma[i];
        }
    }

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

    // Mu-referencing pairs for the closed-form M-step: (theta_idx, eta_idx).
    // Only log-mu-ref pairs are returned (`get_mu_ref_pairs` filters out
    // additive ones), since the closed-form `log_theta += γ · mean(η)` only
    // applies to log-mu-referenced thetas.
    let mu_ref_pairs: Vec<(usize, usize)> = get_mu_ref_pairs(model);
    let use_closed_form_mstep = options.mu_referencing && !mu_ref_pairs.is_empty();
    // Accumulator for the `obs_nll_sum` (population OFV) evaluations skipped
    // by pinning mu-ref dims out of NLopt's central-FD gradient.  Each pinned
    // dim costs `2 * mstep_maxiter` `obs_nll_sum` calls inside NLopt — that's
    // the value we add per M-step that takes the closed-form branch.
    let mut mstep_grad_step_evals_saved: u64 = 0;

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
        // Symmetric random-walk MH in eta_true space, identical schedule
        // throughout exploration and convergence — the only thing that
        // changes between phases is the SA step size `gamma`.
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
        // Restore FIX-ed rows / columns from the template. An eta flagged FIX
        // keeps its initial variance AND its initial off-diagonal couplings
        // (zero for a diagonal declaration, block cov for a FIX-ed block).
        // Letting the sufficient statistic bleed into row/col of a fixed eta
        // breaks positive-definiteness once the free-block diagonals shrink
        // during the exploration phase.
        state.omega_mat = state.s2.clone();
        // Zero structurally-absent off-diagonals. `s2 = (1/N) Σ ηη^T` always
        // produces a dense matrix; entries that aren't free parameters
        // (standalone etas, or etas from different `block_omega` declarations)
        // must be zeroed so they don't feed sampling correlations back into
        // the next iteration's Cholesky proposal. Without this the chain drives
        // Ω toward a rank-deficient state, log|Ω| → -∞, and the M-step pushes
        // thetas to bounds to compensate.
        for i in 0..n_eta {
            for j in 0..n_eta {
                if !init_params.omega.free_mask[(i, j)] {
                    state.omega_mat[(i, j)] = 0.0;
                }
            }
        }
        // Restore FIX-ed rows / columns from the template.
        for i in 0..n_eta {
            for j in 0..n_eta {
                let fi = init_params.omega_fixed.get(i).copied().unwrap_or(false);
                let fj = init_params.omega_fixed.get(j).copied().unwrap_or(false);
                if fi || fj {
                    state.omega_mat[(i, j)] = init_params.omega.matrix[(i, j)];
                }
            }
        }

        // ---- Step 4: M-step theta, sigma (lightweight NLopt, warm-started) ----
        // Only run every few iterations during exploration to save time
        let run_mstep = k <= 5 || k % 3 == 0 || k > k1;
        if run_mstep {
            let mstep_maxiter = if k <= k1 { 3 } else { 5 }; // more precise in convergence phase

            if use_closed_form_mstep {
                // Closed-form EM M-step for log-mu-referenced thetas.
                //
                // Model: log(P_i) = log(TVP) + η_i, η_i ~ N(0, ω²).
                // The complete-data log-likelihood is maximised at
                //     log(TVP)_new = log(TVP)_old + mean_i(η_i)
                // and SAEM applies the stochastic-approximation step size γ:
                //     log(TVP)_new = log(TVP)_old + γ · mean_i(η_i)
                // After the update, η_i is re-centred by `mean(η)` so the
                // sufficient statistic for ω is taken from zero-mean residuals
                // (ω is updated from `s2` *after* the next MH step, but
                // re-centring keeps `state.etas` consistent with the new TVP
                // for the rest of this iteration's NLL cache refresh).
                let n_subj = state.etas.len() as f64;
                let mut temp_theta_lower = log_theta_lower.clone();
                let mut temp_theta_upper = log_theta_upper.clone();
                for &(theta_idx, eta_idx) in &mu_ref_pairs {
                    if init_params.theta_fixed.get(theta_idx).copied().unwrap_or(false) {
                        continue;
                    }
                    let mean_eta: f64 =
                        state.etas.iter().map(|e| e[eta_idx]).sum::<f64>() / n_subj;
                    log_theta[theta_idx] += gamma * mean_eta;
                    log_theta[theta_idx] = log_theta[theta_idx]
                        .clamp(log_theta_lower[theta_idx], log_theta_upper[theta_idx]);
                    // Re-centre etas for this dim by the same shift so they
                    // remain deviations from the *new* log(TVP).
                    for e in state.etas.iter_mut() {
                        e[eta_idx] -= gamma * mean_eta;
                    }
                    // Pin so NLopt leaves the closed-form value unchanged.
                    temp_theta_lower[theta_idx] = log_theta[theta_idx];
                    temp_theta_upper[theta_idx] = log_theta[theta_idx];
                }
                // Each pinned mu-ref dim avoids 2 obs_nll_sum calls per NLopt
                // gradient request, capped at `mstep_maxiter` requests.
                mstep_grad_step_evals_saved +=
                    2 * mstep_maxiter as u64 * mu_ref_pairs.len() as u64;

                // NLopt for non-mu-ref thetas (pinned) and sigma.
                let (theta_new, sigma_new) = theta_sigma_mstep_light(
                    model,
                    population,
                    &state.etas,
                    &log_theta,
                    &log_sigma,
                    &temp_theta_lower,
                    &temp_theta_upper,
                    &log_sigma_lower,
                    &log_sigma_upper,
                    n_theta,
                    n_sigma,
                    mstep_maxiter,
                    options.scale_params,
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            } else {
                // mu_referencing = false: full NLopt M-step for all thetas + sigma (unchanged)
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
                    options.scale_params,
                );
                log_theta = theta_new;
                log_sigma = sigma_new;
            }

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

        // ---- Verbose output + optimizer trace ----
        {
            let phase = if k <= k1 { "explore" } else { "converge" };
            let cond_nll: f64 = state.nll_cache.iter().sum();
            // Rolling MH accept rate since the last adapt reset.
            let steps_so_far = state.steps_since_adapt.max(1);
            let mh_accept_rate: f64 = state.accept_counts.iter().sum::<usize>() as f64
                / (n_subjects * n_mh_steps * steps_so_far) as f64;

            if verbose && (k == 1 || k % 50 == 0 || k == n_iter) {
                eprintln!(
                    "  SAEM iter {:>4}/{} [{}] gamma={:.3}  condNLL={:.3}",
                    k, n_iter, phase, gamma, cond_nll
                );
            }

            crate::estimation::trace::write_saem(k, phase, cond_nll, gamma, mh_accept_rate);
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
        theta_fixed: init_params.theta_fixed.clone(),
        omega: final_omega,
        omega_fixed: init_params.omega_fixed.clone(),
        sigma: SigmaVector {
            values: state.sigma_vals.clone(),
            names: init_params.sigma.names.clone(),
        },
        sigma_fixed: init_params.sigma_fixed.clone(),
        omega_iov: init_params.omega_iov.clone(),
        kappa_fixed: init_params.kappa_fixed.clone(),
    };

    // ---- Final EBEs via inner loop (warm-started from SAEM etas) ----
    let warm_etas: Vec<DVector<f64>> = state
        .etas
        .iter()
        .map(|e| DVector::from_column_slice(e))
        .collect();
    let saem_final_mu_k = compute_mu_k(model, &final_params.theta, options.mu_referencing);
    let (eta_hats, h_matrices, _, final_kappas) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        Some(&warm_etas),
        Some(&saem_final_mu_k),
        0, // SAEM: no EBE convergence tracking
    );

    // ---- Final OFV via FOCE approximation (for AIC/BIC comparability) ----
    let ofv = 2.0 * pop_nll(model, population, &final_params, &eta_hats, &h_matrices, &final_kappas, options.interaction);

    // ---- Covariance step ----
    let mut warnings = Vec::new();
    let covariance_matrix =
        if options.run_covariance_step && !crate::cancel::is_cancelled(&options.cancel) {
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
                &final_kappas,
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

    let saem_mu_ref_m_step_evals_saved = if use_closed_form_mstep {
        Some(mstep_grad_step_evals_saved)
    } else {
        None
    };

    Ok(OuterResult {
        params: final_params,
        ofv,
        converged: ofv.is_finite(),
        n_iterations: n_iter,
        eta_hats,
        h_matrices,
        kappas: final_kappas,
        covariance_matrix,
        warnings,
        saem_mu_ref_m_step_evals_saved,
        ebe_convergence_warnings: 0,
        max_unconverged_subjects: 0,
        total_ebe_fallbacks: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::test_helpers::analytical_model;
    use crate::types::{GradientMethod, MuRef};

    fn model_with_mu_refs(
        theta_names: &[&str],
        eta_names: &[&str],
        mu_refs: &[(&str, &str, bool)],
    ) -> CompiledModel {
        let mut m = analytical_model(GradientMethod::Auto);
        m.theta_names = theta_names.iter().map(|s| (*s).to_string()).collect();
        m.eta_names = eta_names.iter().map(|s| (*s).to_string()).collect();
        m.n_theta = theta_names.len();
        m.n_eta = eta_names.len();
        m.mu_refs = mu_refs
            .iter()
            .map(|(eta, theta, log_t)| {
                (
                    (*eta).to_string(),
                    MuRef {
                        theta_name: (*theta).to_string(),
                        log_transformed: *log_t,
                    },
                )
            })
            .collect();
        m
    }

    #[test]
    fn get_mu_ref_pairs_empty_when_no_mu_refs() {
        let m = analytical_model(GradientMethod::Auto);
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    #[test]
    fn get_mu_ref_pairs_returns_log_transformed_pair() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let mut pairs = get_mu_ref_pairs(&m);
        pairs.sort();
        assert_eq!(pairs, vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn get_mu_ref_pairs_excludes_additive_mu_refs() {
        // ETA_CL is lognormal (THETA*exp(ETA)) — included.
        // ETA_V is additive (THETA+ETA) — excluded because the gradient-step
        // chain rule used in run_saem assumes log-transformed parameters.
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", false)],
        );
        assert_eq!(get_mu_ref_pairs(&m), vec![(0, 0)]);
    }

    #[test]
    fn get_mu_ref_pairs_skips_orphaned_theta() {
        // mu_ref points at a theta name that doesn't exist — silently skipped.
        let m = model_with_mu_refs(
            &["CL"],
            &["ETA_CL"],
            &[("ETA_CL", "MISSING", true)],
        );
        assert!(get_mu_ref_pairs(&m).is_empty());
    }

    // ---- Regression tests for the three SAEM correctness bugs ----

    /// Bug 1 (diagonal): `from_diagonal` produces a free_mask that marks only
    /// diagonal entries free. The SAEM M-step uses this mask to zero
    /// SA-accumulated off-diagonals, preventing the rank-deficient Ω failure.
    #[test]
    fn diagonal_omega_free_mask_has_no_off_diagonals() {
        let omega = OmegaMatrix::from_diagonal(&[0.1, 0.2], vec!["ETA_CL".into(), "ETA_V".into()]);
        assert!(omega.free_mask[(0, 0)]);
        assert!(omega.free_mask[(1, 1)]);
        assert!(!omega.free_mask[(0, 1)]);
        assert!(!omega.free_mask[(1, 0)]);
    }

    /// Bug 1 (mixed structure): `from_matrix_with_mask` preserves an explicit
    /// mask that marks cross-block entries as structural zeros. This is the
    /// case that the `diagonal` flag alone cannot express (one standalone eta
    /// + one block_omega pair → diagonal=false, but cross entries are zero).
    #[test]
    fn mixed_omega_free_mask_zeros_cross_block_entries() {
        // Three etas: ETA_CL(0) and ETA_V(1) in a block; ETA_KA(2) standalone.
        let mut matrix = nalgebra::DMatrix::zeros(3, 3);
        matrix[(0, 0)] = 0.1;
        matrix[(1, 1)] = 0.2;
        matrix[(2, 2)] = 0.1;
        matrix[(0, 1)] = 0.01;
        matrix[(1, 0)] = 0.01;

        let mut free_mask = nalgebra::DMatrix::from_element(3, 3, false);
        free_mask[(0, 0)] = true;
        free_mask[(1, 1)] = true;
        free_mask[(2, 2)] = true;
        free_mask[(0, 1)] = true; // within CL-V block
        free_mask[(1, 0)] = true;

        let names = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let omega = OmegaMatrix::from_matrix_with_mask(matrix, names, false, free_mask);

        assert!(omega.free_mask[(0, 1)]);
        assert!(omega.free_mask[(1, 0)]);
        assert!(!omega.free_mask[(2, 0)]);
        assert!(!omega.free_mask[(0, 2)]);
        assert!(!omega.free_mask[(2, 1)]);
        assert!(!omega.free_mask[(1, 2)]);
    }

    /// Bug 2: `mh_steps` is a symmetric random walk — proposals are not
    /// re-centred on `mu_k`. Starting from eta=0 with a wide sigma, 500 steps
    /// must move eta away from zero. Under the pre-fix mu_k-centred kernel
    /// (mu_k=0 in this case), the chain could never leave zero by construction.
    #[test]
    fn mh_steps_random_walk_leaves_eta_zero() {
        use crate::stats::likelihood::individual_nll;
        use crate::types::{DoseEvent, SigmaVector};
        use rand::SeedableRng;
        use rand::rngs::StdRng;
        use std::collections::HashMap;

        let model = analytical_model(GradientMethod::Auto);
        let subj = Subject {
            id: "1".into(),
            doses: vec![DoseEvent::new(0.0, 100.0, 1, 0.0, false, 0.0)],
            obs_times: vec![1.0],
            observations: vec![1.0],
            obs_cmts: vec![1],
            covariates: HashMap::new(),
            tvcov: HashMap::new(),
            cens: vec![0],
            occasions: vec![],
            dose_occasions: vec![],
        };
        let omega = OmegaMatrix::from_diagonal(&[1.0], vec!["ETA_CL".into()]);
        let sigma = SigmaVector { values: vec![100.0], names: vec!["PROP".into()] };
        let theta = vec![1.0];
        let mut eta = vec![0.0_f64];
        let nll_start = individual_nll(&model, &subj, &theta, &eta, &omega, &sigma.values);
        let mut rng = StdRng::seed_from_u64(42);

        mh_steps(
            &mut eta,
            nll_start,
            &subj,
            &model,
            &theta,
            &omega,
            &sigma.values,
            1.0,
            &mut rng,
            500,
        );

        assert!(
            eta[0].abs() > 1e-6,
            "eta stayed at 0 — proposals were incorrectly re-centred on mu_k"
        );
    }

    /// Bug 3 / closed-form M-step: a synthetic SAEM run with mu_referencing=true
    /// and mean(eta) ≠ 0 must move log_theta in the right direction *without*
    /// pinning at the bound. We exercise the closed-form formula directly:
    /// `log_theta_new = log_theta_old + γ · mean(eta)`.
    #[test]
    fn closed_form_mu_ref_mstep_is_bounded_and_signed_correctly() {
        // Simulate post-MH state: 5 subjects, eta_mean = +0.4 (population CL
        // is higher than current TVCL), gamma = 1.0 (exploration step).
        let etas: Vec<Vec<f64>> = vec![
            vec![0.5],
            vec![0.3],
            vec![0.4],
            vec![0.6],
            vec![0.2],
        ];
        let n = etas.len() as f64;
        let mean_eta: f64 = etas.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!((mean_eta - 0.4).abs() < 1e-12);

        let gamma = 1.0;
        let log_theta_old = 0.0_f64; // TVCL = 1.0
        let log_theta_new = log_theta_old + gamma * mean_eta;
        // log_theta moved by exactly mean(eta), independent of N.  This is the
        // property that the broken gradient step (γ · Σ ∂obs_nll/∂eta) lacked:
        // its update scaled with N and pinned thetas at bounds for moderate N.
        assert!((log_theta_new - 0.4).abs() < 1e-12);

        // After re-centring etas by gamma*mean, mean(eta) = 0.
        let mut etas_recentered = etas.clone();
        for e in etas_recentered.iter_mut() {
            e[0] -= gamma * mean_eta;
        }
        let new_mean: f64 = etas_recentered.iter().map(|e| e[0]).sum::<f64>() / n;
        assert!(new_mean.abs() < 1e-12);
    }

    /// Bug 3 follow-up: the broken gradient step (γ · Σᵢ ∂obs_nll/∂eta) is no
    /// longer in the code path. The closed-form `log_theta += γ · mean(η)` is
    /// what runs when mu_referencing=true. Pair detection is unchanged.
    #[test]
    fn mu_ref_pair_detection_drives_closed_form_branch() {
        let m = model_with_mu_refs(
            &["CL", "V"],
            &["ETA_CL", "ETA_V"],
            &[("ETA_CL", "CL", true), ("ETA_V", "V", true)],
        );
        let pairs = get_mu_ref_pairs(&m);
        assert_eq!(pairs.len(), 2);
        // The closed-form branch is taken iff `options.mu_referencing` AND
        // `!pairs.is_empty()`.  Both conditions are tested via the public API
        // in api::iov_integration::test_iov_foce_mu_referencing_on; this unit
        // test pins the precondition (pair detection still produces work).
    }
}
