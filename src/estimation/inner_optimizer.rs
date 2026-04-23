use crate::pk;
use crate::stats::likelihood::individual_nll;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "autodiff")]
use crate::ad::ad_gradients::{self, FlatDoseData};

/// Resolve [`GradientMethod::Auto`] to a concrete choice of AD vs FD for this
/// model/subject.
///
/// AD wins when forward_cost >> n_eta * perturbation_cost; in practice that
/// means expensive forward models (ODE integration) or models with many etas.
/// Analytical PK with small n_eta is the opposite regime — FD is faster.
///
/// AD also requires (a) the crate to be compiled with `feature = "autodiff"`
/// and (b) the model to have `tv_fn` populated (analytical PK path only).
/// ODE models currently have no AD path, so Auto must resolve to FD.
fn resolve_gradient_method(model: &CompiledModel, n_eta: usize) -> bool {
    #[cfg(not(feature = "autodiff"))]
    {
        let _ = (model, n_eta);
        return false;
    }
    #[cfg(feature = "autodiff")]
    {
        if model.tv_fn.is_none() {
            return false;
        }
        let _ = n_eta;
        match model.gradient_method {
            GradientMethod::Ad => true,
            GradientMethod::Fd => false,
            // Empirically (FERX_TIME_GRADIENTS=1 on warfarin, 3 etas, analytical
            // 1-cpt): reverse-mode AD is ~3× faster per BFGS call than central
            // FD even at small n_eta. The tape/backward overhead is dominated
            // by the savings from one gradient call vs 2n forward perturbations.
            // Prefer AD whenever it is available.
            GradientMethod::Auto => true,
        }
    }
}

/// Global per-fit timing counters for gradient/Jacobian calls. Printed by
/// [`fit_inner`] when `FERX_TIME_GRADIENTS=1` in the environment. Atomics so
/// multiple rayon workers can update concurrently without locking.
pub(crate) struct GradientTimings {
    pub ad_calls: AtomicU64,
    pub ad_nanos: AtomicU64,
    pub fd_calls: AtomicU64,
    pub fd_nanos: AtomicU64,
    pub jac_ad_calls: AtomicU64,
    pub jac_ad_nanos: AtomicU64,
    pub jac_fd_calls: AtomicU64,
    pub jac_fd_nanos: AtomicU64,
}

impl GradientTimings {
    const fn new() -> Self {
        Self {
            ad_calls: AtomicU64::new(0),
            ad_nanos: AtomicU64::new(0),
            fd_calls: AtomicU64::new(0),
            fd_nanos: AtomicU64::new(0),
            jac_ad_calls: AtomicU64::new(0),
            jac_ad_nanos: AtomicU64::new(0),
            jac_fd_calls: AtomicU64::new(0),
            jac_fd_nanos: AtomicU64::new(0),
        }
    }
    #[inline]
    fn record_ad(&self, ns: u64) {
        self.ad_calls.fetch_add(1, Ordering::Relaxed);
        self.ad_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_fd(&self, ns: u64) {
        self.fd_calls.fetch_add(1, Ordering::Relaxed);
        self.fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_ad(&self, ns: u64) {
        self.jac_ad_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_ad_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    #[inline]
    fn record_jac_fd(&self, ns: u64) {
        self.jac_fd_calls.fetch_add(1, Ordering::Relaxed);
        self.jac_fd_nanos.fetch_add(ns, Ordering::Relaxed);
    }
    pub(crate) fn reset(&self) {
        self.ad_calls.store(0, Ordering::Relaxed);
        self.ad_nanos.store(0, Ordering::Relaxed);
        self.fd_calls.store(0, Ordering::Relaxed);
        self.fd_nanos.store(0, Ordering::Relaxed);
        self.jac_ad_calls.store(0, Ordering::Relaxed);
        self.jac_ad_nanos.store(0, Ordering::Relaxed);
        self.jac_fd_calls.store(0, Ordering::Relaxed);
        self.jac_fd_nanos.store(0, Ordering::Relaxed);
    }
    pub(crate) fn snapshot(&self) -> (u64, u64, u64, u64, u64, u64, u64, u64) {
        (
            self.ad_calls.load(Ordering::Relaxed),
            self.ad_nanos.load(Ordering::Relaxed),
            self.fd_calls.load(Ordering::Relaxed),
            self.fd_nanos.load(Ordering::Relaxed),
            self.jac_ad_calls.load(Ordering::Relaxed),
            self.jac_ad_nanos.load(Ordering::Relaxed),
            self.jac_fd_calls.load(Ordering::Relaxed),
            self.jac_fd_nanos.load(Ordering::Relaxed),
        )
    }
}

pub(crate) static GRADIENT_TIMINGS: GradientTimings = GradientTimings::new();

/// Result of inner optimization for a single subject
pub struct EbeResult {
    pub eta: DVector<f64>,
    pub h_matrix: DMatrix<f64>,
    pub converged: bool,
    pub nll: f64,
}

/// Find Empirical Bayes Estimates (EBEs) for a single subject via BFGS.
///
/// When `mu_k` is provided (mu-referencing active), the inner optimizer works
/// in psi-space where `psi = eta_true + mu_k`.  The objective is evaluated as
/// `individual_nll(psi - mu_k)`, so the model always receives `eta_true`.
/// Warm starts (in `eta_true` space) are converted to psi-space on entry;
/// the returned EbeResult always holds `eta_true = psi - mu_k`.
///
/// When `mu_k` is None every shift is zero and the behaviour is identical to
/// the original (eta-space) implementation.
pub fn find_ebe(
    model: &CompiledModel,
    subject: &Subject,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    eta_init: Option<&[f64]>,
    mu_k: Option<&[f64]>,
) -> EbeResult {
    let n_eta = model.n_eta;

    // mu: shift vector (zeros when no mu-referencing)
    let mu: Vec<f64> = mu_k.map(|m| m.to_vec()).unwrap_or_else(|| vec![0.0; n_eta]);

    // psi_init: warm start converted to psi-space, or prior mode (psi = mu, eta_true = 0)
    let mut psi: Vec<f64> = match eta_init {
        Some(warm) => warm.iter().zip(mu.iter()).map(|(e, m)| e + m).collect(),
        None => mu.clone(),
    };

    // Objective in psi-space: model always receives eta_true = psi - mu
    let obj = |p: &[f64]| -> f64 {
        let eta_t: Vec<f64> = p.iter().zip(mu.iter()).map(|(pi, mi)| pi - mi).collect();
        individual_nll(
            model,
            subject,
            &params.theta,
            &eta_t,
            &params.omega,
            &params.sigma.values,
        )
    };

    // Resolve Auto → concrete method based on model/eta characteristics.
    // Autodiff is only available when the crate was compiled with the feature
    // and the model provides tv_fn (the parser attaches it for analytical PK).
    let use_ad = resolve_gradient_method(model, n_eta);

    // Try BFGS — AD gradient if `use_ad`, FD otherwise. The AD gradient of
    // individual_nll w.r.t. psi equals the gradient w.r.t. eta_true (chain
    // rule: d/dpsi = d/d(eta_true), since psi = eta_true + mu).
    #[cfg(feature = "autodiff")]
    let result = if use_ad {
        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("resolve_gradient_method guarantees tv_fn");
        let tv_adjusted = tv_fn(&params.theta, &subject.covariates);
        let dose_data = FlatDoseData::from_subject(subject);
        let omega_inv = params
            .omega
            .matrix
            .clone()
            .cholesky()
            .map(|c| c.inverse())
            .unwrap_or_else(|| nalgebra::DMatrix::identity(n_eta, n_eta));
        let mut omega_inv_flat = Vec::with_capacity(n_eta * n_eta);
        for i in 0..n_eta {
            for j in 0..n_eta {
                omega_inv_flat.push(omega_inv[(i, j)]);
            }
        }
        let log_det_omega = {
            let mut ld = 0.0;
            for i in 0..n_eta {
                let lii = params.omega.chol[(i, i)];
                ld += if lii > 0.0 {
                    lii.ln()
                } else {
                    return EbeResult {
                        eta: DVector::zeros(n_eta),
                        h_matrix: DMatrix::zeros(0, 0),
                        converged: false,
                        nll: 1e20,
                    };
                };
            }
            2.0 * ld
        };

        let pk_indices = &model.pk_indices;
        // Under M3, feed actual CENS flags so the AD path applies -log Φ to
        // censored rows. Otherwise pass zeros — Enzyme will trace the Gaussian
        // branch for every observation, identical to the pre-M3 behavior.
        let cens_f64: Vec<f64> = if matches!(model.bloq_method, BloqMethod::M3) {
            subject.cens.iter().map(|&c| c as f64).collect()
        } else {
            vec![0.0; subject.observations.len()]
        };
        let mu_ad = mu.clone();
        let grad_fn = |p: &[f64]| -> Vec<f64> {
            let eta_t: Vec<f64> = p.iter().zip(mu_ad.iter()).map(|(pi, mi)| pi - mi).collect();
            let t0 = std::time::Instant::now();
            let (_, g) = ad_gradients::compute_nll_gradient_ad(
                &eta_t,
                &tv_adjusted,
                &omega_inv_flat,
                log_det_omega,
                &params.sigma.values,
                &dose_data,
                &subject.obs_times,
                &subject.observations,
                &cens_f64,
                model.pk_model,
                model.error_model,
                &model.pk_idx_f64,
                &model.sel_flat,
            );
            GRADIENT_TIMINGS.record_ad(t0.elapsed().as_nanos() as u64);
            g
        };
        bfgs_minimize_with_grad(&obj, &grad_fn, &mut psi, n_eta, max_iter, tol)
    } else {
        bfgs_minimize(&obj, &mut psi, n_eta, max_iter, tol)
    };

    #[cfg(not(feature = "autodiff"))]
    let result = {
        let _ = use_ad; // silence unused warning on stable builds
        bfgs_minimize(&obj, &mut psi, n_eta, max_iter, tol)
    };

    // If BFGS failed, try Nelder-Mead from the prior mode (psi = mu, eta_true = 0)
    if !result {
        psi = mu.clone();
        nelder_mead_minimize(&obj, &mut psi, n_eta, max_iter * 5, tol);
    }

    let nll = obj(&psi);

    // Recover eta_true = psi - mu (mean-zero, NONMEM-compatible output)
    let eta_true: Vec<f64> = psi.iter().zip(mu.iter()).map(|(p, m)| p - m).collect();

    // Compute Jacobian at eta_true — use AD when available and chosen.
    #[cfg(feature = "autodiff")]
    let h_matrix = if use_ad {
        let tv_fn = model
            .tv_fn
            .as_ref()
            .expect("resolve_gradient_method guarantees tv_fn");
        let tv_adjusted = tv_fn(&params.theta, &subject.covariates);
        let dose_data = FlatDoseData::from_subject(subject);
        let t0 = std::time::Instant::now();
        let j = ad_gradients::compute_jacobian_ad(
            &eta_true,
            &tv_adjusted,
            &dose_data,
            &subject.obs_times,
            subject.obs_times.len(),
            model.pk_model,
            &model.pk_idx_f64,
            &model.sel_flat,
        );
        GRADIENT_TIMINGS.record_jac_ad(t0.elapsed().as_nanos() as u64);
        j
    } else {
        let t0 = std::time::Instant::now();
        let j = compute_jacobian_fd(model, subject, &params.theta, &eta_true);
        GRADIENT_TIMINGS.record_jac_fd(t0.elapsed().as_nanos() as u64);
        j
    };

    #[cfg(not(feature = "autodiff"))]
    let h_matrix = {
        let t0 = std::time::Instant::now();
        let j = compute_jacobian_fd(model, subject, &params.theta, &eta_true);
        GRADIENT_TIMINGS.record_jac_fd(t0.elapsed().as_nanos() as u64);
        j
    };

    EbeResult {
        eta: DVector::from_column_slice(&eta_true),
        h_matrix,
        converged: nll.is_finite(),
        nll,
    }
}

/// BFGS minimization with backtracking line search.
/// Uses analytical-style gradient via forward FD with small step.
fn bfgs_minimize(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = gradient_fd(obj, x, n);
    let mut first_step = true;

    for _iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt();

        // Scale initial Hessian so first step is O(1) not O(gnorm)
        if first_step && gnorm > 1.0 {
            let scale = 1.0 / gnorm;
            h_inv *= scale;
            first_step = false;
        }
        if gnorm < tol {
            return true;
        }

        // Search direction
        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 {
            // Reset to steepest descent
            h_inv = DMatrix::identity(n, n);
            let d: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let alpha = backtracking_line_search(obj, x, &d, &g, n);
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            g = gradient_fd(obj, x, n);
            continue;
        }

        let alpha = backtracking_line_search(obj, x, &d, &g, n);
        if alpha < 1e-16 {
            return false;
        }

        // s = alpha * d
        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }

        let g_new = gradient_fd(obj, x, n);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        // BFGS update
        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let sy = s_vec.dot(&y_vec);
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            let eye = DMatrix::identity(n, n);
            let s_yt = rho * &s_vec * y_vec.transpose();
            let y_st = rho * &y_vec * s_vec.transpose();
            let s_st = rho * &s_vec * s_vec.transpose();
            h_inv = (&eye - &s_yt) * &h_inv * (&eye - &y_st) + s_st;
        }

        g = g_new;
    }

    false
}

/// BFGS minimization with an externally-provided gradient function (for AD).
#[cfg(feature = "autodiff")]
fn bfgs_minimize_with_grad(
    obj: &dyn Fn(&[f64]) -> f64,
    grad: &dyn Fn(&[f64]) -> Vec<f64>,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let mut h_inv = DMatrix::identity(n, n);
    let mut g = grad(x);
    let mut first_step = true;

    for _iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|&gi| gi * gi).sum::<f64>().sqrt();

        if first_step && gnorm > 1.0 {
            let scale = 1.0 / gnorm;
            h_inv *= scale;
            first_step = false;
        }

        if gnorm < tol {
            return true;
        }

        let g_vec = DVector::from_column_slice(&g);
        let d_vec = -&h_inv * &g_vec;
        let d: Vec<f64> = d_vec.iter().copied().collect();

        let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();
        if dg >= 0.0 {
            h_inv = DMatrix::identity(n, n);
            let d: Vec<f64> = g.iter().map(|gi| -gi).collect();
            let alpha = backtracking_line_search(obj, x, &d, &g, n);
            for i in 0..n {
                x[i] += alpha * d[i];
            }
            g = grad(x);
            continue;
        }

        let alpha = backtracking_line_search(obj, x, &d, &g, n);
        if alpha < 1e-16 {
            return false;
        }

        let s: Vec<f64> = (0..n).map(|i| alpha * d[i]).collect();
        for i in 0..n {
            x[i] += s[i];
        }

        let g_new = grad(x);
        let y: Vec<f64> = (0..n).map(|i| g_new[i] - g[i]).collect();

        let s_vec = DVector::from_column_slice(&s);
        let y_vec = DVector::from_column_slice(&y);
        let sy = s_vec.dot(&y_vec);
        if sy > 1e-12 {
            let rho = 1.0 / sy;
            let eye = DMatrix::identity(n, n);
            let s_yt = rho * &s_vec * y_vec.transpose();
            let y_st = rho * &y_vec * s_vec.transpose();
            let s_st = rho * &s_vec * s_vec.transpose();
            h_inv = (&eye - &s_yt) * &h_inv * (&eye - &y_st) + s_st;
        }

        g = g_new;
    }

    false
}

/// Nelder-Mead simplex minimization (fallback)
fn nelder_mead_minimize(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &mut [f64],
    n: usize,
    max_iter: usize,
    tol: f64,
) -> bool {
    let alpha = 1.0;
    let gamma = 2.0;
    let rho = 0.5;
    let sigma = 0.5;

    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(n + 1);
    simplex.push(x.to_vec());
    for i in 0..n {
        let mut point = x.to_vec();
        let delta = if point[i].abs() > 1e-8 {
            0.05 * point[i].abs()
        } else {
            0.00025
        };
        point[i] += delta;
        simplex.push(point);
    }

    let mut fvals: Vec<f64> = simplex.iter().map(|p| obj(p)).collect();

    for _iter in 0..max_iter {
        let mut indices: Vec<usize> = (0..=n).collect();
        indices.sort_by(|&a, &b| fvals[a].partial_cmp(&fvals[b]).unwrap());

        let best = indices[0];
        let worst = indices[n];
        let second_worst = indices[n - 1];

        let frange = fvals[worst] - fvals[best];
        if frange < tol {
            x.copy_from_slice(&simplex[best]);
            return true;
        }

        let mut centroid = vec![0.0; n];
        for &idx in &indices[..n] {
            for j in 0..n {
                centroid[j] += simplex[idx][j];
            }
        }
        for j in 0..n {
            centroid[j] /= n as f64;
        }

        // Reflection
        let reflected: Vec<f64> = (0..n)
            .map(|j| centroid[j] + alpha * (centroid[j] - simplex[worst][j]))
            .collect();
        let fr = obj(&reflected);

        if fr < fvals[second_worst] && fr >= fvals[best] {
            simplex[worst] = reflected;
            fvals[worst] = fr;
            continue;
        }

        if fr < fvals[best] {
            let expanded: Vec<f64> = (0..n)
                .map(|j| centroid[j] + gamma * (reflected[j] - centroid[j]))
                .collect();
            let fe = obj(&expanded);
            if fe < fr {
                simplex[worst] = expanded;
                fvals[worst] = fe;
            } else {
                simplex[worst] = reflected;
                fvals[worst] = fr;
            }
            continue;
        }

        let contracted: Vec<f64> = (0..n)
            .map(|j| centroid[j] + rho * (simplex[worst][j] - centroid[j]))
            .collect();
        let fc = obj(&contracted);
        if fc < fvals[worst] {
            simplex[worst] = contracted;
            fvals[worst] = fc;
            continue;
        }

        let best_point = simplex[best].clone();
        for i in 0..=n {
            if i != best {
                for j in 0..n {
                    simplex[i][j] = best_point[j] + sigma * (simplex[i][j] - best_point[j]);
                }
                fvals[i] = obj(&simplex[i]);
            }
        }
    }

    let best = fvals
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    x.copy_from_slice(&simplex[best]);
    false
}

/// Backtracking line search with Armijo condition
fn backtracking_line_search(
    obj: &dyn Fn(&[f64]) -> f64,
    x: &[f64],
    d: &[f64],
    g: &[f64],
    n: usize,
) -> f64 {
    let c1 = 1e-4;
    let shrink = 0.5;
    let mut alpha = 1.0;
    let f0 = obj(x);
    let dg: f64 = d.iter().zip(g.iter()).map(|(di, gi)| di * gi).sum();

    let mut x_new = vec![0.0; n];
    for _ in 0..40 {
        for i in 0..n {
            x_new[i] = x[i] + alpha * d[i];
        }
        let f_new = obj(&x_new);
        if f_new <= f0 + c1 * alpha * dg {
            return alpha;
        }
        alpha *= shrink;
    }
    alpha
}

/// Central finite difference gradient (optimized step size)
fn gradient_fd(obj: &dyn Fn(&[f64]) -> f64, x: &[f64], n: usize) -> Vec<f64> {
    let t0 = std::time::Instant::now();
    let mut g = vec![0.0; n];
    let mut x_work = x.to_vec();
    for i in 0..n {
        let h = 1e-7 * (1.0 + x[i].abs());
        x_work[i] = x[i] + h;
        let fp = obj(&x_work);
        x_work[i] = x[i] - h;
        let fm = obj(&x_work);
        g[i] = (fp - fm) / (2.0 * h);
        x_work[i] = x[i];
    }
    GRADIENT_TIMINGS.record_fd(t0.elapsed().as_nanos() as u64);
    g
}

/// Compute Jacobian H = d(predictions)/d(eta) via finite differences.
/// H is n_obs x n_eta.
fn compute_jacobian_fd(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
) -> DMatrix<f64> {
    let n_obs = subject.obs_times.len();
    let n_eta = eta.len();
    let eps = 1e-6;

    let mut h = DMatrix::zeros(n_obs, n_eta);
    let mut eta_pert = eta.to_vec();

    for j in 0..n_eta {
        let h_step = eps * (1.0 + eta[j].abs());

        eta_pert[j] = eta[j] + h_step;
        let pk_plus = (model.pk_param_fn)(theta, &eta_pert, &subject.covariates);
        let preds_plus = if let Some(ref ode_spec) = model.ode_spec {
            pk::compute_predictions_ode(ode_spec, subject, &pk_plus.values)
        } else {
            pk::compute_predictions(model.pk_model, subject, &pk_plus)
        };

        eta_pert[j] = eta[j] - h_step;
        let pk_minus = (model.pk_param_fn)(theta, &eta_pert, &subject.covariates);
        let preds_minus = if let Some(ref ode_spec) = model.ode_spec {
            pk::compute_predictions_ode(ode_spec, subject, &pk_minus.values)
        } else {
            pk::compute_predictions(model.pk_model, subject, &pk_minus)
        };

        for i in 0..n_obs {
            h[(i, j)] = (preds_plus[i] - preds_minus[i]) / (2.0 * h_step);
        }

        eta_pert[j] = eta[j];
    }

    h
}

/// Run inner loop for all subjects (parallel via rayon).
/// Warm-starts from previous EBEs when available.
pub fn run_inner_loop(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, bool) {
    run_inner_loop_warm(model, population, params, max_iter, tol, None, None)
}

/// Run inner loop with optional warm-start EBEs and optional mu-referencing shift.
///
/// `prev_etas`: previous-iteration EBEs in eta_true space (used as warm starts).
/// `mu_k`: mu shift vector from `compute_mu_k`; `None` means no mu-referencing.
pub fn run_inner_loop_warm(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    max_iter: usize,
    tol: f64,
    prev_etas: Option<&[DVector<f64>]>,
    mu_k: Option<&[f64]>,
) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>, bool) {
    use rayon::prelude::*;

    let results: Vec<EbeResult> = population
        .subjects
        .par_iter()
        .enumerate()
        .map(|(i, subject)| {
            let init = prev_etas.map(|pe| pe[i].as_slice());
            find_ebe(model, subject, params, max_iter, tol, init, mu_k)
        })
        .collect();

    let any_failed = results.iter().any(|r| !r.converged);
    let eta_hats: Vec<DVector<f64>> = results.iter().map(|r| r.eta.clone()).collect();
    let h_matrices: Vec<DMatrix<f64>> = results.iter().map(|r| r.h_matrix.clone()).collect();

    (eta_hats, h_matrices, any_failed)
}
