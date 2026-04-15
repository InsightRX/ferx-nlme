//! Automatic differentiation gradient functions using `std::autodiff`.
//!
//! The AD functions take `tv_adjusted: &[f64]` — pre-computed typical values
//! that already incorporate covariates and theta. The inner loop computes:
//!   PK_param[i] = tv[i] * exp(eta[i])
//! so only eta is differentiated.

use crate::types::*;
use std::autodiff::{autodiff_forward, autodiff_reverse};

// ─── Individual NLL: reverse-mode AD for gradient w.r.t. eta ────────────────

#[autodiff_reverse(
    individual_nll_ad_grad,
    Duplicated,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Active
)]
pub fn individual_nll_ad(
    eta: &[f64],
    tv: &[f64],             // covariate-adjusted typical values, length n_eta
    omega_inv_flat: &[f64], // n_eta*n_eta, row-major
    log_det_omega: f64,
    sigma_values: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    obs_times: &[f64],
    observations: &[f64],
    pk_idx_f64: &[f64],    // PK parameter indices as f64 (cast to usize inside)
    pk_and_err_model: f64, // pk_model_id * 10 + error_model_id
) -> f64 {
    let n_eta = eta.len();
    let n_doses = dose_times.len();
    let n_obs = obs_times.len();
    let pk_model_id = (pk_and_err_model as i32) / 10;
    let error_model_id = (pk_and_err_model as i32) % 10;

    // Eta prior: eta' * Omega_inv * eta
    let mut eta_prior = 0.0;
    for i in 0..n_eta {
        for j in 0..n_eta {
            eta_prior += eta[i] * omega_inv_flat[i * n_eta + j] * eta[j];
        }
    }

    // PK params: tv[i] * exp(eta[i]), placed at correct PK index
    let mut pk = [0.0f64; MAX_PK_PARAMS];
    pk[PK_IDX_F] = 1.0;
    for i in 0..n_eta {
        let idx = pk_idx_f64[i] as usize;
        pk[idx] = tv[i] * eta[i].exp();
    }

    // Predictions + data likelihood
    let mut data_ll = 0.0;
    for obs_idx in 0..n_obs {
        let t = obs_times[obs_idx];
        let mut conc = 0.0;
        for d in 0..n_doses {
            if dose_times[d] <= t {
                let tau = t - dose_times[d];
                conc += single_dose_ad(
                    pk_model_id,
                    tau,
                    dose_amts[d],
                    dose_rates[d],
                    dose_durations[d],
                    pk[PK_IDX_CL],
                    pk[PK_IDX_V],
                    pk[PK_IDX_Q],
                    pk[PK_IDX_V2],
                    pk[PK_IDX_KA],
                    pk[PK_IDX_F],
                );
            }
        }
        if conc < 0.0 {
            conc = 0.0;
        }

        let v = residual_variance_ad(error_model_id, conc, sigma_values);
        let resid = observations[obs_idx] - conc;
        data_ll += resid * resid / v + v.ln();
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

// ─── Predictions: forward-mode AD for Jacobian ─────────────────────────────

#[autodiff_forward(
    predict_all_ad_tangent,
    Dual,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Const,
    Dual
)]
pub fn predict_all_ad(
    eta: &[f64],
    tv: &[f64],
    dose_times: &[f64],
    dose_amts: &[f64],
    dose_rates: &[f64],
    dose_durations: &[f64],
    obs_times: &[f64],
    pk_idx_f64: &[f64], // PK parameter indices as f64 (cast to usize inside)
    pk_model_id: f64,
    out: &mut [f64],
) {
    let n_eta = eta.len();
    let n_doses = dose_times.len();
    let n_obs = obs_times.len();
    let pk_id = pk_model_id as i32;

    let mut pk = [0.0f64; MAX_PK_PARAMS];
    pk[PK_IDX_F] = 1.0;
    for i in 0..n_eta {
        let idx = pk_idx_f64[i] as usize;
        pk[idx] = tv[i] * eta[i].exp();
    }

    for obs_idx in 0..n_obs {
        let t = obs_times[obs_idx];
        let mut conc = 0.0;
        for d in 0..n_doses {
            if dose_times[d] <= t {
                let tau = t - dose_times[d];
                conc += single_dose_ad(
                    pk_id,
                    tau,
                    dose_amts[d],
                    dose_rates[d],
                    dose_durations[d],
                    pk[PK_IDX_CL],
                    pk[PK_IDX_V],
                    pk[PK_IDX_Q],
                    pk[PK_IDX_V2],
                    pk[PK_IDX_KA],
                    pk[PK_IDX_F],
                );
            }
        }
        out[obs_idx] = if conc > 0.0 { conc } else { 0.0 };
    }
}

// ─── Inlined PK equations ───────────────────────────────────────────────────

fn single_dose_ad(
    pk_model_id: i32,
    tau: f64,
    amt: f64,
    rate: f64,
    dur: f64,
    cl: f64,
    v: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if tau < 0.0 || v <= 0.0 || cl <= 0.0 {
        return 0.0;
    }

    match pk_model_id {
        0 => {
            // OneCptIvBolus
            let k = cl / v;
            (amt / v) * (-k * tau).exp()
        }
        1 => {
            // OneCptOral
            let k = cl / v;
            let d = f_bio * amt;
            if (ka - k).abs() < 1e-6 {
                (d * ka / v) * tau * (-k * tau).exp()
            } else {
                (d * ka / (v * (ka - k))) * ((-k * tau).exp() - (-ka * tau).exp())
            }
        }
        2 => {
            // OneCptInfusion
            let k = cl / v;
            if dur <= 0.0 {
                (amt / v) * (-k * tau).exp()
            } else if tau <= dur {
                (rate / cl) * (1.0 - (-k * tau).exp())
            } else {
                (rate / cl) * (1.0 - (-k * dur).exp()) * (-k * (tau - dur)).exp()
            }
        }
        3 => {
            // TwoCptIvBolus
            let (alpha, beta, k21) = macro_rates(cl, v, q, v2);
            let diff = alpha - beta;
            if diff.abs() < 1e-12 {
                return 0.0;
            }
            let a = (amt / v) * (alpha - k21) / diff;
            let b = (amt / v) * (k21 - beta) / diff;
            a * (-alpha * tau).exp() + b * (-beta * tau).exp()
        }
        4 => {
            // TwoCptOral
            let (alpha, beta, k21) = macro_rates(cl, v, q, v2);
            let diff = alpha - beta;
            if diff.abs() < 1e-12 {
                return 0.0;
            }
            let coeff = f_bio * amt * ka / v;
            let p = if (ka - alpha).abs() < 1e-6 {
                coeff * (alpha - k21) / diff * tau * (-alpha * tau).exp()
            } else {
                coeff * (k21 - alpha) / ((ka - alpha) * (beta - alpha)) * (-alpha * tau).exp()
            };
            let q_val = if (ka - beta).abs() < 1e-6 {
                coeff * (k21 - beta) / diff * tau * (-beta * tau).exp()
            } else {
                coeff * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-beta * tau).exp()
            };
            let r = if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
                0.0
            } else {
                coeff * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-ka * tau).exp()
            };
            p + q_val + r
        }
        5 => {
            // TwoCptInfusion
            let (alpha, beta, k21) = macro_rates(cl, v, q, v2);
            let diff = alpha - beta;
            if diff.abs() < 1e-12 || alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
                return 0.0;
            }
            if dur <= 0.0 {
                let a = (amt / v) * (alpha - k21) / diff;
                let b = (amt / v) * (k21 - beta) / diff;
                a * (-alpha * tau).exp() + b * (-beta * tau).exp()
            } else {
                let a_c = (rate / v) * (alpha - k21) / (diff * alpha);
                let b_c = (rate / v) * (k21 - beta) / (diff * beta);
                if tau <= dur {
                    a_c * (1.0 - (-alpha * tau).exp()) + b_c * (1.0 - (-beta * tau).exp())
                } else {
                    let dt = tau - dur;
                    a_c * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
                        + b_c * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
                }
            }
        }
        _ => 0.0,
    }
}

fn macro_rates(cl: f64, v1: f64, q: f64, v2: f64) -> (f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let sq = s * s - 4.0 * d;
    let disc = (if sq > 0.0 { sq } else { 0.0 }).sqrt();
    let alpha = (s + disc) / 2.0;
    let beta = if alpha > 1e-30 { d / alpha } else { 0.0 };
    (alpha, beta, k21)
}

fn residual_variance_ad(error_model_id: i32, f_pred: f64, sigma: &[f64]) -> f64 {
    let v = match error_model_id {
        0 => sigma[0] * sigma[0],
        1 => {
            let fs = f_pred * sigma[0];
            fs * fs
        }
        2 => {
            let p = f_pred * sigma[0];
            p * p + sigma[1] * sigma[1]
        }
        _ => sigma[0] * sigma[0],
    };
    if v < 1e-12 {
        1e-12
    } else {
        v
    }
}

// ─── Enum → ID converters ───────────────────────────────────────────────────

pub fn pk_model_to_id(m: PkModel) -> i32 {
    match m {
        PkModel::OneCptIvBolus => 0,
        PkModel::OneCptOral => 1,
        PkModel::OneCptInfusion => 2,
        PkModel::TwoCptIvBolus => 3,
        PkModel::TwoCptOral => 4,
        PkModel::TwoCptInfusion => 5,
    }
}

pub fn error_model_to_id(m: ErrorModel) -> i32 {
    match m {
        ErrorModel::Additive => 0,
        ErrorModel::Proportional => 1,
        ErrorModel::Combined => 2,
    }
}

// ─── Flat dose data ─────────────────────────────────────────────────────────

pub struct FlatDoseData {
    pub times: Vec<f64>,
    pub amts: Vec<f64>,
    pub rates: Vec<f64>,
    pub durations: Vec<f64>,
}

impl FlatDoseData {
    pub fn from_subject(subject: &Subject) -> Self {
        Self {
            times: subject.doses.iter().map(|d| d.time).collect(),
            amts: subject.doses.iter().map(|d| d.amt).collect(),
            rates: subject.doses.iter().map(|d| d.rate).collect(),
            durations: subject.doses.iter().map(|d| d.duration).collect(),
        }
    }
}

// ─── Public interface ───────────────────────────────────────────────────────

/// Compute gradient of individual_nll w.r.t. eta using reverse-mode AD.
/// `tv_adjusted` = covariate-adjusted typical values (length n_eta).
pub fn compute_nll_gradient_ad(
    eta: &[f64],
    tv_adjusted: &[f64],
    omega_inv_flat: &[f64],
    log_det_omega: f64,
    sigma_values: &[f64],
    dose_data: &FlatDoseData,
    obs_times: &[f64],
    observations: &[f64],
    pk_model: PkModel,
    error_model: ErrorModel,
    pk_indices: &[usize],
) -> (f64, Vec<f64>) {
    let n_eta = eta.len();
    let mut d_eta = vec![0.0f64; n_eta];

    let pk_and_err = (pk_model_to_id(pk_model) * 10 + error_model_to_id(error_model)) as f64;
    let pk_idx_f64: Vec<f64> = pk_indices.iter().map(|&i| i as f64).collect();

    let nll = individual_nll_ad_grad(
        eta,
        &mut d_eta,
        tv_adjusted,
        omega_inv_flat,
        log_det_omega,
        sigma_values,
        &dose_data.times,
        &dose_data.amts,
        &dose_data.rates,
        &dose_data.durations,
        obs_times,
        observations,
        &pk_idx_f64,
        pk_and_err,
        1.0,
    );

    (nll, d_eta)
}

/// Compute Jacobian d(predictions)/d(eta) using forward-mode AD.
pub fn compute_jacobian_ad(
    eta: &[f64],
    tv_adjusted: &[f64],
    dose_data: &FlatDoseData,
    obs_times: &[f64],
    n_obs: usize,
    pk_model: PkModel,
    pk_indices: &[usize],
) -> nalgebra::DMatrix<f64> {
    let n_eta = eta.len();
    let pk_id = pk_model_to_id(pk_model) as f64;
    let pk_idx_f64: Vec<f64> = pk_indices.iter().map(|&i| i as f64).collect();
    let mut jac = nalgebra::DMatrix::zeros(n_obs, n_eta);

    for j in 0..n_eta {
        let mut d_eta = vec![0.0f64; n_eta];
        d_eta[j] = 1.0;

        let mut out = vec![0.0f64; n_obs];
        let mut d_out = vec![0.0f64; n_obs];

        predict_all_ad_tangent(
            eta,
            &d_eta,
            tv_adjusted,
            &dose_data.times,
            &dose_data.amts,
            &dose_data.rates,
            &dose_data.durations,
            obs_times,
            &pk_idx_f64,
            pk_id,
            &mut out,
            &mut d_out,
        );

        for i in 0..n_obs {
            jac[(i, j)] = d_out[i];
        }
    }

    jac
}
