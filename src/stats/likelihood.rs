use crate::pk;
use crate::stats::residual_error::{compute_r_diag, residual_variance};
use crate::types::*;
use nalgebra::{DMatrix, DVector};

/// Route predictions through analytical PK or ODE solver depending on model.
fn model_predictions(model: &CompiledModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    if let Some(ref ode_spec) = model.ode_spec {
        // For ODE models, pass PK params as flat array to the RHS
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values)
    } else {
        pk::compute_predictions(model.pk_model, subject, pk_params)
    }
}

/// Compute individual negative log-likelihood for EBE estimation (inner loop objective).
///
/// NLL(eta | subject) = 0.5 * [eta'*Omega_inv*eta + log|Omega|
///                             + sum_j((y_j - f_j(eta))^2 / V_j + log(V_j))]
pub fn individual_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta: &[f64],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
) -> f64 {
    // Compute Omega inverse and log-determinant via Cholesky
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(chol) => chol.inverse(),
        None => return 1e20,
    };
    let log_det_omega = omega_log_det(omega);

    // Eta prior: eta' * Omega_inv * eta
    let eta_vec = DVector::from_column_slice(eta);
    let eta_prior = eta_vec.dot(&(&omega_inv * &eta_vec));

    // Compute individual PK parameters and predictions
    let pk_params = (model.pk_param_fn)(theta, eta, &subject.covariates);
    let preds = model_predictions(model, subject, &pk_params);
    let mut data_ll = 0.0;
    for (&y, &f_pred) in subject.observations.iter().zip(preds.iter()) {
        let v = residual_variance(model.error_model, f_pred, sigma_values);
        let resid = y - f_pred;
        data_ll += resid * resid / v + v.ln();
    }

    0.5 * (eta_prior + log_det_omega + data_ll)
}

/// Log-determinant of Omega via Cholesky: log|Omega| = 2 * sum(log(L_ii))
fn omega_log_det(omega: &OmegaMatrix) -> f64 {
    let n = omega.chol.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = omega.chol[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// FOCE per-subject negative log-likelihood (standard, no interaction).
///
/// NLL_i = 0.5 * [(y - f0)' * R_tilde_inv * (y - f0) + log|R_tilde|]
///
/// where f0 = f(eta_hat) - H * eta_hat  (linearized population prediction)
///       R_tilde = H * Omega * H' + R(f0)
pub fn foce_subject_nll(
    model: &CompiledModel,
    subject: &Subject,
    theta: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    // Individual predictions at eta_hat
    let pk_params = (model.pk_param_fn)(theta, eta_hat.as_slice(), &subject.covariates);
    let ipreds = model_predictions(model, subject, &pk_params);

    if interaction {
        // FOCEI: use individual predictions directly
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
        )
    } else {
        // Standard FOCE: linearized predictions
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
        )
    }
}

fn foce_subject_nll_standard(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
) -> f64 {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat (linearized population prediction)
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R diagonal at f0
    let r_diag = compute_r_diag(error_model, &f0, sigma_values);

    // R_tilde = H * Omega * H' + diag(R)
    let r_tilde = compute_r_tilde(h_matrix, &omega.matrix, &r_diag);

    // Cholesky of R_tilde
    let chol = match r_tilde.clone().cholesky() {
        Some(c) => c,
        None => return 1e20,
    };

    // Residuals: y - f0
    let residuals: DVector<f64> = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(f0.iter())
            .map(|(&y, &f)| y - f),
    );

    // (y - f0)' * R_tilde_inv * (y - f0)
    let solved = chol.solve(&residuals);
    let quad_form = residuals.dot(&solved);

    // log|R_tilde|
    let log_det_r = chol_log_det(&chol.l());

    0.5 * (quad_form + log_det_r)
}

fn foce_subject_nll_interaction(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
) -> f64 {
    let n_obs = subject.observations.len();

    // R diagonal at ipred (not linearized)
    let r_diag = compute_r_diag(error_model, ipreds, sigma_values);

    // R_tilde = H * Omega * H' + diag(R(ipred))
    let r_tilde = compute_r_tilde(h_matrix, &omega.matrix, &r_diag);

    let chol = match r_tilde.clone().cholesky() {
        Some(c) => c,
        None => return 1e20,
    };

    // Residuals: y - ipred
    let residuals: DVector<f64> = DVector::from_iterator(
        n_obs,
        subject
            .observations
            .iter()
            .zip(ipreds.iter())
            .map(|(&y, &f)| y - f),
    );

    // Data term: (y - ipred)' * V_inv * (y - ipred) using diagonal V
    let mut data_term = 0.0;
    for j in 0..n_obs {
        let resid = residuals[j];
        data_term += resid * resid / r_diag[j];
    }

    // Prior term: eta_hat' * Omega_inv * eta_hat
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(c) => c.inverse(),
        None => return 1e20,
    };
    let eta_prior = eta_hat.dot(&(&omega_inv * eta_hat));

    // log|R_tilde|
    let log_det = chol_log_det(&chol.l());

    0.5 * (data_term + eta_prior + log_det)
}

/// R_tilde = H * Omega * H' + diag(r_diag)
fn compute_r_tilde(h: &DMatrix<f64>, omega: &DMatrix<f64>, r_diag: &[f64]) -> DMatrix<f64> {
    let n_obs = h.nrows();
    let h_omega = h * omega;
    let mut r_tilde = &h_omega * h.transpose();
    for j in 0..n_obs {
        r_tilde[(j, j)] += r_diag[j];
    }
    r_tilde
}

/// log-determinant from Cholesky factor L: 2 * sum(log(L_ii))
fn chol_log_det(l: &DMatrix<f64>) -> f64 {
    let n = l.nrows();
    let mut ld = 0.0;
    for i in 0..n {
        let lii = l[(i, i)];
        if lii > 0.0 {
            ld += lii.ln();
        } else {
            return 1e20;
        }
    }
    2.0 * ld
}

/// Population FOCE objective: sum over all subjects
pub fn foce_population_nll(
    model: &CompiledModel,
    population: &Population,
    theta: &[f64],
    eta_hats: &[DVector<f64>],
    h_matrices: &[DMatrix<f64>],
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    interaction: bool,
) -> f64 {
    population
        .subjects
        .iter()
        .enumerate()
        .map(|(i, subject)| {
            foce_subject_nll(
                model,
                subject,
                theta,
                &eta_hats[i],
                &h_matrices[i],
                omega,
                sigma_values,
                interaction,
            )
        })
        .sum::<f64>()
}

/// Compute CWRES (Conditional Weighted Residuals) for a subject
pub fn compute_cwres(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
) -> Vec<f64> {
    let n_obs = subject.observations.len();

    // f0 = ipred - H * eta_hat
    let h_eta = h_matrix * eta_hat;
    let f0: Vec<f64> = ipreds
        .iter()
        .enumerate()
        .map(|(j, &ip)| ip - h_eta[j])
        .collect();

    // R_tilde
    let r_diag = compute_r_diag(error_model, &f0, sigma_values);
    let r_tilde = compute_r_tilde(h_matrix, &omega.matrix, &r_diag);

    // CWRES_j = (y_j - f0_j) / sqrt(R_tilde_jj)
    (0..n_obs)
        .map(|j| {
            let resid = subject.observations[j] - f0[j];
            let var = r_tilde[(j, j)].max(1e-12);
            resid / var.sqrt()
        })
        .collect()
}
