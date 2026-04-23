use crate::pk;
use crate::stats::residual_error::{compute_r_diag, residual_variance};
use crate::stats::special::log_normal_cdf;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;

/// Route predictions through analytical PK or ODE solver depending on model.
fn model_predictions(model: &CompiledModel, subject: &Subject, pk_params: &PkParams) -> Vec<f64> {
    if let Some(ref ode_spec) = model.ode_spec {
        // For ODE models, pass PK params as flat array to the RHS
        pk::compute_predictions_ode(ode_spec, subject, &pk_params.values)
    } else {
        pk::compute_predictions(model.pk_model, subject, pk_params)
    }
}

/// True when observation `j` of `subject` is censored AND the model requests M3.
fn is_m3_bloq(model: &CompiledModel, subject: &Subject, j: usize) -> bool {
    matches!(model.bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0
}

/// Compute individual negative log-likelihood for EBE estimation (inner loop objective).
///
/// NLL(eta | subject) = 0.5 * [eta'*Omega_inv*eta + log|Omega|
///                             + sum_j( term_j )]
/// where term_j is:
///   - `(y_j - f_j)² / V_j + log(V_j)` for quantified observations, or
///   - `-2·log Φ((LLOQ_j - f_j)/√V_j)` for M3-censored observations (CENS=1)
///     with LLOQ_j carried in `observations[j]`.
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
    for (j, (&y, &f_pred)) in subject.observations.iter().zip(preds.iter()).enumerate() {
        let v = residual_variance(model.error_model, f_pred, sigma_values);
        if is_m3_bloq(model, subject, j) {
            // y carries LLOQ on CENS=1 rows.
            let z = (y - f_pred) / v.sqrt();
            data_ll += -2.0 * log_normal_cdf(z);
        } else {
            let resid = y - f_pred;
            data_ll += resid * resid / v + v.ln();
        }
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

/// FOCE per-subject negative log-likelihood.
///
/// Non-interaction (standard FOCE):
///   NLL_i = 0.5 * [(y - f0)' * R_tilde_inv * (y - f0) + log|R_tilde|]
///   where f0 = f(eta_hat) - H * eta_hat  (linearized population prediction)
///         R_tilde = H * Omega * H' + R(f0)
///
/// When M3 BLOQ is active and the subject has any CENS=1 row, we route through
/// the interaction path: mixing a linearized Gaussian term with a non-linearized
/// `log Φ(·)` BLOQ term produces inconsistent OFVs near the LLOQ boundary, so we
/// promote the whole subject to FOCEI — which is what NONMEM LAPLACE+M3 does in
/// practice.
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

    let m3_active = matches!(model.bloq_method, BloqMethod::M3) && subject.has_bloq();

    if interaction || m3_active {
        foce_subject_nll_interaction(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
            model.bloq_method,
        )
    } else {
        foce_subject_nll_standard(
            subject,
            &ipreds,
            eta_hat,
            h_matrix,
            omega,
            sigma_values,
            model.error_model,
            model.bloq_method,
        )
    }
}

/// Standard FOCE (no interaction). When any CENS rows are present AND
/// `bloq_method == M3`, the dispatcher has already routed to the interaction
/// path — so inside this function the only case we need to handle is
/// `bloq_method == Drop` (treat CENS rows as ordinary obs) or no CENS at all.
pub fn foce_subject_nll_standard(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
    _bloq_method: BloqMethod,
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

/// FOCEI per-subject NLL. With `bloq_method == M3`, BLOQ observations are
/// dropped from the Gaussian residual sum and the R_tilde Cholesky, and instead
/// contribute `-2·log Φ((LLOQ - f)/√V)` evaluated at η̂ (scalar, independent
/// of the other observations given η̂).
pub fn foce_subject_nll_interaction(
    subject: &Subject,
    ipreds: &[f64],
    eta_hat: &DVector<f64>,
    h_matrix: &DMatrix<f64>,
    omega: &OmegaMatrix,
    sigma_values: &[f64],
    error_model: ErrorModel,
    bloq_method: BloqMethod,
) -> f64 {
    let n_obs = subject.observations.len();

    // Partition observation indices into quantified vs BLOQ.
    let (quant_idx, bloq_idx): (Vec<usize>, Vec<usize>) = (0..n_obs).partition(|&j| {
        !(matches!(bloq_method, BloqMethod::M3) && subject.cens.get(j).copied().unwrap_or(0) != 0)
    });

    let n_quant = quant_idx.len();

    // Build sub-H over quantified rows.
    let n_eta = eta_hat.len();
    let h_quant = DMatrix::from_fn(n_quant, n_eta, |r, c| h_matrix[(quant_idx[r], c)]);
    let ipreds_quant: Vec<f64> = quant_idx.iter().map(|&j| ipreds[j]).collect();

    let r_diag_quant = compute_r_diag(error_model, &ipreds_quant, sigma_values);

    // R_tilde over quantified rows: H_q · Ω · H_qᵀ + diag(R_q)
    let r_tilde = compute_r_tilde(&h_quant, &omega.matrix, &r_diag_quant);

    let log_det = if n_quant > 0 {
        let chol = match r_tilde.clone().cholesky() {
            Some(c) => c,
            None => return 1e20,
        };
        chol_log_det(&chol.l())
    } else {
        0.0
    };

    // Gaussian residual sum over quantified rows, using diagonal R(ipred).
    let mut data_term = 0.0;
    for (k, &j) in quant_idx.iter().enumerate() {
        let resid = subject.observations[j] - ipreds[j];
        data_term += resid * resid / r_diag_quant[k];
    }

    // BLOQ contributions: -2·log Φ((lloq - f)/√V) at η̂ (ipred-based variance).
    for &j in &bloq_idx {
        let lloq = subject.observations[j];
        let f = ipreds[j];
        let v = residual_variance(error_model, f, sigma_values);
        let z = (lloq - f) / v.sqrt();
        data_term += -2.0 * log_normal_cdf(z);
    }

    // Prior term: eta_hat' * Omega_inv * eta_hat
    let omega_inv = match omega.matrix.clone().cholesky() {
        Some(c) => c.inverse(),
        None => return 1e20,
    };
    let eta_prior = eta_hat.dot(&(&omega_inv * eta_hat));

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
        .par_iter()
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

/// Compute CWRES (Conditional Weighted Residuals) for a subject.
/// BLOQ observations get `NaN` since a weighted Gaussian residual is undefined
/// when the observed value is censored.
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

    // CWRES_j = (y_j - f0_j) / sqrt(R_tilde_jj), or NaN if censored.
    (0..n_obs)
        .map(|j| {
            if subject.cens.get(j).copied().unwrap_or(0) != 0 {
                f64::NAN
            } else {
                let resid = subject.observations[j] - f0[j];
                let var = r_tilde[(j, j)].max(1e-12);
                resid / var.sqrt()
            }
        })
        .collect()
}
