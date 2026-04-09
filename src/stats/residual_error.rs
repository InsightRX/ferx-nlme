use crate::types::ErrorModel;

const MIN_VARIANCE: f64 = 1e-12;

/// Compute residual variance for a single observation
/// sigma_values: [sigma1] for additive/proportional, [sigma1, sigma2] for combined
pub fn residual_variance(error_model: ErrorModel, f_pred: f64, sigma_values: &[f64]) -> f64 {
    let v = match error_model {
        ErrorModel::Additive => {
            // V = sigma1^2
            sigma_values[0] * sigma_values[0]
        }
        ErrorModel::Proportional => {
            // V = (f * sigma1)^2
            let fs = f_pred * sigma_values[0];
            fs * fs
        }
        ErrorModel::Combined => {
            // V = (f * sigma1)^2 + sigma2^2
            let prop = f_pred * sigma_values[0];
            prop * prop + sigma_values[1] * sigma_values[1]
        }
    };
    v.max(MIN_VARIANCE)
}

/// Compute the R diagonal (vector of residual variances for all observations)
pub fn compute_r_diag(
    error_model: ErrorModel,
    ipreds: &[f64],
    sigma_values: &[f64],
) -> Vec<f64> {
    ipreds
        .iter()
        .map(|&f| residual_variance(error_model, f, sigma_values))
        .collect()
}

/// Individual weighted residual: IWRES_j = (y_j - f_j) / sqrt(V_j)
pub fn iwres(obs: f64, ipred: f64, error_model: ErrorModel, sigma_values: &[f64]) -> f64 {
    let v = residual_variance(error_model, ipred, sigma_values);
    (obs - ipred) / v.sqrt()
}

/// Compute IWRES for all observations
pub fn compute_iwres(
    observations: &[f64],
    ipreds: &[f64],
    error_model: ErrorModel,
    sigma_values: &[f64],
) -> Vec<f64> {
    observations
        .iter()
        .zip(ipreds.iter())
        .map(|(&y, &f)| iwres(y, f, error_model, sigma_values))
        .collect()
}
