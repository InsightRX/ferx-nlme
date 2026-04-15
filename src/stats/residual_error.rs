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
pub fn compute_r_diag(error_model: ErrorModel, ipreds: &[f64], sigma_values: &[f64]) -> Vec<f64> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_additive_variance() {
        let v = residual_variance(ErrorModel::Additive, 10.0, &[0.5]);
        assert_relative_eq!(v, 0.25, epsilon = 1e-12);
    }

    #[test]
    fn test_additive_variance_independent_of_prediction() {
        let v1 = residual_variance(ErrorModel::Additive, 1.0, &[0.5]);
        let v2 = residual_variance(ErrorModel::Additive, 100.0, &[0.5]);
        assert_relative_eq!(v1, v2, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance() {
        // V = (f * sigma)^2 = (10 * 0.1)^2 = 1.0
        let v = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        assert_relative_eq!(v, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn test_proportional_variance_scales_with_prediction() {
        let v1 = residual_variance(ErrorModel::Proportional, 10.0, &[0.1]);
        let v2 = residual_variance(ErrorModel::Proportional, 20.0, &[0.1]);
        assert_relative_eq!(v2 / v1, 4.0, epsilon = 1e-12);
    }

    #[test]
    fn test_combined_variance() {
        // V = (f * sigma1)^2 + sigma2^2 = (10 * 0.1)^2 + 0.5^2 = 1.0 + 0.25 = 1.25
        let v = residual_variance(ErrorModel::Combined, 10.0, &[0.1, 0.5]);
        assert_relative_eq!(v, 1.25, epsilon = 1e-12);
    }

    #[test]
    fn test_min_variance_floor() {
        // Proportional with f=0 gives V=0, should be floored to MIN_VARIANCE
        let v = residual_variance(ErrorModel::Proportional, 0.0, &[0.1]);
        assert_relative_eq!(v, MIN_VARIANCE, epsilon = 1e-20);
    }

    #[test]
    fn test_iwres_perfect_prediction() {
        let r = iwres(10.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_iwres_known_value() {
        // IWRES = (y - f) / sqrt(V) = (12 - 10) / sqrt(1) = 2.0
        let r = iwres(12.0, 10.0, ErrorModel::Additive, &[1.0]);
        assert_relative_eq!(r, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_r_diag_length() {
        let ipreds = vec![1.0, 2.0, 3.0];
        let r = compute_r_diag(ErrorModel::Additive, &ipreds, &[0.5]);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn test_compute_iwres_vectorized() {
        let obs = vec![12.0, 22.0];
        let ipreds = vec![10.0, 20.0];
        let result = compute_iwres(&obs, &ipreds, ErrorModel::Additive, &[1.0]);
        assert_eq!(result.len(), 2);
        assert_relative_eq!(result[0], 2.0, epsilon = 1e-12);
        assert_relative_eq!(result[1], 2.0, epsilon = 1e-12);
    }
}
