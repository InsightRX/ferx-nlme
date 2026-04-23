use crate::types::{CompiledModel, ModelParameters, OmegaMatrix, SigmaVector};
use nalgebra::DMatrix;

/// Bounds for the packed parameter vector
pub struct PackedBounds {
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

/// Pack ModelParameters into a flat unconstrained vector for optimization.
///
/// Layout: [log(theta_1), ..., log(theta_n),
///          log(L_11), L_21, log(L_22), ...,   (Cholesky lower triangle)
///          log(sigma_1), ..., log(sigma_m)]
pub fn pack_params(params: &ModelParameters) -> Vec<f64> {
    let mut v = Vec::new();

    // Theta: log-transformed
    for &th in &params.theta {
        v.push(th.max(1e-10).ln());
    }

    // Omega Cholesky factor: diagonal as log, off-diagonal as-is
    let l = &params.omega.chol;
    let n_eta = l.nrows();
    if params.omega.diagonal {
        for i in 0..n_eta {
            v.push(l[(i, i)].max(1e-10).ln());
        }
    } else {
        for j in 0..n_eta {
            for i in j..n_eta {
                if i == j {
                    v.push(l[(i, j)].max(1e-10).ln());
                } else {
                    v.push(l[(i, j)]);
                }
            }
        }
    }

    // Sigma: log-transformed
    for &s in &params.sigma.values {
        v.push(s.max(1e-10).ln());
    }

    v
}

/// Unpack a flat unconstrained vector back into ModelParameters.
pub fn unpack_params(v: &[f64], template: &ModelParameters) -> ModelParameters {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();
    let mut idx = 0;

    // Theta
    let theta: Vec<f64> = (0..n_theta)
        .map(|_| {
            let val = v[idx].exp();
            idx += 1;
            val
        })
        .collect();

    // Omega Cholesky
    let mut l = DMatrix::zeros(n_eta, n_eta);
    if template.omega.diagonal {
        for i in 0..n_eta {
            l[(i, i)] = v[idx].exp();
            idx += 1;
        }
    } else {
        for j in 0..n_eta {
            for i in j..n_eta {
                if i == j {
                    l[(i, j)] = v[idx].exp();
                } else {
                    l[(i, j)] = v[idx];
                }
                idx += 1;
            }
        }
    }
    let omega_matrix = &l * l.transpose();
    let omega = OmegaMatrix {
        matrix: omega_matrix,
        chol: l,
        eta_names: template.omega.eta_names.clone(),
        diagonal: template.omega.diagonal,
    };

    // Sigma
    let sigma_values: Vec<f64> = (0..n_sigma)
        .map(|_| {
            let val = v[idx].exp();
            idx += 1;
            val
        })
        .collect();
    let sigma = SigmaVector {
        values: sigma_values,
        names: template.sigma.names.clone(),
    };

    ModelParameters {
        theta,
        theta_names: template.theta_names.clone(),
        theta_lower: template.theta_lower.clone(),
        theta_upper: template.theta_upper.clone(),
        omega,
        sigma,
    }
}

/// Compute the number of packed parameters
pub fn packed_len(template: &ModelParameters) -> usize {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_omega = if template.omega.diagonal {
        n_eta
    } else {
        n_eta * (n_eta + 1) / 2
    };
    let n_sigma = template.sigma.values.len();
    n_theta + n_omega + n_sigma
}

/// Compute box constraints for the packed parameter vector
pub fn compute_bounds(template: &ModelParameters) -> PackedBounds {
    let n_theta = template.theta.len();
    let n_eta = template.omega.dim();
    let n_sigma = template.sigma.values.len();

    let mut lower = Vec::new();
    let mut upper = Vec::new();

    // Theta bounds (log-transformed)
    for i in 0..n_theta {
        lower.push(template.theta_lower[i].max(1e-10).ln());
        upper.push(template.theta_upper[i].min(1e9).ln());
    }

    // Omega Cholesky bounds
    if template.omega.diagonal {
        for _ in 0..n_eta {
            lower.push(-6.0); // exp(-6) ≈ 0.0025
            upper.push(4.0); // exp(4) ≈ 55
        }
    } else {
        for j in 0..n_eta {
            for i in j..n_eta {
                if i == j {
                    lower.push(-6.0);
                    upper.push(4.0);
                } else {
                    lower.push(-10.0);
                    upper.push(10.0);
                }
            }
        }
    }

    // Sigma bounds (log-transformed)
    for _ in 0..n_sigma {
        lower.push(-8.0); // exp(-8) ≈ 3e-4
        upper.push(5.0); // exp(5) ≈ 148
    }

    PackedBounds { lower, upper }
}

/// Return initial ETA vector: warm-start if available, else mu_refs, else zeros.
pub fn get_eta_init(
    n_eta: usize,
    warm_start: Option<&[f64]>,
    mu_refs: Option<&[f64]>,
) -> Vec<f64> {
    if let Some(ws) = warm_start {
        ws.to_vec()
    } else if let Some(mu) = mu_refs {
        mu.to_vec()
    } else {
        vec![0.0; n_eta]
    }
}

/// Compute the mu_k shift vector from current theta for mu-referenced ETAs.
///
/// For each ETA that has a detected mu-reference, mu[i] = log(theta) or theta
/// depending on whether the relationship is log-transformed.  ETAs without a
/// mu-reference get mu[i] = 0 (no shift), preserving the standard behaviour.
/// When `enabled` is false, returns a zero vector (disables mu-referencing).
pub fn compute_mu_k(model: &CompiledModel, theta: &[f64], enabled: bool) -> Vec<f64> {
    if !enabled {
        return vec![0.0; model.n_eta];
    }
    let mut mu = vec![0.0; model.n_eta];
    for (eta_idx, eta_name) in model.eta_names.iter().enumerate() {
        if let Some(mu_ref) = model.mu_refs.get(eta_name) {
            if let Some(theta_idx) = model
                .theta_names
                .iter()
                .position(|n| n == &mu_ref.theta_name)
            {
                let theta_val = theta[theta_idx];
                mu[eta_idx] = if mu_ref.log_transformed {
                    theta_val.max(1e-10).ln()
                } else {
                    theta_val
                };
            }
        }
    }
    mu
}

/// Clamp a vector to box constraints
pub fn clamp_to_bounds(x: &mut [f64], bounds: &PackedBounds) {
    for i in 0..x.len() {
        x[i] = x[i].clamp(bounds.lower[i], bounds.upper[i]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn make_template() -> ModelParameters {
        let omega =
            OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["eta_cl".into(), "eta_v".into()]);
        let sigma = SigmaVector {
            values: vec![0.3],
            names: vec!["sigma_prop".into()],
        };
        ModelParameters {
            theta: vec![10.0, 100.0],
            theta_names: vec!["cl".into(), "v".into()],
            theta_lower: vec![0.01, 0.01],
            theta_upper: vec![1000.0, 10000.0],
            omega,
            sigma,
        }
    }

    #[test]
    fn test_packed_len_diagonal() {
        let template = make_template();
        // 2 theta + 2 diagonal omega + 1 sigma = 5
        assert_eq!(packed_len(&template), 5);
    }

    #[test]
    fn test_pack_unpack_round_trip() {
        let template = make_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);

        // Theta values should round-trip
        for (orig, rec) in template.theta.iter().zip(recovered.theta.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }

        // Omega diagonal should round-trip
        let n = template.omega.dim();
        for i in 0..n {
            assert_relative_eq!(
                template.omega.matrix[(i, i)],
                recovered.omega.matrix[(i, i)],
                epsilon = 1e-8
            );
        }

        // Sigma should round-trip
        for (orig, rec) in template
            .sigma
            .values
            .iter()
            .zip(recovered.sigma.values.iter())
        {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_pack_values_are_log_transformed() {
        let template = make_template();
        let packed = pack_params(&template);
        // First packed value should be log(theta[0]) = log(10)
        assert_relative_eq!(packed[0], 10.0_f64.ln(), epsilon = 1e-10);
        assert_relative_eq!(packed[1], 100.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn test_compute_bounds_dimensions() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let expected_len = packed_len(&template);
        assert_eq!(bounds.lower.len(), expected_len);
        assert_eq!(bounds.upper.len(), expected_len);
    }

    #[test]
    fn test_bounds_lower_less_than_upper() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        for (lo, hi) in bounds.lower.iter().zip(bounds.upper.iter()) {
            assert!(lo < hi, "lower {} should be < upper {}", lo, hi);
        }
    }

    #[test]
    fn test_clamp_to_bounds() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let mut x = vec![100.0; packed_len(&template)]; // way above upper bounds
        clamp_to_bounds(&mut x, &bounds);
        for (val, hi) in x.iter().zip(bounds.upper.iter()) {
            assert!(*val <= *hi + 1e-12);
        }
    }

    #[test]
    fn test_clamp_to_bounds_below() {
        let template = make_template();
        let bounds = compute_bounds(&template);
        let mut x = vec![-100.0; packed_len(&template)]; // way below lower bounds
        clamp_to_bounds(&mut x, &bounds);
        for (val, lo) in x.iter().zip(bounds.lower.iter()) {
            assert!(*val >= *lo - 1e-12);
        }
    }

    fn make_block_template() -> ModelParameters {
        // Build a 2x2 block omega with covariance
        let mut m = DMatrix::zeros(2, 2);
        m[(0, 0)] = 0.09; // var(eta_cl)
        m[(1, 1)] = 0.04; // var(eta_v)
        m[(0, 1)] = 0.02; // cov(eta_cl, eta_v)
        m[(1, 0)] = 0.02;
        let omega = OmegaMatrix::from_matrix(m, vec!["eta_cl".into(), "eta_v".into()], false);
        let sigma = SigmaVector {
            values: vec![0.3],
            names: vec!["sigma_prop".into()],
        };
        ModelParameters {
            theta: vec![10.0, 100.0],
            theta_names: vec!["cl".into(), "v".into()],
            theta_lower: vec![0.01, 0.01],
            theta_upper: vec![1000.0, 10000.0],
            omega,
            sigma,
        }
    }

    #[test]
    fn test_packed_len_block() {
        let template = make_block_template();
        // 2 theta + 3 omega (lower triangle of 2x2) + 1 sigma = 6
        assert_eq!(packed_len(&template), 6);
    }

    #[test]
    fn test_pack_unpack_block_round_trip() {
        let template = make_block_template();
        let packed = pack_params(&template);
        assert_eq!(packed.len(), packed_len(&template));

        let recovered = unpack_params(&packed, &template);

        // Theta round-trip
        for (orig, rec) in template.theta.iter().zip(recovered.theta.iter()) {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }

        // Full omega matrix round-trip (including off-diagonals)
        let n = template.omega.dim();
        for i in 0..n {
            for j in 0..n {
                assert_relative_eq!(
                    template.omega.matrix[(i, j)],
                    recovered.omega.matrix[(i, j)],
                    epsilon = 1e-6
                );
            }
        }

        // Sigma round-trip
        for (orig, rec) in template
            .sigma
            .values
            .iter()
            .zip(recovered.sigma.values.iter())
        {
            assert_relative_eq!(orig, rec, epsilon = 1e-8);
        }
    }

    #[test]
    fn test_block_omega_not_diagonal() {
        let template = make_block_template();
        assert!(!template.omega.diagonal);
    }

    // ── mu-referencing helpers ──────────────────────────────────────────

    use crate::types::{BloqMethod, CompiledModel, ErrorModel, MuRef, PkModel, PkParams};
    use std::collections::HashMap;

    /// Build a minimal CompiledModel with the given mu-refs. Only fields
    /// that `compute_mu_k` actually reads need to be meaningful; the rest
    /// are filled with defaults.
    fn make_model_with_mu_refs(mu_refs: Vec<(&str, &str, bool)>) -> CompiledModel {
        let theta_names: Vec<String> = vec!["TVCL".into(), "TVV".into(), "TVKA".into()];
        let eta_names: Vec<String> = vec!["ETA_CL".into(), "ETA_V".into(), "ETA_KA".into()];
        let mut refs = HashMap::new();
        for (eta, theta, log_t) in mu_refs {
            refs.insert(
                eta.to_string(),
                MuRef {
                    theta_name: theta.to_string(),
                    log_transformed: log_t,
                },
            );
        }
        let omega = OmegaMatrix::from_diagonal(
            &[0.09, 0.04, 0.30],
            eta_names.clone(),
        );
        let sigma = SigmaVector {
            values: vec![0.02],
            names: vec!["PROP_ERR".into()],
        };
        let default_params = ModelParameters {
            theta: vec![0.2, 10.0, 1.5],
            theta_names: theta_names.clone(),
            theta_lower: vec![0.001, 0.1, 0.01],
            theta_upper: vec![10.0, 500.0, 50.0],
            omega,
            sigma,
        };
        CompiledModel {
            name: "test".into(),
            pk_model: PkModel::OneCptIvBolus,
            error_model: ErrorModel::Proportional,
            pk_param_fn: Box::new(|_, _, _| PkParams::default()),
            n_theta: 3,
            n_eta: 3,
            n_epsilon: 1,
            theta_names,
            eta_names,
            default_params,
            mu_refs: refs,
            tv_fn: None,
            pk_indices: vec![0, 1, 4],
            ode_spec: None,
            bloq_method: BloqMethod::Drop,
            referenced_covariates: Vec::new(),
        }
    }

    #[test]
    fn test_compute_mu_k_no_refs_returns_zeros() {
        // Model with no detected mu-refs → every shift is zero, even when enabled.
        let model = make_model_with_mu_refs(vec![]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_eq!(mu.len(), 3);
        for v in &mu {
            assert_eq!(*v, 0.0);
        }
    }

    #[test]
    fn test_compute_mu_k_disabled_returns_zeros() {
        // `enabled = false` must short-circuit even if mu-refs exist.
        let model = make_model_with_mu_refs(vec![
            ("ETA_CL", "TVCL", true),
            ("ETA_V", "TVV", true),
        ]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], false);
        assert_eq!(mu, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_compute_mu_k_log_transformed() {
        // log-transformed mu-ref (exp / multiplicative pattern) → mu = ln(theta).
        let model = make_model_with_mu_refs(vec![
            ("ETA_CL", "TVCL", true),
            ("ETA_V", "TVV", true),
        ]);
        let theta = vec![0.2_f64, 10.0_f64, 1.5_f64];
        let mu = compute_mu_k(&model, &theta, true);
        assert_relative_eq!(mu[0], 0.2_f64.ln(), epsilon = 1e-12);
        assert_relative_eq!(mu[1], 10.0_f64.ln(), epsilon = 1e-12);
        // ETA_KA has no mu-ref → zero shift.
        assert_eq!(mu[2], 0.0);
    }

    #[test]
    fn test_compute_mu_k_additive_uses_theta_directly() {
        // Additive pattern (THETA + ETA) → mu = theta (no log).
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", false)]);
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_relative_eq!(mu[0], 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_mu_k_clamps_log_of_nonpositive_theta() {
        // ln() of a non-positive theta would be -inf or NaN — the
        // implementation clamps to 1e-10 first. Verify that guard holds.
        let model = make_model_with_mu_refs(vec![("ETA_CL", "TVCL", true)]);
        let mu = compute_mu_k(&model, &[0.0, 10.0, 1.5], true);
        assert!(mu[0].is_finite());
        assert_relative_eq!(mu[0], 1e-10_f64.ln(), epsilon = 1e-6);
    }

    #[test]
    fn test_compute_mu_k_unknown_theta_name_is_ignored() {
        // If the recorded theta_name doesn't exist in theta_names
        // (shouldn't happen in practice, but guard is real), shift stays zero.
        let mut model = make_model_with_mu_refs(vec![]);
        model.mu_refs.insert(
            "ETA_CL".into(),
            MuRef {
                theta_name: "NON_EXISTENT".into(),
                log_transformed: true,
            },
        );
        let mu = compute_mu_k(&model, &[0.2, 10.0, 1.5], true);
        assert_eq!(mu, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_get_eta_init_warm_start_preferred() {
        // Warm start wins over mu_refs.
        let warm = vec![0.5, -0.1, 0.2];
        let mu = vec![1.0, 1.0, 1.0];
        let out = get_eta_init(3, Some(&warm), Some(&mu));
        assert_eq!(out, warm);
    }

    #[test]
    fn test_get_eta_init_falls_back_to_mu_refs() {
        // No warm start → use mu_refs.
        let mu = vec![0.1, 0.2, 0.3];
        let out = get_eta_init(3, None, Some(&mu));
        assert_eq!(out, mu);
    }

    #[test]
    fn test_get_eta_init_falls_back_to_zeros() {
        // Nothing provided → zeros of the requested length.
        let out = get_eta_init(4, None, None);
        assert_eq!(out, vec![0.0; 4]);
    }

    #[test]
    fn test_compute_bounds_block_dimensions() {
        let template = make_block_template();
        let bounds = compute_bounds(&template);
        let expected_len = packed_len(&template);
        assert_eq!(bounds.lower.len(), expected_len);
        assert_eq!(bounds.upper.len(), expected_len);
    }
}
