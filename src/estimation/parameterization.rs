use crate::types::{ModelParameters, OmegaMatrix, SigmaVector};
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
            upper.push(4.0);  // exp(4) ≈ 55
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
        upper.push(5.0);  // exp(5) ≈ 148
    }

    PackedBounds { lower, upper }
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
        let omega = OmegaMatrix::from_diagonal(&[0.09, 0.04], vec!["eta_cl".into(), "eta_v".into()]);
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
        for (orig, rec) in template.sigma.values.iter().zip(recovered.sigma.values.iter()) {
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
}
