//! Sampling Importance Resampling (SIR) for parameter uncertainty estimation.
//!
//! Implements the SIR procedure described in Dosne et al. (2017):
//! "Improving the estimation of parameter uncertainty distributions in
//! nonlinear mixed effects models using sampling importance resampling"
//!
//! SIR provides a non-parametric estimate of parameter uncertainty that is
//! more robust than the asymptotic covariance matrix.

use crate::estimation::inner_optimizer::run_inner_loop;
use crate::estimation::parameterization::{pack_params, unpack_params};
use crate::stats::likelihood::foce_population_nll;
use crate::types::*;
use nalgebra::{DMatrix, DVector};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::StandardNormal;

/// Results from the SIR procedure.
#[derive(Debug, Clone)]
pub struct SirResult {
    /// 95% CI (2.5th, 97.5th percentile) for each theta on original scale
    pub ci_theta: Vec<(f64, f64)>,
    /// 95% CI for each omega diagonal element
    pub ci_omega: Vec<(f64, f64)>,
    /// 95% CI for each sigma
    pub ci_sigma: Vec<(f64, f64)>,
    /// Effective sample size (ESS = 1 / sum(w_k^2))
    pub effective_sample_size: f64,
}

/// Run the SIR procedure after maximum likelihood estimation.
///
/// # Arguments
/// * `model` - The compiled model
/// * `population` - The dataset
/// * `params` - ML parameter estimates
/// * `eta_hats` - ML EBE estimates (for warm-starting inner loop)
/// * `proposal_cov` - Covariance matrix in packed (log-transformed) parameter space
/// * `ofv_hat` - OFV at ML estimates
/// * `options` - Fit options containing SIR settings
pub fn run_sir(
    model: &CompiledModel,
    population: &Population,
    params: &ModelParameters,
    eta_hats: &[DVector<f64>],
    proposal_cov: &DMatrix<f64>,
    ofv_hat: f64,
    options: &FitOptions,
) -> Result<SirResult, String> {
    let n_samples = options.sir_samples;
    let n_resamples = options.sir_resamples;

    if n_resamples > n_samples {
        return Err("sir_resamples must be <= sir_samples".to_string());
    }

    // Pack ML estimates as the proposal center
    let x_hat = pack_params(params);
    let n_packed = x_hat.len();

    if proposal_cov.nrows() != n_packed || proposal_cov.ncols() != n_packed {
        return Err(format!(
            "Covariance matrix dimensions ({},{}) don't match packed parameters ({})",
            proposal_cov.nrows(),
            proposal_cov.ncols(),
            n_packed,
        ));
    }

    // Cholesky decomposition of proposal covariance for sampling
    let proposal_chol = proposal_cov
        .clone()
        .cholesky()
        .ok_or("Proposal covariance is not positive definite")?
        .l();

    // Log-determinant of proposal covariance (for density computation)
    let log_det_proposal = 2.0
        * (0..n_packed)
            .map(|i| proposal_chol[(i, i)].ln())
            .sum::<f64>();

    let mut rng = match options.sir_seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::seed_from_u64(12345),
    };

    if options.verbose {
        eprintln!(
            "  SIR: drawing {} samples, resampling {}...",
            n_samples, n_resamples
        );
    }

    // Step 1: Sample from proposal MVN and compute importance weights
    let mut log_weights = Vec::with_capacity(n_samples);
    let mut samples = Vec::with_capacity(n_samples);

    // Log-proposal density at the ML estimate (it's the center, so quadratic form = 0)
    let log_q_hat = -0.5 * (n_packed as f64 * (2.0 * std::f64::consts::PI).ln() + log_det_proposal);

    for k in 0..n_samples {
        // Sample z ~ N(0, I), then x_k = x_hat + L * z
        let z: Vec<f64> = (0..n_packed).map(|_| rng.sample(StandardNormal)).collect();
        let z_vec = DVector::from_column_slice(&z);
        let delta = &proposal_chol * &z_vec;
        let x_k: Vec<f64> = x_hat.iter().zip(delta.iter()).map(|(a, b)| a + b).collect();

        // Unpack to model parameters
        let params_k = unpack_params(&x_k, params);

        // Check for invalid parameters (negative volumes, etc.)
        let any_invalid = params_k.theta.iter().any(|&t| !t.is_finite() || t <= 0.0)
            || params_k
                .sigma
                .values
                .iter()
                .any(|&s| !s.is_finite() || s <= 0.0);

        if any_invalid {
            log_weights.push(f64::NEG_INFINITY);
            samples.push(x_k);
            continue;
        }

        // Run inner loop to find EBEs for this parameter vector
        let (ehs, hms, _) = run_inner_loop(
            model,
            population,
            &params_k,
            options.inner_maxiter,
            options.inner_tol,
        );

        // Compute OFV at sampled parameters
        let nll_k = foce_population_nll(
            model,
            population,
            &params_k.theta,
            &ehs,
            &hms,
            &params_k.omega,
            &params_k.sigma.values,
            options.interaction,
        );
        let ofv_k = 2.0 * nll_k;

        if !ofv_k.is_finite() {
            log_weights.push(f64::NEG_INFINITY);
            samples.push(x_k);
            continue;
        }

        let dofv = ofv_k - ofv_hat;

        // Log-proposal density at x_k: log q(x_k) = -0.5 * (n*log(2pi) + log|Σ| + (x_k - x_hat)' Σ^-1 (x_k - x_hat))
        // The quadratic form is ||z||^2 since x_k = x_hat + L*z
        let quad_form: f64 = z.iter().map(|zi| zi * zi).sum();
        let log_q_k = -0.5
            * (n_packed as f64 * (2.0 * std::f64::consts::PI).ln() + log_det_proposal + quad_form);

        // Importance weight: log w_k = log target - log proposal
        //   log target ∝ -0.5 * OFV_k
        //   log w_k = -0.5 * OFV_k - log_q_k
        // We use dOFV relative to OFV_hat for numerical stability:
        //   log w_k = -0.5 * dOFV_k - log_q_k + log_q_hat
        let log_w = -0.5 * dofv - log_q_k + log_q_hat;

        log_weights.push(log_w);
        samples.push(x_k);

        if options.verbose && (k + 1) % 100 == 0 {
            eprintln!("  SIR: {}/{} samples evaluated", k + 1, n_samples);
        }
    }

    // Step 2: Normalize weights using log-sum-exp trick
    let max_log_w = log_weights
        .iter()
        .cloned()
        .filter(|w| w.is_finite())
        .fold(f64::NEG_INFINITY, f64::max);

    if max_log_w == f64::NEG_INFINITY {
        return Err("All SIR samples had invalid weights".to_string());
    }

    let weights: Vec<f64> = log_weights
        .iter()
        .map(|lw| (lw - max_log_w).exp())
        .collect();
    let sum_w: f64 = weights.iter().sum();
    let normalized_weights: Vec<f64> = weights.iter().map(|w| w / sum_w).collect();

    // Effective sample size
    let sum_w2: f64 = normalized_weights.iter().map(|w| w * w).sum();
    let ess = if sum_w2 > 0.0 { 1.0 / sum_w2 } else { 0.0 };

    if options.verbose {
        eprintln!("  SIR: effective sample size = {:.1}", ess);
    }

    // Step 3: Resample with replacement proportional to weights
    let indices: Vec<usize> = (0..n_samples).collect();
    let resampled_indices: Vec<usize> = (0..n_resamples)
        .map(|_| {
            *indices
                .choose_weighted(&mut rng, |&i| normalized_weights[i])
                .unwrap_or(&0)
        })
        .collect();

    // Step 4: Unpack resampled parameter vectors and compute CIs
    let n_theta = params.theta.len();
    let n_eta = params.omega.dim();
    let n_sigma = params.sigma.values.len();

    let mut theta_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_theta];
    let mut omega_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_eta];
    let mut sigma_samples: Vec<Vec<f64>> = vec![Vec::with_capacity(n_resamples); n_sigma];

    for &idx in &resampled_indices {
        let p = unpack_params(&samples[idx], params);
        for (j, &th) in p.theta.iter().enumerate() {
            theta_samples[j].push(th);
        }
        for j in 0..n_eta {
            omega_samples[j].push(p.omega.matrix[(j, j)]);
        }
        for (j, &s) in p.sigma.values.iter().enumerate() {
            sigma_samples[j].push(s);
        }
    }

    let ci_theta: Vec<(f64, f64)> = theta_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_omega: Vec<(f64, f64)> = omega_samples.iter().map(|s| percentile_ci(s)).collect();
    let ci_sigma: Vec<(f64, f64)> = sigma_samples.iter().map(|s| percentile_ci(s)).collect();

    Ok(SirResult {
        ci_theta,
        ci_omega,
        ci_sigma,
        effective_sample_size: ess,
    })
}

/// Compute 2.5th and 97.5th percentiles from a sample.
fn percentile_ci(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let lo_idx = ((n as f64) * 0.025).floor() as usize;
    let hi_idx = ((n as f64) * 0.975).ceil() as usize;
    let lo = sorted[lo_idx.min(n - 1)];
    let hi = sorted[hi_idx.min(n - 1)];
    (lo, hi)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_percentile_ci_sorted() {
        let values: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let (lo, hi) = percentile_ci(&values);
        assert!(lo >= 0.02 && lo <= 0.03, "lo={}", lo);
        assert!(hi >= 0.97 && hi <= 0.98, "hi={}", hi);
    }

    #[test]
    fn test_percentile_ci_single() {
        let (lo, hi) = percentile_ci(&[5.0]);
        assert_eq!(lo, 5.0);
        assert_eq!(hi, 5.0);
    }

    #[test]
    fn test_percentile_ci_empty() {
        let (lo, hi) = percentile_ci(&[]);
        assert!(lo.is_nan());
        assert!(hi.is_nan());
    }
}
