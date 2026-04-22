use argmin::core::{CostFunction, Error, Executor, Gradient, Hessian, State};
use argmin::solver::trustregion::{Steihaug, TrustRegion};
use nalgebra::{DMatrix, DVector};

use crate::estimation::inner_optimizer::run_inner_loop_warm;
use crate::estimation::outer_optimizer::{compute_covariance, OuterResult};
use crate::estimation::parameterization::{
    clamp_to_bounds, compute_bounds, pack_params, unpack_params, PackedBounds,
};
use crate::stats::likelihood::foce_population_nll;
use crate::types::{CompiledModel, FitOptions, ModelParameters, Population};

struct FoceiProblem<'a> {
    model: &'a CompiledModel,
    population: &'a Population,
    options: &'a FitOptions,
    init_params: &'a ModelParameters,
    bounds: PackedBounds,
    cached_etas: std::sync::Mutex<Vec<DVector<f64>>>,
}

impl FoceiProblem<'_> {
    fn run_inner(&self, x: &[f64]) -> (Vec<DVector<f64>>, Vec<DMatrix<f64>>) {
        let params = unpack_params(x, self.init_params);
        let warm = self.cached_etas.lock().unwrap().clone();
        let warm_ref = if warm.is_empty() { None } else { Some(warm.as_slice()) };
        let (etas, h_mats, _) = run_inner_loop_warm(
            self.model,
            self.population,
            &params,
            self.options.inner_maxiter,
            self.options.inner_tol,
            warm_ref,
        );
        *self.cached_etas.lock().unwrap() = etas.clone();
        (etas, h_mats)
    }

    fn ofv_fixed(&self, x: &[f64], etas: &[DVector<f64>], h_mats: &[DMatrix<f64>]) -> f64 {
        let params = unpack_params(x, self.init_params);
        let nll = foce_population_nll(
            self.model,
            self.population,
            &params.theta,
            etas,
            h_mats,
            &params.omega,
            &params.sigma.values,
            self.options.interaction,
        );
        let raw = 2.0 * nll;
        if raw.is_finite() { raw } else { 1e20 }
    }

    fn grad_fixed(&self, x: &[f64], etas: &[DVector<f64>], h_mats: &[DMatrix<f64>]) -> Vec<f64> {
        let n = x.len();
        let eps = 1e-5_f64;
        let mut g = vec![0.0_f64; n];
        let mut xw = x.to_vec();

        for i in 0..n {
            let h = eps * (1.0 + x[i].abs());
            let xi_plus = (x[i] + h).min(self.bounds.upper[i]);
            let xi_minus = (x[i] - h).max(self.bounds.lower[i]);
            let dh = xi_plus - xi_minus;
            if dh.abs() < 1e-16 {
                continue;
            }
            xw[i] = xi_plus;
            let fp = self.ofv_fixed(&xw, etas, h_mats);
            xw[i] = xi_minus;
            let fm = self.ofv_fixed(&xw, etas, h_mats);
            xw[i] = x[i];
            let gi = (fp - fm) / dh;
            if gi.is_finite() {
                g[i] = gi;
            }
        }
        g
    }
}

// Use Vec<f64> / Vec<Vec<f64>> as the argmin param/gradient/hessian types.
// argmin-math provides trait impls for Vec natively, avoiding nalgebra version conflicts.

impl CostFunction for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Output = f64;

    fn cost(&self, p: &Vec<f64>) -> Result<f64, Error> {
        let (etas, h_mats) = self.run_inner(p);
        Ok(self.ofv_fixed(p, &etas, &h_mats))
    }
}

impl Gradient for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Gradient = Vec<f64>;

    fn gradient(&self, p: &Vec<f64>) -> Result<Vec<f64>, Error> {
        let (etas, h_mats) = self.run_inner(p);
        Ok(self.grad_fixed(p, &etas, &h_mats))
    }
}

impl Hessian for FoceiProblem<'_> {
    type Param = Vec<f64>;
    type Hessian = Vec<Vec<f64>>;

    fn hessian(&self, p: &Vec<f64>) -> Result<Vec<Vec<f64>>, Error> {
        let n = p.len();
        let eps = 1e-4_f64;
        let (etas, h_mats) = self.run_inner(p);
        let g0 = self.grad_fixed(p, &etas, &h_mats);

        let mut h_fwd = vec![vec![0.0_f64; n]; n];
        let mut xp = p.clone();
        for i in 0..n {
            let hi = eps * (1.0 + p[i].abs());
            let xi_plus = (p[i] + hi).min(self.bounds.upper[i]);
            let actual_hi = xi_plus - p[i];
            if actual_hi.abs() < 1e-16 {
                continue;
            }
            xp[i] = xi_plus;
            let g1 = self.grad_fixed(&xp, &etas, &h_mats);
            xp[i] = p[i];
            for j in 0..n {
                h_fwd[i][j] = (g1[j] - g0[j]) / actual_hi;
            }
        }

        // Symmetrize: H_sym[i][j] = (H[i][j] + H[j][i]) / 2
        let mut h_sym = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                h_sym[i][j] = (h_fwd[i][j] + h_fwd[j][i]) / 2.0;
            }
        }
        Ok(h_sym)
    }
}

pub fn optimize_trust_region(
    model: &CompiledModel,
    population: &Population,
    init_params: &ModelParameters,
    options: &FitOptions,
) -> OuterResult {
    let bounds = compute_bounds(init_params);
    let mut x0 = pack_params(init_params);
    clamp_to_bounds(&mut x0, &bounds);

    let mut warnings = Vec::new();

    let n_subj = population.subjects.len();
    let n_eta = model.n_eta;

    let problem = FoceiProblem {
        model,
        population,
        options,
        init_params,
        bounds,
        cached_etas: std::sync::Mutex::new(vec![DVector::zeros(n_eta); n_subj]),
    };

    if options.verbose {
        eprintln!(
            "Starting trust-region optimization ({} parameters)...",
            x0.len()
        );
    }

    let subproblem = Steihaug::new().with_max_iters(options.steihaug_max_iters as u64);
    let solver = TrustRegion::new(subproblem)
        .with_radius(1.0)
        .expect("trust region radius must be positive")
        .with_max_radius(10.0)
        .expect("trust region max radius must be positive");

    let result = Executor::new(problem, solver)
        .configure(|state| {
            state
                .param(x0.clone())
                .max_iters(options.outer_maxiter as u64)
        })
        .run();

    let (converged, mut best_x) = match result {
        Ok(res) => {
            if options.verbose {
                eprintln!("Trust-region finished: {} iters", res.state().get_iter());
            }
            let vec = res
                .state()
                .get_best_param()
                .cloned()
                .unwrap_or_else(|| x0.clone());
            (true, vec)
        }
        Err(e) => {
            if options.verbose {
                eprintln!("Trust-region stopped: {}", e);
            }
            warnings.push(format!("Trust-region did not converge: {}", e));
            (false, x0.clone())
        }
    };

    clamp_to_bounds(&mut best_x, &compute_bounds(init_params));

    let final_params = unpack_params(&best_x, init_params);
    let (final_ehs, final_hms, _) = run_inner_loop_warm(
        model,
        population,
        &final_params,
        options.inner_maxiter,
        options.inner_tol,
        None,
    );

    let final_ofv = {
        let nll = foce_population_nll(
            model,
            population,
            &final_params.theta,
            &final_ehs,
            &final_hms,
            &final_params.omega,
            &final_params.sigma.values,
            options.interaction,
        );
        2.0 * nll
    };

    if options.verbose {
        eprintln!("Final OFV = {:.6}", final_ofv);
    }

    let covariance_matrix = if options.run_covariance_step {
        if options.verbose {
            eprintln!("Computing covariance matrix...");
        }
        compute_covariance(
            &best_x,
            init_params,
            model,
            population,
            &final_ehs,
            &final_hms,
            options,
        )
    } else {
        None
    };

    if covariance_matrix.is_none() && options.run_covariance_step {
        warnings.push("Covariance step failed".to_string());
    }

    OuterResult {
        params: final_params,
        ofv: final_ofv,
        converged,
        n_iterations: 0,
        eta_hats: final_ehs,
        h_matrices: final_hms,
        covariance_matrix,
        warnings,
    }
}
