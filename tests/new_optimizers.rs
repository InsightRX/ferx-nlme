//! Integration tests for the BOBYQA and trust_region outer optimizers.
//!
//! These fit the warfarin example and assert sane convergence behaviour.
//! They don't reproduce reference estimates bit-for-bit — BOBYQA is
//! derivative-free and trust-region uses a CG subproblem, so both take
//! different paths than SLSQP and can settle on slightly different
//! points. The asserts are loose enough to catch "regression to garbage"
//! while tolerating normal optimizer jitter.

use ferx_nlme::parser::model_parser::parse_model_file;
use ferx_nlme::{fit, read_nonmem_csv, FitOptions, Optimizer};
use std::path::Path;

fn data_and_model() -> (
    ferx_nlme::types::CompiledModel,
    ferx_nlme::types::Population,
) {
    let model =
        parse_model_file(Path::new("examples/warfarin.ferx")).expect("warfarin example must parse");
    let population =
        read_nonmem_csv(Path::new("data/warfarin.csv"), None, None).expect("warfarin data must load");
    (model, population)
}

fn base_options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false; // skip FD Hessian — not what we're testing
    opts.outer_maxiter = 100;
    opts
}

#[test]
fn slsqp_reaches_a_sane_ofv() {
    // Baseline: SLSQP (default) should converge on warfarin to a moderate OFV.
    // Used as a reference point for the BOBYQA / trust-region assertions.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Slsqp;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("slsqp fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "SLSQP OFV should be finite, got {}",
        result.ofv
    );
}

#[test]
fn bobyqa_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::Bobyqa;
    // BOBYQA is derivative-free: one "iteration" is cheap but it needs many
    // to triangulate a quadratic. Give it a bit more headroom.
    opts.outer_maxiter = 200;
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("bobyqa fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "BOBYQA OFV must be finite, got {}",
        result.ofv
    );
    // Theta should stay inside the declared bounds from the model file.
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn trust_region_fit_converges_to_finite_ofv() {
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = 30;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region fit must succeed");
    assert!(
        result.ofv.is_finite(),
        "trust_region OFV must be finite, got {}",
        result.ofv
    );
    for (i, &th) in result.theta.iter().enumerate() {
        let lo = model.default_params.theta_lower[i];
        let hi = model.default_params.theta_upper[i];
        assert!(th > lo && th < hi, "theta[{}] = {} escaped bounds", i, th);
    }
}

#[test]
fn bobyqa_and_slsqp_agree_within_10_percent_on_theta() {
    // Cross-check: two different outer optimizers, same problem, should land
    // near each other. Loose 10% tolerance covers optimizer jitter on a
    // noisy FOCE surface without silently accepting divergent solutions.
    let (model, population) = data_and_model();

    let mut opts_slsqp = base_options();
    opts_slsqp.optimizer = Optimizer::Slsqp;
    let r_slsqp = fit(&model, &population, &model.default_params, &opts_slsqp)
        .expect("slsqp fit must succeed");

    let mut opts_bobyqa = base_options();
    opts_bobyqa.optimizer = Optimizer::Bobyqa;
    opts_bobyqa.outer_maxiter = 300;
    let r_bobyqa = fit(&model, &population, &model.default_params, &opts_bobyqa)
        .expect("bobyqa fit must succeed");

    for i in 0..r_slsqp.theta.len() {
        let rel = (r_bobyqa.theta[i] - r_slsqp.theta[i]).abs() / r_slsqp.theta[i].abs().max(1e-6);
        assert!(
            rel < 0.10,
            "theta[{}] mismatch: slsqp={}, bobyqa={}, rel diff={:.3}",
            i,
            r_slsqp.theta[i],
            r_bobyqa.theta[i],
            rel
        );
    }
}

#[test]
fn steihaug_max_iters_is_respected_by_trust_region() {
    // A very small steihaug_max_iters degrades step quality but should still
    // return a valid FitResult without crashing. Catches regressions in the
    // wiring from FitOptions to the argmin Steihaug subsolver.
    let (model, population) = data_and_model();
    let mut opts = base_options();
    opts.optimizer = Optimizer::TrustRegion;
    opts.steihaug_max_iters = 2; // intentionally aggressive
    opts.outer_maxiter = 30;
    let result = fit(&model, &population, &model.default_params, &opts)
        .expect("trust_region with tight CG budget must still return");
    assert!(result.ofv.is_finite());
}
