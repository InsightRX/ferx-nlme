//! Output-shape regression tests for the per-subject EBE / individual-
//! parameter changes.
//!
//! Two contracts pinned here:
//! 1. `CompiledModel::indiv_param_names` mirrors the top-level
//!    `[individual_parameters]` declarations in source order. The downstream
//!    R FFI uses this list to label `fit$individual_estimates` columns.
//! 2. `output::sdtab()` no longer emits `ETA1..ETAn` columns. Per-subject
//!    EBEs live in `fit$ebe_etas` on the R side; sdtab is now strictly
//!    per-observation diagnostic data.

use ferx_nlme::io::output::sdtab;
use ferx_nlme::parser::model_parser::parse_model_file;
use ferx_nlme::{fit, read_nonmem_csv, FitOptions, Optimizer};
use std::path::Path;

fn warfarin_setup() -> (
    ferx_nlme::types::CompiledModel,
    ferx_nlme::types::Population,
) {
    let model = parse_model_file(Path::new("examples/warfarin.ferx"))
        .expect("warfarin example must parse");
    let population = read_nonmem_csv(Path::new("data/warfarin.csv"), None, None)
        .expect("warfarin data must load");
    (model, population)
}

fn fast_options() -> FitOptions {
    let mut opts = FitOptions::default();
    opts.verbose = false;
    opts.run_covariance_step = false;
    opts.optimizer = Optimizer::Slsqp;
    opts.outer_maxiter = 30; // enough to populate diagnostics; not chasing convergence
    opts
}

#[test]
fn indiv_param_names_mirrors_warfarin_individual_parameters_block() {
    // Warfarin's [individual_parameters] block declares CL, V, KA in that
    // order. The compiled model must surface the same names so the R FFI
    // can label individual_estimates columns correctly.
    let (model, _) = warfarin_setup();
    assert_eq!(
        model.indiv_param_names,
        vec!["CL".to_string(), "V".to_string(), "KA".to_string()],
        "indiv_param_names must match the [individual_parameters] block in source order"
    );
    // Parallel to pk_indices (used by the FFI to read each value out of
    // the PkParams slot for analytical models).
    assert_eq!(
        model.indiv_param_names.len(),
        model.pk_indices.len(),
        "indiv_param_names and pk_indices must be aligned"
    );
}

#[test]
fn sdtab_omits_eta_columns_after_fit() {
    // Regression: sdtab used to carry ETA1..ETAn columns. They have moved
    // to `ebe_etas` on the R side, so sdtab must no longer surface any
    // column whose name starts with "ETA".
    let (model, population) = warfarin_setup();
    let opts = fast_options();
    let result =
        fit(&model, &population, &model.default_params, &opts).expect("warfarin fit must succeed");

    let cols = sdtab(&result, &population);
    let names: Vec<&str> = cols.iter().map(|(n, _)| n.as_str()).collect();
    let eta_cols: Vec<&&str> = names.iter().filter(|n| n.starts_with("ETA")).collect();
    assert!(
        eta_cols.is_empty(),
        "sdtab must not contain ETA columns; found: {:?}",
        eta_cols
    );

    // Spot-check the per-observation columns we still expect, so a future
    // accidental column drop also fails this test.
    for required in [
        "ID", "TIME", "DV", "PRED", "IPRED", "CWRES", "IWRES", "EBE_OFV", "N_OBS",
    ] {
        assert!(
            names.contains(&required),
            "sdtab missing required column `{}`; have: {:?}",
            required,
            names
        );
    }
}
