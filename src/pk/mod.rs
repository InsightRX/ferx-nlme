pub mod one_compartment;
pub mod two_compartment;

use crate::types::{DoseEvent, PkModel, PkParams, Subject};

pub use one_compartment::*;
pub use two_compartment::*;

/// Predict concentration at a given time for a subject, summing contributions
/// from all prior doses (superposition principle).
pub fn predict_concentration(
    pk_model: PkModel,
    doses: &[DoseEvent],
    t: f64,
    pk_params: &PkParams,
) -> f64 {
    let mut conc = 0.0;
    for dose in doses {
        if dose.time <= t {
            let tau = t - dose.time;
            conc += single_dose_concentration(pk_model, dose, tau, pk_params);
        }
    }
    conc.max(0.0)
}

/// Concentration contribution from a single dose at elapsed time tau
fn single_dose_concentration(
    pk_model: PkModel,
    dose: &DoseEvent,
    tau: f64,
    p: &PkParams,
) -> f64 {
    let cl = p.cl();
    let v = p.v();

    match pk_model {
        PkModel::OneCptIvBolus => one_cpt_iv_bolus(dose, tau, cl, v),
        PkModel::OneCptInfusion => one_cpt_infusion(dose, tau, cl, v),
        PkModel::OneCptOral => {
            one_cpt_oral_f(dose, tau, cl, v, p.ka(), p.f_bio())
        }
        PkModel::TwoCptIvBolus => {
            two_cpt_iv_bolus(dose, tau, cl, v, p.q(), p.v2())
        }
        PkModel::TwoCptInfusion => {
            two_cpt_infusion(dose, tau, cl, v, p.q(), p.v2())
        }
        PkModel::TwoCptOral => {
            two_cpt_oral_f(dose, tau, cl, v, p.q(), p.v2(), p.ka(), p.f_bio())
        }
    }
}

/// Compute predictions for all observation times of a subject.
/// Uses analytical equations for standard PK models, or delegates to ODE solver
/// when an OdeSpec is provided.
pub fn compute_predictions(
    pk_model: PkModel,
    subject: &Subject,
    pk_params: &PkParams,
) -> Vec<f64> {
    subject
        .obs_times
        .iter()
        .map(|&t| predict_concentration(pk_model, &subject.doses, t, pk_params))
        .collect()
}

/// Compute predictions using ODE integration.
/// `pk_params_flat` is the flat parameter vector passed to the ODE RHS function.
pub fn compute_predictions_ode(
    ode_spec: &crate::ode::OdeSpec,
    subject: &Subject,
    pk_params_flat: &[f64],
) -> Vec<f64> {
    crate::ode::ode_predictions(ode_spec, pk_params_flat, subject)
}
