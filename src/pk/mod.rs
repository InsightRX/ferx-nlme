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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::collections::HashMap;

    fn bolus_dose(time: f64, amt: f64) -> DoseEvent {
        DoseEvent::new(time, amt, 1, 0.0, false, 0.0)
    }

    fn make_pk_params(cl: f64, v: f64) -> PkParams {
        let mut p = PkParams::default();
        p.values[0] = cl;
        p.values[1] = v;
        p
    }

    #[test]
    fn test_superposition_single_dose() {
        let doses = vec![bolus_dose(0.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let c = predict_concentration(PkModel::OneCptIvBolus, &doses, 0.0, &pk);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_two_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(10.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);
        let k: f64 = 10.0 / 100.0;

        // At t=10, first dose has decayed, second dose just given
        let c = predict_concentration(PkModel::OneCptIvBolus, &doses, 10.0, &pk);
        let expected = (1000.0_f64 / 100.0) * (-k * 10.0).exp() + 1000.0 / 100.0;
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_superposition_ignores_future_doses() {
        let doses = vec![bolus_dose(0.0, 1000.0), bolus_dose(100.0, 1000.0)];
        let pk = make_pk_params(10.0, 100.0);

        // At t=5, second dose hasn't happened yet
        let c_single = predict_concentration(PkModel::OneCptIvBolus, &[bolus_dose(0.0, 1000.0)], 5.0, &pk);
        let c_two = predict_concentration(PkModel::OneCptIvBolus, &doses, 5.0, &pk);
        assert_relative_eq!(c_single, c_two, epsilon = 1e-12);
    }

    #[test]
    fn test_compute_predictions_length() {
        let subject = Subject {
            id: "1".to_string(),
            doses: vec![bolus_dose(0.0, 1000.0)],
            obs_times: vec![1.0, 2.0, 4.0, 8.0],
            observations: vec![0.0; 4],
            obs_cmts: vec![1; 4],
            covariates: HashMap::new(),
            tvcov: HashMap::new(),
        };
        let pk = make_pk_params(10.0, 100.0);
        let preds = compute_predictions(PkModel::OneCptIvBolus, &subject, &pk);
        assert_eq!(preds.len(), 4);
        // Predictions should be monotonically decreasing for IV bolus
        for i in 1..preds.len() {
            assert!(preds[i] < preds[i - 1]);
        }
    }
}
