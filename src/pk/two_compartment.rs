use crate::types::DoseEvent;

/// Compute macro-rate constants (alpha, beta) from micro-constants.
/// Uses Vieta's formula for beta to avoid catastrophic cancellation
/// when s >> sqrt(s^2 - 4d).
fn macro_rates(cl: f64, v1: f64, q: f64, v2: f64) -> (f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q / v1;
    let k21 = q / v2;
    let s = k10 + k12 + k21;
    let d = k10 * k21;
    let disc = { let x = s * s - 4.0 * d; if x > 0.0 { x.sqrt() } else { 0.0 } };
    let alpha = (s + disc) / 2.0;
    // Vieta's formula: alpha * beta = d, so beta = d / alpha
    // This avoids subtracting two nearly-equal large numbers.
    let beta = if alpha > 1e-30 { d / alpha } else { 0.0 };
    (alpha, beta, k21)
}

/// Two-compartment IV bolus
/// C(t) = A*exp(-alpha*t) + B*exp(-beta*t)
pub fn two_cpt_iv_bolus(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }

    let a = (dose.amt / v1) * (alpha - k21) / diff;
    let b = (dose.amt / v1) * (k21 - beta) / diff;

    a * (-alpha * t).exp() + b * (-beta * t).exp()
}

/// Two-compartment infusion
pub fn two_cpt_infusion(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 || alpha.abs() < 1e-12 || beta.abs() < 1e-12 {
        return 0.0;
    }

    let rate = dose.rate;
    let dur = dose.duration;
    if dur <= 0.0 {
        return two_cpt_iv_bolus(dose, t, cl, v1, q, v2);
    }

    let a_coeff = (rate / v1) * (alpha - k21) / (diff * alpha);
    let b_coeff = (rate / v1) * (k21 - beta) / (diff * beta);

    if t <= dur {
        a_coeff * (1.0 - (-alpha * t).exp()) + b_coeff * (1.0 - (-beta * t).exp())
    } else {
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
    }
}

/// Two-compartment oral absorption
/// C(t) = P*exp(-alpha*t) + Q*exp(-beta*t) + R*exp(-ka*t)
pub fn two_cpt_oral(dose: &DoseEvent, t: f64, cl: f64, v1: f64, q: f64, v2: f64, ka: f64) -> f64 {
    two_cpt_oral_f(dose, t, cl, v1, q, v2, ka, 1.0)
}

pub fn two_cpt_oral_f(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if t < 0.0 || v1 <= 0.0 || cl <= 0.0 || ka <= 0.0 {
        return 0.0;
    }
    let (alpha, beta, k21) = macro_rates(cl, v1, q, v2);
    let diff = alpha - beta;
    if diff.abs() < 1e-12 {
        return 0.0;
    }

    let d = f_bio * dose.amt * ka / v1;

    // Standard formula:
    //   C(t) = d * [ (k21-α)/((ka-α)(β-α)) · e^{-αt}
    //              + (k21-β)/((ka-β)(α-β)) · e^{-βt}
    //              + (k21-ka)/((α-ka)(β-ka)) · e^{-ka·t} ]
    //
    // Handle singularities when ka ≈ alpha or ka ≈ beta via L'Hopital limits.
    let p = if (ka - alpha).abs() < 1e-6 {
        d * (alpha - k21) / diff * t * (-alpha * t).exp()
    } else {
        d * (k21 - alpha) / ((ka - alpha) * (beta - alpha)) * (-alpha * t).exp()
    };

    let q_val = if (ka - beta).abs() < 1e-6 {
        d * (k21 - beta) / diff * t * (-beta * t).exp()
    } else {
        d * (k21 - beta) / ((ka - beta) * (alpha - beta)) * (-beta * t).exp()
    };

    let r = if (ka - alpha).abs() < 1e-6 || (ka - beta).abs() < 1e-6 {
        0.0
    } else {
        d * (k21 - ka) / ((alpha - ka) * (beta - ka)) * (-ka * t).exp()
    };

    p + q_val + r
}

/// Predict concentration from a single dose at elapsed time t using 2-cmt model.
pub fn two_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q: f64,
    v2: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        two_cpt_infusion(dose, t, cl, v1, q, v2)
    } else if let Some(ka_val) = ka {
        two_cpt_oral_f(dose, t, cl, v1, q, v2, ka_val, f_bio.unwrap_or(1.0))
    } else {
        two_cpt_iv_bolus(dose, t, cl, v1, q, v2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn bolus_dose(amt: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, 0.0, false, 0.0)
    }

    fn infusion_dose(amt: f64, rate: f64) -> DoseEvent {
        DoseEvent::new(0.0, amt, 1, rate, false, 0.0)
    }

    // Typical 2-cpt PK parameters
    const CL: f64 = 10.0;
    const V1: f64 = 100.0;
    const Q: f64 = 5.0;
    const V2: f64 = 200.0;

    // --- Macro rates ---

    #[test]
    fn test_macro_rates_positive() {
        let (alpha, beta, k21) = macro_rates(CL, V1, Q, V2);
        assert!(alpha > beta);
        assert!(alpha > 0.0);
        assert!(beta > 0.0);
        assert!(k21 > 0.0);
    }

    #[test]
    fn test_macro_rates_vieta() {
        // alpha * beta = k10 * k21 (Vieta's formula)
        let k10 = CL / V1;
        let k21 = Q / V2;
        let (alpha, beta, _) = macro_rates(CL, V1, Q, V2);
        assert_relative_eq!(alpha * beta, k10 * k21, epsilon = 1e-10);
    }

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_iv_bolus(&dose, 0.0, CL, V1, Q, V2);
        assert_relative_eq!(c, 1000.0 / V1, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_iv_bolus(&dose, 10000.0, CL, V1, Q, V2);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_iv_bolus_monotone_decrease_eventually() {
        // After distribution phase, concentrations should decrease
        let dose = bolus_dose(1000.0);
        let c1 = two_cpt_iv_bolus(&dose, 50.0, CL, V1, Q, V2);
        let c2 = two_cpt_iv_bolus(&dose, 100.0, CL, V1, Q, V2);
        assert!(c2 < c1);
    }

    #[test]
    fn test_iv_bolus_guard_clauses() {
        let dose = bolus_dose(1000.0);
        assert_eq!(two_cpt_iv_bolus(&dose, -1.0, CL, V1, Q, V2), 0.0);
        assert_eq!(two_cpt_iv_bolus(&dose, 1.0, CL, 0.0, Q, V2), 0.0);
        assert_eq!(two_cpt_iv_bolus(&dose, 1.0, 0.0, V1, Q, V2), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_during() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let c = two_cpt_infusion(&dose, 5.0, CL, V1, Q, V2);
        assert!(c > 0.0);
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let dur = 10.0;
        let c_at = two_cpt_infusion(&dose, dur, CL, V1, Q, V2);
        let c_after = two_cpt_infusion(&dose, dur + 1e-10, CL, V1, Q, V2);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_infusion_after_decays() {
        let dose = infusion_dose(1000.0, 100.0); // dur=10
        let c1 = two_cpt_infusion(&dose, 50.0, CL, V1, Q, V2);
        let c2 = two_cpt_infusion(&dose, 100.0, CL, V1, Q, V2);
        assert!(c2 < c1);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 0.0, CL, V1, Q, V2, 1.5);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_positive_at_peak() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 2.0, CL, V1, Q, V2, 1.5);
        assert!(c > 0.0);
    }

    #[test]
    fn test_oral_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = two_cpt_oral(&dose, 10000.0, CL, V1, Q, V2, 1.5);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_oral_bioavailability_scaling() {
        let dose = bolus_dose(1000.0);
        let c_full = two_cpt_oral_f(&dose, 2.0, CL, V1, Q, V2, 1.5, 1.0);
        let c_half = two_cpt_oral_f(&dose, 2.0, CL, V1, Q, V2, 1.5, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(1000.0);
        let direct = two_cpt_iv_bolus(&dose, 2.0, CL, V1, Q, V2);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(1000.0);
        let direct = two_cpt_oral(&dose, 2.0, CL, V1, Q, V2, 1.5);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, Some(1.5), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(1000.0, 100.0);
        let direct = two_cpt_infusion(&dose, 2.0, CL, V1, Q, V2);
        let via_predict = two_cpt_predict(&dose, 2.0, CL, V1, Q, V2, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }
}
