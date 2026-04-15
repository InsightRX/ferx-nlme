use crate::types::DoseEvent;

/// One-compartment IV bolus: C(t) = (Dose/V) * exp(-k*t)
pub fn one_cpt_iv_bolus(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let k = cl / v;
    (dose.amt / v) * (-k * t).exp()
}

/// One-compartment infusion
/// During infusion (t <= T): C(t) = (Rate/CL) * (1 - exp(-k*t))
/// After infusion (t > T):   C(t) = (Rate/CL) * (1 - exp(-k*T)) * exp(-k*(t-T))
pub fn one_cpt_infusion(dose: &DoseEvent, t: f64, cl: f64, v: f64) -> f64 {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 {
        return 0.0;
    }
    let k = cl / v;
    let rate = dose.rate;
    let dur = dose.duration;

    if dur <= 0.0 {
        // Fallback to bolus
        return one_cpt_iv_bolus(dose, t, cl, v);
    }

    if t <= dur {
        (rate / cl) * (1.0 - (-k * t).exp())
    } else {
        (rate / cl) * (1.0 - (-k * dur).exp()) * (-k * (t - dur)).exp()
    }
}

/// One-compartment oral absorption
/// C(t) = (F*Dose*KA) / (V*(KA - k)) * [exp(-k*t) - exp(-KA*t)]
/// Handles singularity when KA ≈ k via L'Hopital limit
pub fn one_cpt_oral(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64) -> f64 {
    one_cpt_oral_f(dose, t, cl, v, ka, 1.0)
}

pub fn one_cpt_oral_f(dose: &DoseEvent, t: f64, cl: f64, v: f64, ka: f64, f_bio: f64) -> f64 {
    if t < 0.0 || v <= 0.0 || cl <= 0.0 || ka <= 0.0 {
        return 0.0;
    }
    let k = cl / v;
    let d = f_bio * dose.amt;

    if (ka - k).abs() < 1e-6 {
        // L'Hopital limit: C(t) = (D*ka/V) * t * exp(-k*t)
        (d * ka / v) * t * (-k * t).exp()
    } else {
        (d * ka / (v * (ka - k))) * ((-k * t).exp() - (-ka * t).exp())
    }
}

/// Predict concentration from a single dose at elapsed time t using 1-cmt model.
/// Parameters are passed as a HashMap-like slice: [cl, v, ka (optional), f (optional)]
pub fn one_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        one_cpt_infusion(dose, t, cl, v)
    } else if let Some(ka_val) = ka {
        one_cpt_oral_f(dose, t, cl, v, ka_val, f_bio.unwrap_or(1.0))
    } else {
        one_cpt_iv_bolus(dose, t, cl, v)
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

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_iv_bolus(&dose, 0.0, 10.0, 100.0);
        assert_relative_eq!(c, 10.0, epsilon = 1e-10); // Dose/V = 1000/100
    }

    #[test]
    fn test_iv_bolus_decay() {
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v; // 0.1
        let t = 5.0;
        let expected = (1000.0 / v) * (-k * t).exp();
        let c = one_cpt_iv_bolus(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_iv_bolus(&dose, 1000.0, 10.0, 100.0);
        assert!(c < 1e-30);
    }

    #[test]
    fn test_iv_bolus_negative_time() {
        let dose = bolus_dose(1000.0);
        assert_eq!(one_cpt_iv_bolus(&dose, -1.0, 10.0, 100.0), 0.0);
    }

    #[test]
    fn test_iv_bolus_zero_volume() {
        let dose = bolus_dose(1000.0);
        assert_eq!(one_cpt_iv_bolus(&dose, 1.0, 10.0, 0.0), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_during() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10h
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v;
        let t: f64 = 5.0; // during infusion
        let expected = (100.0 / cl) * (1.0 - (-k * t).exp());
        let c = one_cpt_infusion(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_infusion_after() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10h
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v;
        let dur: f64 = 10.0;
        let t: f64 = 15.0; // after infusion
        let expected = (100.0 / cl) * (1.0 - (-k * dur).exp()) * (-k * (t - dur)).exp();
        let c = one_cpt_infusion(&dose, t, cl, v);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(1000.0, 100.0); // duration = 10
        let cl = 10.0;
        let v = 100.0;
        let dur = 10.0;
        let c_at = one_cpt_infusion(&dose, dur, cl, v);
        let c_after = one_cpt_infusion(&dose, dur + 1e-10, cl, v);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-6);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(1000.0);
        let c = one_cpt_oral(&dose, 0.0, 10.0, 100.0, 1.0);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_known_value() {
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let ka: f64 = 1.5;
        let k = cl / v;
        let t: f64 = 2.0;
        let expected = (1000.0 * ka / (v * (ka - k))) * ((-k * t).exp() - (-ka * t).exp());
        let c = one_cpt_oral(&dose, t, cl, v, ka);
        assert_relative_eq!(c, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_singularity_ka_equals_ke() {
        // When ka ≈ k, L'Hopital: C(t) = (D*ka/V) * t * exp(-k*t)
        let dose = bolus_dose(1000.0);
        let cl: f64 = 10.0;
        let v: f64 = 100.0;
        let k = cl / v; // 0.1
        let ka = k; // singularity
        let t: f64 = 5.0;
        let expected = (1000.0 * ka / v) * t * (-k * t).exp();
        let c = one_cpt_oral(&dose, t, cl, v, ka);
        assert_relative_eq!(c, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_oral_with_bioavailability() {
        let dose = bolus_dose(1000.0);
        let c_full = one_cpt_oral_f(&dose, 2.0, 10.0, 100.0, 1.5, 1.0);
        let c_half = one_cpt_oral_f(&dose, 2.0, 10.0, 100.0, 1.5, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(1000.0);
        let direct = one_cpt_iv_bolus(&dose, 2.0, 10.0, 100.0);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(1000.0);
        let direct = one_cpt_oral(&dose, 2.0, 10.0, 100.0, 1.5);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, Some(1.5), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(1000.0, 100.0);
        let direct = one_cpt_infusion(&dose, 2.0, 10.0, 100.0);
        let via_predict = one_cpt_predict(&dose, 2.0, 10.0, 100.0, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }
}
