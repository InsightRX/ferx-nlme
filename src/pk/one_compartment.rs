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
