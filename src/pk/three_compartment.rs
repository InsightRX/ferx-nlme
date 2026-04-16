use crate::types::DoseEvent;

/// Compute macro-rate constants (alpha, beta, gamma) from micro-constants
/// for a three-compartment model using the trigonometric (Vieta) method.
///
/// Returns (alpha, beta, gamma, k21, k31) where alpha > beta > gamma > 0.
fn macro_rates_three_cpt(
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> (f64, f64, f64, f64, f64) {
    let k10 = cl / v1;
    let k12 = q2 / v1;
    let k21 = q2 / v2;
    let k13 = q3 / v1;
    let k31 = q3 / v3;

    // Symmetric functions of the roots (Vieta's formulas)
    let s2 = k10 + k12 + k13 + k21 + k31;
    let s1 = k10 * k21 + k10 * k31 + k21 * k31 + k12 * k31 + k13 * k21;
    let s0 = k10 * k21 * k31;

    // Depress the cubic: lambda = x + s2/3
    let h = s2 / 3.0;
    let p = s1 - s2 * s2 / 3.0;
    let q = s1 * s2 / 3.0 - 2.0 * s2 * s2 * s2 / 27.0 - s0;

    // Trigonometric solution (three distinct real roots guaranteed for valid PK params)
    let p_safe = p.min(-1e-30); // guard against p=0
    let m = 2.0 * (-p_safe / 3.0).sqrt();
    let arg = (3.0 * q / (p_safe * m)).clamp(-1.0, 1.0);
    let phi = arg.acos() / 3.0;

    let pi_2_3 = 2.0 * std::f64::consts::FRAC_PI_3;
    let lambda0 = m * phi.cos() + h;
    let lambda1 = m * (phi - pi_2_3).cos() + h;
    let lambda2 = m * (phi - 2.0 * pi_2_3).cos() + h;

    // Sort: alpha > beta > gamma
    let alpha = lambda0.max(lambda1).max(lambda2);
    let gamma = lambda0.min(lambda1).min(lambda2);
    let beta = s2 - alpha - gamma;

    (alpha, beta, gamma, k21, k31)
}

/// Three-compartment IV bolus
/// C(t) = A*exp(-alpha*t) + B*exp(-beta*t) + G*exp(-gamma*t)
pub fn three_cpt_iv_bolus(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }

    let d = dose.amt / v1;
    let a = d * (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = d * (beta - k21) * (beta - k31) / (-ab * bg);
    let g = d * (gamma - k21) * (gamma - k31) / (ag * bg);

    a * (-alpha * t).exp() + b * (-beta * t).exp() + g * (-gamma * t).exp()
}

/// Three-compartment constant-rate IV infusion
pub fn three_cpt_infusion(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
) -> f64 {
    if t < 0.0 || v1 <= 0.0 || v2 <= 0.0 || v3 <= 0.0 || cl <= 0.0 || q2 < 0.0 || q3 < 0.0 {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12
        || ag.abs() < 1e-12
        || bg.abs() < 1e-12
        || alpha.abs() < 1e-12
        || beta.abs() < 1e-12
        || gamma.abs() < 1e-12
    {
        return 0.0;
    }

    let rate = dose.rate;
    let dur = dose.duration;
    if dur <= 0.0 {
        return three_cpt_iv_bolus(dose, t, cl, v1, q2, v2, q3, v3);
    }

    let rv = rate / v1;
    let a_coeff = rv * (alpha - k21) * (alpha - k31) / (ab * ag * alpha);
    let b_coeff = rv * (beta - k21) * (beta - k31) / (-ab * bg * beta);
    let g_coeff = rv * (gamma - k21) * (gamma - k31) / (ag * bg * gamma);

    if t <= dur {
        a_coeff * (1.0 - (-alpha * t).exp())
            + b_coeff * (1.0 - (-beta * t).exp())
            + g_coeff * (1.0 - (-gamma * t).exp())
    } else {
        let dt = t - dur;
        a_coeff * (1.0 - (-alpha * dur).exp()) * (-alpha * dt).exp()
            + b_coeff * (1.0 - (-beta * dur).exp()) * (-beta * dt).exp()
            + g_coeff * (1.0 - (-gamma * dur).exp()) * (-gamma * dt).exp()
    }
}

/// Three-compartment oral absorption with bioavailability
pub fn three_cpt_oral_f(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
    f_bio: f64,
) -> f64 {
    if t < 0.0
        || v1 <= 0.0
        || v2 <= 0.0
        || v3 <= 0.0
        || cl <= 0.0
        || q2 < 0.0
        || q3 < 0.0
        || ka <= 0.0
    {
        return 0.0;
    }
    let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(cl, v1, q2, v2, q3, v3);
    let ab = alpha - beta;
    let ag = alpha - gamma;
    let bg = beta - gamma;
    if ab.abs() < 1e-12 || ag.abs() < 1e-12 || bg.abs() < 1e-12 {
        return 0.0;
    }

    let coeff = f_bio * dose.amt * ka / v1;

    // Normalized residue coefficients
    let a = (alpha - k21) * (alpha - k31) / (ab * ag);
    let b = (beta - k21) * (beta - k31) / (-ab * bg);
    let c = (gamma - k21) * (gamma - k31) / (ag * bg);

    // Bateman function with L'Hôpital singularity handling
    let bateman = |lambda: f64| -> f64 {
        if (ka - lambda).abs() < 1e-6 {
            t * (-lambda * t).exp()
        } else {
            ((-lambda * t).exp() - (-ka * t).exp()) / (ka - lambda)
        }
    };

    coeff * (a * bateman(alpha) + b * bateman(beta) + c * bateman(gamma))
}

/// Three-compartment oral absorption (F=1.0 default)
pub fn three_cpt_oral(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: f64,
) -> f64 {
    three_cpt_oral_f(dose, t, cl, v1, q2, v2, q3, v3, ka, 1.0)
}

/// Predict concentration from a single dose at elapsed time t using 3-cmt model.
pub fn three_cpt_predict(
    dose: &DoseEvent,
    t: f64,
    cl: f64,
    v1: f64,
    q2: f64,
    v2: f64,
    q3: f64,
    v3: f64,
    ka: Option<f64>,
    f_bio: Option<f64>,
) -> f64 {
    if dose.is_infusion() {
        three_cpt_infusion(dose, t, cl, v1, q2, v2, q3, v3)
    } else if let Some(ka_val) = ka {
        three_cpt_oral_f(
            dose,
            t,
            cl,
            v1,
            q2,
            v2,
            q3,
            v3,
            ka_val,
            f_bio.unwrap_or(1.0),
        )
    } else {
        three_cpt_iv_bolus(dose, t, cl, v1, q2, v2, q3, v3)
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

    // Typical 3-cpt PK parameters
    const CL: f64 = 5.0;
    const V1: f64 = 10.0;
    const Q2: f64 = 2.0;
    const V2: f64 = 20.0;
    const Q3: f64 = 1.5;
    const V3: f64 = 30.0;
    const KA: f64 = 1.5;

    // --- Macro rates ---

    #[test]
    fn test_macro_rates_positive_sorted() {
        let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert!(alpha > beta, "alpha={} should be > beta={}", alpha, beta);
        assert!(beta > gamma, "beta={} should be > gamma={}", beta, gamma);
        assert!(gamma > 0.0, "gamma={} should be > 0", gamma);
        assert!(k21 > 0.0);
        assert!(k31 > 0.0);
    }

    #[test]
    fn test_macro_rates_vieta_product() {
        // alpha * beta * gamma = S0 = k10 * k21 * k31
        let k10 = CL / V1;
        let k21 = Q2 / V2;
        let k31 = Q3 / V3;
        let (alpha, beta, gamma, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha * beta * gamma, k10 * k21 * k31, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_rates_vieta_sum() {
        // alpha + beta + gamma = S2
        let k10 = CL / V1;
        let k12 = Q2 / V1;
        let k21 = Q2 / V2;
        let k13 = Q3 / V1;
        let k31 = Q3 / V3;
        let s2 = k10 + k12 + k13 + k21 + k31;
        let (alpha, beta, gamma, _, _) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha + beta + gamma, s2, epsilon = 1e-10);
    }

    #[test]
    fn test_macro_rates_values_vs_julia() {
        let (alpha, beta, gamma, k21, k31) = macro_rates_three_cpt(CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(alpha, 0.8844821357511055, epsilon = 1e-10);
        assert_relative_eq!(beta, 0.08033284468795598, epsilon = 1e-10);
        assert_relative_eq!(gamma, 0.03518501956093856, epsilon = 1e-10);
        assert_relative_eq!(k21, 0.1, epsilon = 1e-10);
        assert_relative_eq!(k31, 0.05, epsilon = 1e-10);
    }

    // --- IV Bolus ---

    #[test]
    fn test_iv_bolus_at_time_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_iv_bolus(&dose, 0.0, CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(c, 100.0 / V1, epsilon = 1e-10);
    }

    #[test]
    fn test_iv_bolus_vs_julia() {
        let dose = bolus_dose(100.0);
        let cases = [
            (0.0, 10.0),
            (0.5, 6.563342046483885),
            (1.0, 4.351421728970351),
            (3.0, 1.029346912256041),
            (10.0, 0.25111455768517715),
            (50.0, 0.04607673790588789),
        ];
        for (t, expected) in cases {
            let c = three_cpt_iv_bolus(&dose, t, CL, V1, Q2, V2, Q3, V3);
            assert_relative_eq!(c, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_iv_bolus_approaches_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_iv_bolus(&dose, 10000.0, CL, V1, Q2, V2, Q3, V3);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_iv_bolus_guard_clauses() {
        let dose = bolus_dose(100.0);
        assert_eq!(three_cpt_iv_bolus(&dose, -1.0, CL, V1, Q2, V2, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, 0.0, Q2, V2, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, 0.0, V1, Q2, V2, Q3, V3), 0.0);
        // v2=0 and v3=0 must not produce NaN
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, V1, Q2, 0.0, Q3, V3), 0.0);
        assert_eq!(three_cpt_iv_bolus(&dose, 1.0, CL, V1, Q2, V2, Q3, 0.0), 0.0);
    }

    // --- Infusion ---

    #[test]
    fn test_infusion_vs_julia() {
        let dose = infusion_dose(100.0, 50.0); // dur=2h
        let cases = [
            (0.5, 2.0389498366437318),
            (1.0, 3.383071924491621),
            (2.0, 4.888266633245966),
            (3.0, 2.229504095597091),
            (10.0, 0.26611910032293007),
        ];
        for (t, expected) in cases {
            let c = three_cpt_infusion(&dose, t, CL, V1, Q2, V2, Q3, V3);
            assert_relative_eq!(c, expected, epsilon = 1e-10, max_relative = 1e-10);
        }
    }

    #[test]
    fn test_infusion_continuity_at_end() {
        let dose = infusion_dose(100.0, 50.0); // dur=2h
        let dur = 2.0;
        let c_at = three_cpt_infusion(&dose, dur, CL, V1, Q2, V2, Q3, V3);
        let c_after = three_cpt_infusion(&dose, dur + 1e-10, CL, V1, Q2, V2, Q3, V3);
        assert_relative_eq!(c_at, c_after, epsilon = 1e-5);
    }

    #[test]
    fn test_infusion_after_decays() {
        let dose = infusion_dose(100.0, 50.0);
        let c1 = three_cpt_infusion(&dose, 50.0, CL, V1, Q2, V2, Q3, V3);
        let c2 = three_cpt_infusion(&dose, 100.0, CL, V1, Q2, V2, Q3, V3);
        assert!(c2 < c1);
    }

    // --- Oral ---

    #[test]
    fn test_oral_at_time_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_oral(&dose, 0.0, CL, V1, Q2, V2, Q3, V3, KA);
        assert_relative_eq!(c, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_oral_vs_julia() {
        let dose = bolus_dose(100.0);
        let cases = [
            (0.0, 0.0),
            (0.5, 4.191965255586688),
            (1.0, 4.745318116730585),
            (3.0, 1.7475777818041354),
            (10.0, 0.2614874522040594),
        ];
        for (t, expected) in cases {
            let c = three_cpt_oral(&dose, t, CL, V1, Q2, V2, Q3, V3, KA);
            assert_relative_eq!(c, expected, epsilon = 1e-8, max_relative = 1e-8);
        }
    }

    #[test]
    fn test_oral_approaches_zero() {
        let dose = bolus_dose(100.0);
        let c = three_cpt_oral(&dose, 10000.0, CL, V1, Q2, V2, Q3, V3, KA);
        assert!(c < 1e-20);
    }

    #[test]
    fn test_oral_bioavailability_scaling() {
        let dose = bolus_dose(100.0);
        let c_full = three_cpt_oral_f(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA, 1.0);
        let c_half = three_cpt_oral_f(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA, 0.5);
        assert_relative_eq!(c_half / c_full, 0.5, epsilon = 1e-10);
    }

    // --- Predict dispatcher ---

    #[test]
    fn test_predict_routes_iv_bolus() {
        let dose = bolus_dose(100.0);
        let direct = three_cpt_iv_bolus(&dose, 2.0, CL, V1, Q2, V2, Q3, V3);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_oral() {
        let dose = bolus_dose(100.0);
        let direct = three_cpt_oral(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, KA);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, Some(KA), None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }

    #[test]
    fn test_predict_routes_infusion() {
        let dose = infusion_dose(100.0, 50.0);
        let direct = three_cpt_infusion(&dose, 2.0, CL, V1, Q2, V2, Q3, V3);
        let via_predict = three_cpt_predict(&dose, 2.0, CL, V1, Q2, V2, Q3, V3, None, None);
        assert_relative_eq!(direct, via_predict, epsilon = 1e-12);
    }
}
