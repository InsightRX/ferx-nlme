//! ODE-based predictions for subjects with dose events.
//!
//! Matches Julia's `_ode_predictions`: breaks the timeline at dose times,
//! applies bolus doses as state discontinuities, and integrates between.

use crate::ode::solver::{solve_ode, OdeSolverOptions};
use crate::types::Subject;
use std::collections::HashMap;

/// ODE specification for a model
pub struct OdeSpec {
    /// RHS function: (u, pk_params_flat, t, du) — writes derivatives into du
    pub rhs: Box<dyn Fn(&[f64], &[f64], f64, &mut [f64]) + Send + Sync>,
    /// Number of ODE states
    pub n_states: usize,
    /// Names of state variables (e.g., ["depot", "central"])
    pub state_names: Vec<String>,
    /// Index of the observable compartment (0-based) for DV
    pub obs_cmt_idx: usize,
}

/// Compute ODE-based predictions for a single subject.
///
/// `pk_params_flat` is a flat array of PK parameters passed to the RHS function.
/// Dose events are handled as state discontinuities between integration segments.
pub fn ode_predictions(
    ode: &OdeSpec,
    pk_params_flat: &[f64],
    subject: &Subject,
) -> Vec<f64> {
    let n = ode.n_states;
    let n_obs = subject.obs_times.len();
    let opts = OdeSolverOptions::default();

    let mut u = vec![0.0; n];
    let mut predictions = vec![f64::NAN; n_obs];

    // Build obs_time → index map
    let obs_map: HashMap<u64, usize> = subject
        .obs_times
        .iter()
        .enumerate()
        .map(|(i, &t)| (t.to_bits(), i))
        .collect();

    // Break timeline at dose times
    let t_last = subject.obs_times.iter().cloned().fold(0.0f64, f64::max);
    let mut break_times: Vec<f64> = vec![0.0];
    for dose in &subject.doses {
        break_times.push(dose.time);
    }
    break_times.push(t_last);
    break_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    break_times.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

    for k in 0..(break_times.len() - 1) {
        let t_start = break_times[k];
        let t_end = break_times[k + 1];

        // Apply bolus doses at t_start
        for dose in &subject.doses {
            if (dose.time - t_start).abs() < 1e-12 {
                assert!(
                    dose.rate == 0.0,
                    "Infusion doses (rate > 0) not yet supported in ODE models"
                );
                // dose.cmt is 1-based; state indices are 0-based
                let cmt_idx = dose.cmt - 1;
                if cmt_idx < n {
                    u[cmt_idx] += dose.amt;
                }
            }
        }

        // Record observations exactly at t_start (after dose)
        if let Some(&obs_idx) = obs_map.get(&t_start.to_bits()) {
            predictions[obs_idx] = u[ode.obs_cmt_idx];
        }

        // Observation times in this segment (t_start < t <= t_end)
        let mut saveat: Vec<f64> = subject
            .obs_times
            .iter()
            .filter(|&&t| t > t_start + 1e-12 && t <= t_end + 1e-12)
            .cloned()
            .collect();
        // Always include t_end so u is updated for next segment
        if saveat.is_empty() || (saveat.last().unwrap() - t_end).abs() > 1e-12 {
            saveat.push(t_end);
        }
        saveat.sort_by(|a, b| a.partial_cmp(b).unwrap());
        saveat.dedup_by(|a, b| (*a - *b).abs() < 1e-15);

        if (t_end - t_start).abs() < 1e-15 {
            continue;
        }

        // Integrate
        let sol = solve_ode(
            &*ode.rhs,
            &u,
            (t_start, t_end),
            pk_params_flat,
            &saveat,
            &opts,
        );

        // Extract predictions and update state
        for pt in &sol {
            if let Some(&obs_idx) = obs_map.get(&pt.t.to_bits()) {
                predictions[obs_idx] = pt.u[ode.obs_cmt_idx];
            }
        }

        // State at end of segment
        if let Some(last) = sol.last() {
            u.copy_from_slice(&last.u);
        }
    }

    // Clamp negatives
    for p in &mut predictions {
        if *p < 0.0 || p.is_nan() {
            *p = 0.0;
        }
    }

    predictions
}
