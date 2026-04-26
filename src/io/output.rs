use crate::types::*;

/// Print NONMEM-style results to stderr
pub fn print_results(result: &FitResult) {
    eprintln!("\n{}", "=".repeat(60));
    eprintln!("NONLINEAR MIXED EFFECTS MODEL ESTIMATION");
    eprintln!("{}", "=".repeat(60));

    eprintln!(
        "\nConverged: {}",
        if result.converged { "YES" } else { "NO" }
    );
    if result.method_chain.len() > 1 {
        let chain: Vec<&str> = result.method_chain.iter().map(|m| m.label()).collect();
        eprintln!("Estimation chain:  {}", chain.join(" → "));
    } else {
        eprintln!("Estimation method: {}", result.method.label());
    }
    eprintln!("Iterations: {}", result.n_iterations);

    eprintln!("\n--- Objective Function ---");
    eprintln!("OFV:  {:.4}", result.ofv);
    eprintln!("AIC:  {:.4}", result.aic);
    eprintln!("BIC:  {:.4}", result.bic);

    eprintln!(
        "\nSubjects: {}  Observations: {}  Parameters: {}",
        result.n_subjects, result.n_obs, result.n_parameters
    );

    // Theta estimates
    eprintln!("\n--- THETA Estimates ---");
    eprintln!(
        "{:<16} {:>12} {:>12} {:>10}",
        "Parameter", "Estimate", "SE", "%RSE"
    );
    eprintln!("{}", "-".repeat(52));
    for (i, name) in result.theta_names.iter().enumerate() {
        let est = result.theta[i];
        let is_fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let (se_str, rse_str) = if is_fixed {
            ("FIXED".to_string(), "FIXED".to_string())
        } else {
            match &result.se_theta {
                Some(se) => {
                    let se_val = se[i];
                    let rse = if est.abs() > 1e-12 {
                        (se_val / est.abs()) * 100.0
                    } else {
                        f64::NAN
                    };
                    (format!("{:.6}", se_val), format!("{:.1}", rse))
                }
                None => ("N/A".to_string(), "N/A".to_string()),
            }
        };
        eprintln!("{:<16} {:>12.6} {:>12} {:>10}", name, est, se_str, rse_str);
    }

    // Omega estimates
    eprintln!("\n--- OMEGA Estimates ---");
    let n_eta = result.omega.nrows();
    // Check if omega has off-diagonal elements
    let has_offdiag = (0..n_eta).any(|i| (0..i).any(|j| result.omega[(i, j)].abs() > 1e-15));
    for i in 0..n_eta {
        let var = result.omega[(i, i)];
        let cv = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
        let is_fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
        let se_str = if is_fixed {
            "FIXED".to_string()
        } else {
            match &result.se_omega {
                Some(se) if i < se.len() => format!("{:.6}", se[i]),
                _ => "N/A".to_string(),
            }
        };
        eprintln!(
            "  OMEGA({},{}) = {:.6}  (CV% = {:.1})  SE = {}",
            i + 1,
            i + 1,
            var,
            cv,
            se_str
        );
    }
    if has_offdiag {
        eprintln!("  --- Correlations ---");
        for i in 0..n_eta {
            for j in 0..i {
                let cov = result.omega[(i, j)];
                let var_i = result.omega[(i, i)];
                let var_j = result.omega[(j, j)];
                let corr = if var_i > 0.0 && var_j > 0.0 {
                    cov / (var_i.sqrt() * var_j.sqrt())
                } else {
                    0.0
                };
                eprintln!(
                    "  OMEGA({},{}) = {:.6}  (corr = {:.4})",
                    i + 1,
                    j + 1,
                    cov,
                    corr,
                );
            }
        }
    }

    // Sigma estimates
    eprintln!("\n--- SIGMA Estimates ---");
    for (i, &s) in result.sigma.iter().enumerate() {
        let is_fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let se_str = if is_fixed {
            "FIXED".to_string()
        } else {
            match &result.se_sigma {
                Some(se) if i < se.len() => format!("{:.6}", se[i]),
                _ => "N/A".to_string(),
            }
        };
        eprintln!("  SIGMA({}) = {:.6}  SE = {}", i + 1, s, se_str);
    }

    // IOV (KAPPA) estimates
    if let Some(ref iov) = result.omega_iov {
        eprintln!("\n--- KAPPA (IOV) Estimates ---");
        let n_kappa = iov.nrows();
        for i in 0..n_kappa {
            let var = iov[(i, i)];
            let cv = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
            let is_fixed = result.kappa_fixed.get(i).copied().unwrap_or(false);
            let se_str = if is_fixed {
                "FIXED".to_string()
            } else {
                match &result.se_kappa {
                    Some(se) if i < se.len() => format!("{:.6}", se[i]),
                    _ => "N/A".to_string(),
                }
            };
            let name = result.kappa_names.get(i).map(|s| s.as_str()).unwrap_or("KAPPA");
            eprintln!(
                "  {} = {:.6}  (CV% = {:.1})  SE = {}",
                name, var, cv, se_str
            );
        }
    }

    // SIR results
    if let Some(ess) = result.sir_ess {
        eprintln!("\n--- SIR Uncertainty (95% CI) ---");
        eprintln!("Effective sample size: {:.1}", ess);
        if let Some(ref ci) = result.sir_ci_theta {
            for (i, name) in result.theta_names.iter().enumerate() {
                if i < ci.len() {
                    eprintln!("  {} : [{:.6}, {:.6}]", name, ci[i].0, ci[i].1);
                }
            }
        }
        if let Some(ref ci) = result.sir_ci_omega {
            let n_eta = result.omega.nrows();
            for i in 0..n_eta.min(ci.len()) {
                eprintln!(
                    "  OMEGA({},{}) : [{:.6}, {:.6}]",
                    i + 1,
                    i + 1,
                    ci[i].0,
                    ci[i].1
                );
            }
        }
        if let Some(ref ci) = result.sir_ci_sigma {
            for (i, (lo, hi)) in ci.iter().enumerate() {
                eprintln!("  SIGMA({}) : [{:.6}, {:.6}]", i + 1, lo, hi);
            }
        }
    }

    // Warnings
    if !result.warnings.is_empty() {
        eprintln!("\n--- Warnings ---");
        for w in &result.warnings {
            eprintln!("  * {}", w);
        }
    }

    eprintln!("{}\n", "=".repeat(60));
}

/// Generate SDTAB-like output as vectors of (header, values) pairs
pub fn sdtab(result: &FitResult, population: &Population) -> Vec<(String, Vec<f64>)> {
    let n_total: usize = result.subjects.iter().map(|s| s.ipred.len()).sum();
    let n_eta = if result.subjects.is_empty() {
        0
    } else {
        result.subjects[0].eta.len()
    };

    let any_cens = result
        .subjects
        .iter()
        .any(|s| s.cens.iter().any(|&c| c != 0));
    let any_occ = population.subjects.iter().any(|s| !s.occasions.is_empty());

    let mut ids = Vec::with_capacity(n_total);
    let mut times = Vec::with_capacity(n_total);
    let mut dvs = Vec::with_capacity(n_total);
    let mut cens_col = Vec::with_capacity(n_total);
    let mut occ_col = Vec::with_capacity(n_total);
    let mut preds = Vec::with_capacity(n_total);
    let mut ipreds = Vec::with_capacity(n_total);
    let mut cwres_vec = Vec::with_capacity(n_total);
    let mut iwres_vec = Vec::with_capacity(n_total);
    let mut eta_cols: Vec<Vec<f64>> = (0..n_eta).map(|_| Vec::with_capacity(n_total)).collect();

    for (si, sr) in result.subjects.iter().enumerate() {
        let subj = &population.subjects[si];
        for j in 0..sr.ipred.len() {
            ids.push(si as f64 + 1.0);
            times.push(subj.obs_times[j]);
            dvs.push(subj.observations[j]);
            cens_col.push(sr.cens.get(j).copied().unwrap_or(0) as f64);
            occ_col.push(subj.occasions.get(j).copied().unwrap_or(0) as f64);
            preds.push(sr.pred[j]);
            ipreds.push(sr.ipred[j]);
            cwres_vec.push(sr.cwres[j]);
            iwres_vec.push(sr.iwres[j]);
            for k in 0..n_eta {
                eta_cols[k].push(sr.eta[k]);
            }
        }
    }

    let mut cols = vec![
        ("ID".to_string(), ids),
        ("TIME".to_string(), times),
        ("DV".to_string(), dvs),
    ];
    if any_cens {
        cols.push(("CENS".to_string(), cens_col));
    }
    if any_occ {
        cols.push(("OCC".to_string(), occ_col));
    }
    cols.extend([
        ("PRED".to_string(), preds),
        ("IPRED".to_string(), ipreds),
        ("CWRES".to_string(), cwres_vec),
        ("IWRES".to_string(), iwres_vec),
    ]);
    for k in 0..n_eta {
        cols.push((format!("ETA{}", k + 1), eta_cols[k].clone()));
    }

    cols
}

/// Write SDTAB as a CSV file
pub fn write_sdtab_csv(
    result: &FitResult,
    population: &Population,
    path: &str,
) -> Result<(), String> {
    let cols = sdtab(result, population);
    if cols.is_empty() {
        return Err("No data to write".to_string());
    }

    let n_rows = cols[0].1.len();
    let mut f =
        std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

    use std::io::Write;

    // Header
    let header: Vec<&str> = cols.iter().map(|(name, _)| name.as_str()).collect();
    writeln!(f, "{}", header.join(",")).map_err(|e| e.to_string())?;

    // Rows. NaN (used for BLOQ IWRES/CWRES) is written as an empty cell so
    // downstream tools handle it as missing rather than a sentinel.
    for row in 0..n_rows {
        let vals: Vec<String> = cols
            .iter()
            .map(|(_, values)| {
                let v = values[row];
                if v.is_nan() {
                    String::new()
                } else {
                    format!("{:.6}", v)
                }
            })
            .collect();
        writeln!(f, "{}", vals.join(",")).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Write parameter estimates and uncertainty as YAML
pub fn write_estimates_yaml(result: &FitResult, path: &str) -> Result<(), String> {
    use std::io::Write;

    let mut f =
        std::fs::File::create(path).map_err(|e| format!("Failed to create {}: {}", path, e))?;

    writeln!(f, "model:").map_err(|e| e.to_string())?;
    writeln!(f, "  converged: {}", result.converged).map_err(|e| e.to_string())?;
    writeln!(f, "  method: {}", result.method.label()).map_err(|e| e.to_string())?;
    if result.method_chain.len() > 1 {
        let chain: Vec<&str> = result.method_chain.iter().map(|m| m.label()).collect();
        writeln!(f, "  method_chain: [{}]", chain.join(", ")).map_err(|e| e.to_string())?;
    }

    writeln!(f, "\nobjective_function:").map_err(|e| e.to_string())?;
    writeln!(f, "  ofv: {:.6}", result.ofv).map_err(|e| e.to_string())?;
    writeln!(f, "  aic: {:.6}", result.aic).map_err(|e| e.to_string())?;
    writeln!(f, "  bic: {:.6}", result.bic).map_err(|e| e.to_string())?;

    writeln!(f, "\ndata:").map_err(|e| e.to_string())?;
    writeln!(f, "  n_subjects: {}", result.n_subjects).map_err(|e| e.to_string())?;
    writeln!(f, "  n_observations: {}", result.n_obs).map_err(|e| e.to_string())?;
    writeln!(f, "  n_parameters: {}", result.n_parameters).map_err(|e| e.to_string())?;

    writeln!(f, "\ntheta:").map_err(|e| e.to_string())?;
    for (i, name) in result.theta_names.iter().enumerate() {
        let est = result.theta[i];
        let is_fixed = result.theta_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_theta.as_ref().map(|v| v[i]);
        let rse = se.map(|s| {
            if est.abs() > 1e-12 {
                (s / est.abs()) * 100.0
            } else {
                f64::NAN
            }
        });
        writeln!(f, "  {}:", name).map_err(|e| e.to_string())?;
        writeln!(f, "    estimate: {:.6}", est).map_err(|e| e.to_string())?;
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
            writeln!(f, "    rse_pct: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(s) => {
                    writeln!(f, "    se: {:.6}", s).map_err(|e| e.to_string())?;
                    writeln!(f, "    rse_pct: {:.2}", rse.unwrap()).map_err(|e| e.to_string())?;
                }
                None => {
                    writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
                    writeln!(f, "    rse_pct: ~").map_err(|e| e.to_string())?;
                }
            }
        }
    }

    let n_eta = result.omega.nrows();
    writeln!(f, "\nomega:").map_err(|e| e.to_string())?;
    for i in 0..n_eta {
        let var = result.omega[(i, i)];
        let cv_pct = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
        let is_fixed = result.omega_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_omega.as_ref().and_then(|v| v.get(i).copied());
        writeln!(f, "  omega_{}_{}:", i + 1, i + 1).map_err(|e| e.to_string())?;
        writeln!(f, "    variance: {:.6}", var).map_err(|e| e.to_string())?;
        writeln!(f, "    cv_pct: {:.2}", cv_pct).map_err(|e| e.to_string())?;
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(s) => writeln!(f, "    se: {:.6}", s).map_err(|e| e.to_string())?,
                None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
            }
        }
    }
    // Off-diagonal covariances
    for i in 0..n_eta {
        for j in 0..i {
            let cov = result.omega[(i, j)];
            if cov.abs() > 1e-15 {
                let var_i = result.omega[(i, i)];
                let var_j = result.omega[(j, j)];
                let corr = if var_i > 0.0 && var_j > 0.0 {
                    cov / (var_i.sqrt() * var_j.sqrt())
                } else {
                    0.0
                };
                writeln!(f, "  omega_{}_{}:", i + 1, j + 1).map_err(|e| e.to_string())?;
                writeln!(f, "    covariance: {:.6}", cov).map_err(|e| e.to_string())?;
                writeln!(f, "    correlation: {:.6}", corr).map_err(|e| e.to_string())?;
            }
        }
    }

    writeln!(f, "\nsigma:").map_err(|e| e.to_string())?;
    for (i, &s) in result.sigma.iter().enumerate() {
        let is_fixed = result.sigma_fixed.get(i).copied().unwrap_or(false);
        let se = result.se_sigma.as_ref().and_then(|v| v.get(i).copied());
        writeln!(f, "  sigma_{}:", i + 1).map_err(|e| e.to_string())?;
        writeln!(f, "    estimate: {:.6}", s).map_err(|e| e.to_string())?;
        if is_fixed {
            writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
            writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
        } else {
            match se {
                Some(sv) => writeln!(f, "    se: {:.6}", sv).map_err(|e| e.to_string())?,
                None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
            }
        }
    }

    // IOV (KAPPA) block
    if let Some(ref iov) = result.omega_iov {
        writeln!(f, "\nomega_iov:").map_err(|e| e.to_string())?;
        let n_kappa = iov.nrows();
        for i in 0..n_kappa {
            let var = iov[(i, i)];
            let cv_pct = if var > 0.0 { var.sqrt() * 100.0 } else { 0.0 };
            let is_fixed = result.kappa_fixed.get(i).copied().unwrap_or(false);
            let se = result.se_kappa.as_ref().and_then(|v| v.get(i).copied());
            let name = result.kappa_names.get(i).cloned().unwrap_or_else(|| format!("kappa_{}", i + 1));
            writeln!(f, "  {}:", name).map_err(|e| e.to_string())?;
            writeln!(f, "    variance: {:.6}", var).map_err(|e| e.to_string())?;
            writeln!(f, "    cv_pct: {:.2}", cv_pct).map_err(|e| e.to_string())?;
            if is_fixed {
                writeln!(f, "    fixed: true").map_err(|e| e.to_string())?;
                writeln!(f, "    se: ~").map_err(|e| e.to_string())?;
            } else {
                match se {
                    Some(sv) => writeln!(f, "    se: {:.6}", sv).map_err(|e| e.to_string())?,
                    None => writeln!(f, "    se: ~").map_err(|e| e.to_string())?,
                }
            }
        }
    }

    // SIR section
    if let Some(ess) = result.sir_ess {
        writeln!(f, "\nsir:").map_err(|e| e.to_string())?;
        writeln!(f, "  effective_sample_size: {:.1}", ess).map_err(|e| e.to_string())?;
        if let Some(ref ci) = result.sir_ci_theta {
            writeln!(f, "  ci_theta:").map_err(|e| e.to_string())?;
            for (i, name) in result.theta_names.iter().enumerate() {
                if i < ci.len() {
                    writeln!(f, "    {}:", name).map_err(|e| e.to_string())?;
                    writeln!(f, "      lower: {:.6}", ci[i].0).map_err(|e| e.to_string())?;
                    writeln!(f, "      upper: {:.6}", ci[i].1).map_err(|e| e.to_string())?;
                }
            }
        }
        if let Some(ref ci) = result.sir_ci_omega {
            writeln!(f, "  ci_omega:").map_err(|e| e.to_string())?;
            for (i, (lo, hi)) in ci.iter().enumerate() {
                writeln!(f, "    omega_{}{}:", i + 1, i + 1).map_err(|e| e.to_string())?;
                writeln!(f, "      lower: {:.6}", lo).map_err(|e| e.to_string())?;
                writeln!(f, "      upper: {:.6}", hi).map_err(|e| e.to_string())?;
            }
        }
        if let Some(ref ci) = result.sir_ci_sigma {
            writeln!(f, "  ci_sigma:").map_err(|e| e.to_string())?;
            for (i, (lo, hi)) in ci.iter().enumerate() {
                writeln!(f, "    sigma_{}:", i + 1).map_err(|e| e.to_string())?;
                writeln!(f, "      lower: {:.6}", lo).map_err(|e| e.to_string())?;
                writeln!(f, "      upper: {:.6}", hi).map_err(|e| e.to_string())?;
            }
        }
    }

    if !result.warnings.is_empty() {
        writeln!(f, "\nwarnings:").map_err(|e| e.to_string())?;
        for w in &result.warnings {
            writeln!(f, "  - \"{}\"", w).map_err(|e| e.to_string())?;
        }
    }

    Ok(())
}

/// Parameter table as text
pub fn parameter_table(result: &FitResult) -> String {
    let mut lines = Vec::new();
    lines.push(format!(
        "{:<20} {:>12} {:>12} {:>10} {:>8}",
        "Parameter", "Estimate", "SE", "%RSE", "Type"
    ));
    lines.push("-".repeat(64));

    for (i, name) in result.theta_names.iter().enumerate() {
        let est = result.theta[i];
        let (se_str, rse_str) = match &result.se_theta {
            Some(se) => {
                let se_val = se[i];
                let rse = if est.abs() > 1e-12 {
                    (se_val / est.abs()) * 100.0
                } else {
                    f64::NAN
                };
                (format!("{:.6}", se_val), format!("{:.1}", rse))
            }
            None => ("---".to_string(), "---".to_string()),
        };
        lines.push(format!(
            "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
            name, est, se_str, rse_str, "THETA"
        ));
    }

    let n_eta = result.omega.nrows();
    for i in 0..n_eta {
        for j in 0..=i {
            let val = result.omega[(i, j)];
            let name = format!("OMEGA({},{})", i + 1, j + 1);
            lines.push(format!(
                "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
                name, val, "---", "---", "OMEGA"
            ));
        }
    }

    for (i, &s) in result.sigma.iter().enumerate() {
        let name = format!("SIGMA({})", i + 1);
        lines.push(format!(
            "{:<20} {:>12.6} {:>12} {:>10} {:>8}",
            name, s, "---", "---", "SIGMA"
        ));
    }

    lines.join("\n")
}
