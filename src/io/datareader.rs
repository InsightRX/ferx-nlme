use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;
use std::path::Path;

/// Read a NONMEM-format CSV file into a Population.
///
/// Expected columns (case-insensitive):
///   ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS, [covariates...]
///
/// EVID: 0=observation, 1=dose, 4=reset+dose
/// MDV: 1=missing dependent variable
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
) -> Result<Population, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {}", e))?;

    // Normalize headers to lowercase
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .iter()
        .map(|h| h.trim().to_lowercase())
        .collect();

    let col_idx = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let id_col = col_idx("id").ok_or("Missing ID column")?;
    let time_col = col_idx("time").ok_or("Missing TIME column")?;
    let dv_col = col_idx("dv").ok_or("Missing DV column")?;
    let evid_col = col_idx("evid");
    let amt_col = col_idx("amt");
    let cmt_col = col_idx("cmt");
    let rate_col = col_idx("rate");
    let mdv_col = col_idx("mdv");
    let ii_col = col_idx("ii");
    let ss_col = col_idx("ss");

    // Identify covariate columns
    let cov_names: Vec<String> = match covariate_columns {
        Some(cols) => cols.iter().map(|c| c.to_lowercase()).collect(),
        None => {
            // Auto-detect: columns not in standard set
            let standard = [
                "id", "time", "dv", "evid", "amt", "cmt", "rate", "mdv", "ii", "ss",
            ];
            headers
                .iter()
                .filter(|h| !standard.contains(&h.as_str()))
                .cloned()
                .collect()
        }
    };
    let cov_indices: Vec<(String, usize)> = cov_names
        .iter()
        .filter_map(|name| col_idx(name).map(|idx| (name.clone(), idx)))
        .collect();

    // Parse rows grouped by ID
    let mut rows_by_id: Vec<(String, Vec<Vec<String>>)> = Vec::new();
    let mut current_id = String::new();

    for result in rdr.records() {
        let record = result.map_err(|e| format!("CSV parse error: {}", e))?;
        let fields: Vec<String> = record.iter().map(|f| f.trim().to_string()).collect();

        let id = fields.get(id_col).cloned().unwrap_or_default();
        if id != current_id {
            current_id = id.clone();
            rows_by_id.push((id, Vec::new()));
        }
        rows_by_id.last_mut().unwrap().1.push(fields);
    }

    // Build subjects
    let mut subjects = Vec::new();
    for (id, rows) in &rows_by_id {
        let subject = parse_subject(
            id,
            rows,
            time_col,
            dv_col,
            evid_col,
            amt_col,
            cmt_col,
            rate_col,
            mdv_col,
            ii_col,
            ss_col,
            &cov_indices,
        )?;
        subjects.push(subject);
    }

    Ok(Population {
        subjects,
        covariate_names: cov_names,
        dv_column: "dv".to_string(),
    })
}

fn parse_f64(s: &str) -> f64 {
    s.parse::<f64>().unwrap_or(0.0)
}

fn parse_usize(s: &str) -> usize {
    s.parse::<usize>().unwrap_or(1)
}

fn parse_subject(
    id: &str,
    rows: &[Vec<String>],
    time_col: usize,
    dv_col: usize,
    evid_col: Option<usize>,
    amt_col: Option<usize>,
    cmt_col: Option<usize>,
    rate_col: Option<usize>,
    mdv_col: Option<usize>,
    ii_col: Option<usize>,
    ss_col: Option<usize>,
    cov_indices: &[(String, usize)],
) -> Result<Subject, String> {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();

    // Time-constant covariates: first non-missing value
    let mut covariates: HashMap<String, f64> = HashMap::new();
    for (name, idx) in cov_indices {
        for row in rows {
            if let Some(val_str) = row.get(*idx) {
                if let Ok(val) = val_str.parse::<f64>() {
                    if val.is_finite() {
                        covariates.insert(name.clone(), val);
                        break;
                    }
                }
            }
        }
    }

    // Time-varying covariates (LOCF) — detect if values change across rows
    let mut tvcov: HashMap<String, Vec<f64>> = HashMap::new();
    for (name, idx) in cov_indices {
        let vals: Vec<f64> = rows
            .iter()
            .map(|row| {
                row.get(*idx)
                    .and_then(|s| s.parse::<f64>().ok())
                    .unwrap_or(f64::NAN)
            })
            .collect();

        // Check if values change
        let first_val = vals.iter().find(|v| v.is_finite()).copied();
        let is_tv = first_val.map_or(false, |fv| {
            vals.iter()
                .any(|v| v.is_finite() && (*v - fv).abs() > 1e-12)
        });
        if is_tv {
            // LOCF fill
            let mut filled = Vec::with_capacity(vals.len());
            let mut last = first_val.unwrap_or(0.0);
            for v in &vals {
                if v.is_finite() {
                    last = *v;
                }
                filled.push(last);
            }
            tvcov.insert(name.clone(), filled);
        }
    }

    for row in rows {
        let time = parse_f64(row.get(time_col).map(|s| s.as_str()).unwrap_or("0"));
        let evid = evid_col
            .and_then(|c| row.get(c))
            .map(|s| parse_usize(s))
            .unwrap_or(0);
        let mdv = mdv_col
            .and_then(|c| row.get(c))
            .map(|s| parse_usize(s))
            .unwrap_or(0);

        if evid == 1 || evid == 4 {
            // Dose record
            let amt = amt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            let rate = rate_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ii = ii_col
                .and_then(|c| row.get(c))
                .map(|s| parse_f64(s))
                .unwrap_or(0.0);
            let ss = ss_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s) > 0)
                .unwrap_or(false);

            doses.push(DoseEvent::new(time, amt, cmt, rate, ss, ii));
        } else if evid == 0 && mdv == 0 {
            // Observation record
            let dv = parse_f64(row.get(dv_col).map(|s| s.as_str()).unwrap_or("0"));
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            obs_times.push(time);
            observations.push(dv);
            obs_cmts.push(cmt);
        }
    }

    // Sort doses by time
    doses.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap());

    Ok(Subject {
        id: id.to_string(),
        doses,
        obs_times,
        observations,
        obs_cmts,
        covariates,
        tvcov,
    })
}
