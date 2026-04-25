use crate::types::{DoseEvent, Population, Subject};
use std::collections::HashMap;
use std::path::Path;

/// Read a NONMEM-format CSV file into a Population.
///
/// Expected columns (case-insensitive):
///   ID, TIME, DV, EVID, AMT, CMT, RATE, MDV, II, SS, CENS, [covariates...]
///
/// EVID: 0=observation, 1=dose, 4=reset+dose
/// MDV: 1=missing dependent variable
/// CENS: 1=observation is below LLOQ (DV carries the LLOQ value); 0 otherwise
///
/// `iov_column`: when `Some(name)`, that column is read as the occasion index
/// (integer) and stored in `Subject::occasions` / `Subject::dose_occasions`.
/// The column is excluded from the covariate auto-detection list.
pub fn read_nonmem_csv(
    path: &Path,
    covariate_columns: Option<&[&str]>,
    iov_column: Option<&str>,
) -> Result<Population, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("Failed to open CSV: {}", e))?;

    // Preserve original header casing for covariate names. Standard NONMEM
    // columns are matched case-insensitively so that legacy CSVs (e.g. `Id`,
    // `TIME`) keep working; covariate lookups remain case-sensitive.
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("Failed to read headers: {}", e))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect();

    let col_idx_ci =
        |name: &str| -> Option<usize> { headers.iter().position(|h| h.eq_ignore_ascii_case(name)) };
    let col_idx_cs = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let id_col = col_idx_ci("id").ok_or("Missing ID column")?;
    let time_col = col_idx_ci("time").ok_or("Missing TIME column")?;
    let dv_col = col_idx_ci("dv").ok_or("Missing DV column")?;
    let evid_col = col_idx_ci("evid");
    let amt_col = col_idx_ci("amt");
    let cmt_col = col_idx_ci("cmt");
    let rate_col = col_idx_ci("rate");
    let mdv_col = col_idx_ci("mdv");
    let ii_col = col_idx_ci("ii");
    let ss_col = col_idx_ci("ss");
    let cens_col = col_idx_ci("cens");

    // IOV occasion column (case-insensitive lookup of user-specified name)
    let occ_col: Option<usize> = iov_column.and_then(|name| col_idx_ci(name));
    if iov_column.is_some() && occ_col.is_none() {
        return Err(format!(
            "iov_column '{}' not found in dataset headers",
            iov_column.unwrap()
        ));
    }

    const STANDARD_COLS: &[&str] = &[
        "id", "time", "dv", "evid", "amt", "cmt", "rate", "mdv", "ii", "ss", "cens",
    ];
    let is_standard = |h: &str| {
        STANDARD_COLS.iter().any(|s| h.eq_ignore_ascii_case(s))
            || iov_column.map_or(false, |iov| h.eq_ignore_ascii_case(iov))
    };

    // Identify covariate columns (names preserved in their original case).
    let cov_names: Vec<String> = match covariate_columns {
        Some(cols) => cols.iter().map(|c| c.to_string()).collect(),
        None => headers
            .iter()
            .filter(|h| !is_standard(h))
            .cloned()
            .collect(),
    };
    let cov_indices: Vec<(String, usize)> = cov_names
        .iter()
        .filter_map(|name| col_idx_cs(name).map(|idx| (name.clone(), idx)))
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
            cens_col,
            occ_col,
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

fn parse_u32(s: &str) -> u32 {
    s.parse::<u32>().unwrap_or(0)
}

#[allow(clippy::too_many_arguments)]
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
    cens_col: Option<usize>,
    occ_col: Option<usize>,
    cov_indices: &[(String, usize)],
) -> Result<Subject, String> {
    let mut doses = Vec::new();
    let mut obs_times = Vec::new();
    let mut observations = Vec::new();
    let mut obs_cmts = Vec::new();
    let mut cens = Vec::new();
    let mut occasions: Vec<u32> = Vec::new();
    let mut dose_occasions: Vec<u32> = Vec::new();

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
        let occ = occ_col
            .and_then(|c| row.get(c))
            .map(|s| parse_u32(s))
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
            if occ_col.is_some() {
                dose_occasions.push(occ);
            }
        } else if evid == 0 && mdv == 0 {
            // Observation record
            let dv = parse_f64(row.get(dv_col).map(|s| s.as_str()).unwrap_or("0"));
            let cmt = cmt_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(1);
            let cens_flag = cens_col
                .and_then(|c| row.get(c))
                .map(|s| parse_usize(s))
                .unwrap_or(0);
            obs_times.push(time);
            observations.push(dv);
            obs_cmts.push(cmt);
            cens.push(if cens_flag > 0 { 1u8 } else { 0u8 });
            if occ_col.is_some() {
                occasions.push(occ);
            }
        }
    }

    // Sort doses by time (keeping dose_occasions in sync)
    let mut dose_pairs: Vec<(DoseEvent, u32)> = if dose_occasions.is_empty() {
        doses.iter().cloned().map(|d| (d, 0)).collect()
    } else {
        doses.into_iter().zip(dose_occasions.into_iter()).collect()
    };
    dose_pairs.sort_by(|a, b| a.0.time.partial_cmp(&b.0.time).unwrap());
    let (sorted_doses, sorted_dose_occ): (Vec<_>, Vec<_>) = dose_pairs.into_iter().unzip();
    let dose_occasions_out = if occ_col.is_some() { sorted_dose_occ } else { Vec::new() };

    Ok(Subject {
        id: id.to_string(),
        doses: sorted_doses,
        obs_times,
        observations,
        obs_cmts,
        covariates,
        tvcov,
        cens,
        occasions,
        dose_occasions: dose_occasions_out,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_csv(content: &str) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_occ_absent_gives_empty_occasions() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n1,1,5.0,0,.\n1,2,3.0,0,.\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, None).unwrap();
        assert!(pop.subjects[0].occasions.is_empty());
        assert!(pop.subjects[0].dose_occasions.is_empty());
    }

    #[test]
    fn test_parse_subject_reads_occ_column() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC\n\
                   1,0,.,1,100,1\n\
                   1,1,5.0,0,.,1\n\
                   1,2,3.0,0,.,1\n\
                   1,7,.,1,100,2\n\
                   1,8,4.0,0,.,2\n\
                   1,9,2.5,0,.,2\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        let subj = &pop.subjects[0];
        // Two obs in occ 1, two in occ 2 (dose rows are stripped from occasions)
        assert_eq!(subj.occasions, vec![1, 1, 2, 2]);
        assert_eq!(subj.dose_occasions, vec![1, 2]);
    }

    #[test]
    fn test_occ_column_excluded_from_covariates() {
        let csv = "ID,TIME,DV,EVID,AMT,OCC,WT\n\
                   1,0,.,1,100,1,70\n\
                   1,1,5.0,0,.,1,70\n";
        let f = write_csv(csv);
        let pop = read_nonmem_csv(f.path(), None, Some("OCC")).unwrap();
        // OCC should NOT appear as a covariate; WT should
        assert!(!pop.covariate_names.contains(&"OCC".to_string()));
        assert!(pop.covariate_names.contains(&"WT".to_string()));
    }

    #[test]
    fn test_missing_iov_column_errors() {
        let csv = "ID,TIME,DV,EVID,AMT\n1,0,.,1,100\n";
        let f = write_csv(csv);
        let result = read_nonmem_csv(f.path(), None, Some("OCC"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("iov_column"));
    }
}
