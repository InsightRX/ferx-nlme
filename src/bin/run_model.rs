use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ferx <model.ferx> --data <data.csv> [--threads N|auto]");
        eprintln!("       ferx <model.ferx> --simulate          [--threads N|auto]");
        eprintln!();
        eprintln!("Fits a NLME model and writes sdtab.csv with residuals.");
        eprintln!("Data must be in NONMEM format (ID, TIME, DV, EVID, AMT, CMT, ...)");
        eprintln!();
        eprintln!("--threads N    use N rayon workers (N > 0)");
        eprintln!("--threads 0    use rayon default (one worker per logical CPU)");
        eprintln!("--threads auto alias for --threads 0");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let data_path = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1));
    let simulate = args.iter().any(|a| a == "--simulate");
    let threads = parse_threads_flag(&args);

    // Configure rayon's global pool before any parallel work starts. build_global()
    // is once-per-process — correct for a CLI binary. Without a --threads flag we
    // leave rayon's default (one worker per logical CPU) in place.
    if let Some(n) = threads {
        if let Err(e) = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global()
        {
            eprintln!(
                "Warning: failed to configure thread pool with {} threads: {}",
                n, e
            );
        }
    }

    let t_start = Instant::now();
    let result = if let Some(csv_path) = data_path {
        ferx_nlme::run_model_with_data(model_path, csv_path)
    } else if simulate {
        ferx_nlme::run_model_simulate(model_path)
    } else {
        eprintln!("Error: specify --data <file.csv> or --simulate");
        std::process::exit(1);
    };
    let elapsed = t_start.elapsed();

    match result {
        Ok((fit_result, population)) => {
            // Derive model name from model file path
            let model_name = std::path::Path::new(model_path)
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("model");

            let sdtab_path = format!("{}-sdtab.csv", model_name);
            match ferx_nlme::io::output::write_sdtab_csv(&fit_result, &population, &sdtab_path) {
                Ok(()) => eprintln!("Residuals written to {}", sdtab_path),
                Err(e) => eprintln!("Warning: failed to write sdtab: {}", e),
            }

            let yaml_path = format!("{}-fit.yaml", model_name);
            match ferx_nlme::io::output::write_estimates_yaml(&fit_result, &yaml_path) {
                Ok(()) => eprintln!("Estimates written to {}", yaml_path),
                Err(e) => eprintln!("Warning: failed to write estimates: {}", e),
            }

            let elapsed_secs = elapsed.as_secs_f64();
            eprintln!("Elapsed fit time: {:.3}s", elapsed_secs);

            // Write timing file alongside outputs
            let timing_path = format!("{}-timing.txt", model_name);
            if let Ok(()) = std::fs::write(
                &timing_path,
                format!("elapsed_seconds={:.6}\n", elapsed_secs),
            ) {
                eprintln!("Timing written to {}", timing_path);
            }

            println!("\nFit completed!");
            println!("OFV: {:.4}", fit_result.ofv);
            println!("Elapsed: {:.3}s", elapsed_secs);
            for (name, val) in fit_result.theta_names.iter().zip(fit_result.theta.iter()) {
                println!("  {} = {:.6}", name, val);
            }
        }
        Err(e) => {
            eprintln!("Error: {}", e);
            std::process::exit(1);
        }
    }
}

/// Parse the optional `--threads` flag. Returns `None` when the flag is
/// absent, when its value is `0`, or when its value is `auto` — all of which
/// mean "leave rayon's default pool alone". Exits the process on a missing
/// value or any other non-parseable input so typos don't silently fall
/// through to the default.
fn parse_threads_flag(args: &[String]) -> Option<usize> {
    let idx = args.iter().position(|a| a == "--threads")?;
    let value = args.get(idx + 1).unwrap_or_else(|| {
        eprintln!("Error: --threads requires a value (positive integer, 0, or 'auto')");
        std::process::exit(1);
    });
    if value.eq_ignore_ascii_case("auto") || value == "0" {
        return None;
    }
    match value.parse::<usize>() {
        Ok(n) if n > 0 => Some(n),
        _ => {
            eprintln!(
                "Error: --threads expects a positive integer, 0, or 'auto'; got '{}'",
                value
            );
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_threads_flag;

    fn args(extra: &[&str]) -> Vec<String> {
        std::iter::once("ferx")
            .chain(extra.iter().copied())
            .map(String::from)
            .collect()
    }

    #[test]
    fn absent_flag_is_none() {
        assert_eq!(parse_threads_flag(&args(&["model.ferx"])), None);
    }

    #[test]
    fn positive_integer_parses() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "4"])), Some(4));
    }

    #[test]
    fn zero_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "0"])), None);
    }

    #[test]
    fn auto_means_default() {
        assert_eq!(parse_threads_flag(&args(&["--threads", "auto"])), None);
        assert_eq!(parse_threads_flag(&args(&["--threads", "AUTO"])), None);
    }
}
