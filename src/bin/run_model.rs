use std::env;
use std::time::Instant;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: ferx <model.ferx> --data <data.csv> [--threads N]");
        eprintln!("       ferx <model.ferx> --simulate          [--threads N]");
        eprintln!();
        eprintln!("Fits a NLME model and writes sdtab.csv with residuals.");
        eprintln!("Data must be in NONMEM format (ID, TIME, DV, EVID, AMT, CMT, ...)");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let data_path = args
        .iter()
        .position(|a| a == "--data")
        .and_then(|i| args.get(i + 1));
    let simulate = args.iter().any(|a| a == "--simulate");
    let threads = args
        .iter()
        .position(|a| a == "--threads")
        .and_then(|i| args.get(i + 1))
        .map(|s| {
            s.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Error: --threads expects a positive integer, got '{}'", s);
                std::process::exit(1);
            })
        })
        .filter(|&n| n > 0);

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
