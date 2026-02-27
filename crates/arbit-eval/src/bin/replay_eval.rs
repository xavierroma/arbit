use std::env;
use std::process;

fn main() {
    let mut args = env::args();
    let _bin = args.next();
    let Some(path) = args.next() else {
        eprintln!("usage: replay_eval <dataset.json>");
        process::exit(2);
    };

    match arbit_eval::evaluate_dataset_path(&path) {
        Ok(report) => {
            println!("dataset: {}", report.dataset_name);
            println!("frames: {}", report.frame_count);
            println!("ATE RMSE (m): {:.6}", report.ate_rmse_m);
            println!("RPE RMSE (m): {:.6}", report.rpe_rmse_m);
            println!(
                "Relocalization p95 (s): {:.6}",
                report.relocalization_p95_seconds
            );
            if report.failures.is_empty() {
                println!("status: PASS");
                process::exit(0);
            }

            println!("status: FAIL");
            for failure in report.failures {
                println!("- {}", failure);
            }
            process::exit(1);
        }
        Err(err) => {
            eprintln!("evaluation failed: {err}");
            process::exit(1);
        }
    }
}
