use std::path::PathBuf;

use arbit_eval::evaluate_dataset_path;

#[test]
fn micro_replay_dataset_passes_quality_gates() {
    let dataset_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("data")
        .join("micro_replay_v1.json");

    let report = evaluate_dataset_path(&dataset_path).expect("dataset should evaluate");

    assert!(
        report.passed(),
        "replay gates failed: {:?}",
        report.failures
    );
    assert!(report.ate_rmse_m >= 0.0);
    assert!(report.rpe_rmse_m >= 0.0);
    assert!(report.relocalization_p95_seconds >= 0.0);
}
