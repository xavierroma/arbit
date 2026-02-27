# arbit-eval

Deterministic replay and quality-gate evaluation for ARBit.

## What it measures

- Absolute Trajectory Error (ATE RMSE, translation-only)
- Relative Pose Error (RPE RMSE, translation-only)
- Relocalization recovery latency (p95)

## Run

```bash
./scripts/run-replay-eval.sh
```

Or directly:

```bash
cargo run -p arbit-eval --bin replay_eval -- crates/arbit-eval/tests/data/micro_replay_v1.json
```

## Dataset format (JSON)

Top-level keys:

- `name`
- `camera`: `fx`, `fy`, `cx`, `cy`, `skew`, `width`, `height`
- `thresholds`: `max_ate_rmse_m`, `max_rpe_rmse_m`, `max_relocalization_seconds`
- `frames`: array of replay frames
- `imu`: optional array of IMU samples
- `relocalization_events`: optional array of drop windows

Each frame has:

- `timestamp_seconds`
- `gt_pose_wc`: 4x4 row-major pose matrix
- `pattern`: synthetic frame generator (`checkerboard`, `gradient_x`, `flat`)

Each relocalization event has:

- `drop_start_frame`
- `drop_end_frame`
