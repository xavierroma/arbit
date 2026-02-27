use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use arbit_core::contracts::{EngineSnapshot, TrackingState};
use arbit_core::time::{Clock, TimestampPolicy};
use arbit_engine::{EngineConfig, SlamEngine};
use arbit_providers::{ArKitFrame, ArKitIntrinsics, IosCameraProvider, PixelFormat};
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ReplayEvalError {
    #[error("failed to read dataset file '{path}': {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("failed to parse dataset JSON: {0}")]
    Parse(#[from] serde_json::Error),
    #[error("dataset is invalid: {0}")]
    InvalidDataset(String),
    #[error("engine frame {frame_id} did not complete before timeout")]
    FrameTimeout { frame_id: u64 },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayDataset {
    pub name: String,
    pub camera: ReplayCamera,
    pub thresholds: ReplayThresholds,
    pub frames: Vec<ReplayFrame>,
    #[serde(default)]
    pub imu: Vec<ReplayImuSample>,
    #[serde(default)]
    pub relocalization_events: Vec<RelocalizationEvent>,
}

impl ReplayDataset {
    fn validate(&self) -> Result<(), ReplayEvalError> {
        if self.frames.is_empty() {
            return Err(ReplayEvalError::InvalidDataset(
                "dataset must contain at least one frame".to_string(),
            ));
        }

        if self.camera.width == 0 || self.camera.height == 0 {
            return Err(ReplayEvalError::InvalidDataset(
                "camera width/height must be positive".to_string(),
            ));
        }

        let mut last_ts = f64::NEG_INFINITY;
        for (index, frame) in self.frames.iter().enumerate() {
            if frame.timestamp_seconds < last_ts {
                return Err(ReplayEvalError::InvalidDataset(format!(
                    "frame timestamps must be non-decreasing (index {index})"
                )));
            }
            last_ts = frame.timestamp_seconds;
        }

        for event in &self.relocalization_events {
            if event.drop_start_frame > event.drop_end_frame {
                return Err(ReplayEvalError::InvalidDataset(
                    "relocalization event start must be <= end".to_string(),
                ));
            }
            if event.drop_end_frame >= self.frames.len() {
                return Err(ReplayEvalError::InvalidDataset(
                    "relocalization event references out-of-range frame".to_string(),
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayCamera {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayThresholds {
    pub max_ate_rmse_m: f64,
    pub max_rpe_rmse_m: f64,
    pub max_relocalization_seconds: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayFrame {
    pub timestamp_seconds: f64,
    pub gt_pose_wc: [f64; 16],
    pub pattern: FramePattern,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum FramePattern {
    Checkerboard {
        cell_px: u32,
        low: u8,
        high: u8,
        phase_per_frame_px: u32,
    },
    GradientX {
        low: u8,
        high: u8,
    },
    Flat {
        value: u8,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ReplayImuSample {
    pub timestamp_seconds: f64,
    pub accel_mps2: [f64; 3],
    pub gyro_rps: [f64; 3],
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RelocalizationEvent {
    pub drop_start_frame: usize,
    pub drop_end_frame: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayEvaluationReport {
    pub dataset_name: String,
    pub frame_count: usize,
    pub ate_rmse_m: f64,
    pub rpe_rmse_m: f64,
    pub relocalization_p95_seconds: f64,
    pub relocalization_samples_seconds: Vec<f64>,
    pub failures: Vec<String>,
}

impl ReplayEvaluationReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

#[derive(Debug, Clone)]
struct DeterministicClock {
    now: Duration,
    step: Duration,
}

impl Default for DeterministicClock {
    fn default() -> Self {
        Self {
            now: Duration::from_millis(0),
            step: Duration::from_millis(1),
        }
    }
}

impl Clock for DeterministicClock {
    fn now(&mut self) -> Duration {
        self.now = self.now.saturating_add(self.step);
        self.now
    }
}

pub fn load_dataset(path: impl AsRef<Path>) -> Result<ReplayDataset, ReplayEvalError> {
    let path = path.as_ref();
    let payload = fs::read_to_string(path).map_err(|source| ReplayEvalError::Io {
        path: path.display().to_string(),
        source,
    })?;

    let dataset: ReplayDataset = serde_json::from_str(&payload)?;
    dataset.validate()?;
    Ok(dataset)
}

pub fn evaluate_dataset_path(
    path: impl AsRef<Path>,
) -> Result<ReplayEvaluationReport, ReplayEvalError> {
    let dataset = load_dataset(path)?;
    evaluate_dataset(&dataset)
}

pub fn evaluate_dataset(dataset: &ReplayDataset) -> Result<ReplayEvaluationReport, ReplayEvalError> {
    dataset.validate()?;

    let engine = SlamEngine::with_config(EngineConfig {
        frame_queue_size: 64,
        imu_queue_size: 8_192,
        keyframe_queue_size: 128,
        backend_queue_size: 64,
        ..EngineConfig::default()
    });

    let mut provider = IosCameraProvider::with_policy(TimestampPolicy::with_clock(
        DeterministicClock::default(),
    ));

    let mut estimates = Vec::with_capacity(dataset.frames.len());
    let mut timestamps = Vec::with_capacity(dataset.frames.len());
    let mut states = Vec::with_capacity(dataset.frames.len());

    let mut imu_cursor = 0usize;

    for (frame_index, frame) in dataset.frames.iter().enumerate() {
        while imu_cursor < dataset.imu.len()
            && dataset.imu[imu_cursor].timestamp_seconds <= frame.timestamp_seconds
        {
            let imu = &dataset.imu[imu_cursor];
            let _ = engine.ingest_imu(imu.timestamp_seconds, imu.accel_mps2, imu.gyro_rps);
            imu_cursor += 1;
        }

        let sample = provider.ingest_frame(ArKitFrame {
            timestamp: Duration::from_secs_f64(frame.timestamp_seconds),
            intrinsics: ArKitIntrinsics {
                fx: dataset.camera.fx,
                fy: dataset.camera.fy,
                cx: dataset.camera.cx,
                cy: dataset.camera.cy,
                skew: dataset.camera.skew,
                width: dataset.camera.width,
                height: dataset.camera.height,
                distortion: None,
            },
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: dataset.camera.width as usize * 4,
            data: synthesize_frame_bgra(
                &frame.pattern,
                dataset.camera.width,
                dataset.camera.height,
                frame_index,
            ),
        });

        if !engine.ingest_frame(&sample) {
            return Err(ReplayEvalError::InvalidDataset(
                "engine rejected a replay frame".to_string(),
            ));
        }

        let snapshot = wait_for_frame(&engine, (frame_index + 1) as u64, Duration::from_millis(800))?;
        estimates.push(snapshot.tracking.pose_wc);
        timestamps.push(snapshot.timestamp_seconds);
        states.push(snapshot.tracking.state);
    }

    // Keep the analyzer API deterministic: reset at end of run.
    engine.reset_session();

    let ate_rmse_m = compute_ate_rmse(&estimates, &dataset.frames);
    let rpe_rmse_m = compute_rpe_rmse(&estimates, &dataset.frames);
    let (relocalization_samples_seconds, mut failures) =
        compute_relocalization_recovery(&states, &timestamps, &dataset.relocalization_events);
    let relocalization_p95_seconds = percentile(&relocalization_samples_seconds, 0.95);

    if ate_rmse_m > dataset.thresholds.max_ate_rmse_m {
        failures.push(format!(
            "ATE RMSE {ate_rmse_m:.4} m exceeded threshold {:.4} m",
            dataset.thresholds.max_ate_rmse_m
        ));
    }

    if rpe_rmse_m > dataset.thresholds.max_rpe_rmse_m {
        failures.push(format!(
            "RPE RMSE {rpe_rmse_m:.4} m exceeded threshold {:.4} m",
            dataset.thresholds.max_rpe_rmse_m
        ));
    }

    if relocalization_p95_seconds > dataset.thresholds.max_relocalization_seconds {
        failures.push(format!(
            "Relocalization p95 {relocalization_p95_seconds:.4} s exceeded threshold {:.4} s",
            dataset.thresholds.max_relocalization_seconds
        ));
    }

    Ok(ReplayEvaluationReport {
        dataset_name: dataset.name.clone(),
        frame_count: dataset.frames.len(),
        ate_rmse_m,
        rpe_rmse_m,
        relocalization_p95_seconds,
        relocalization_samples_seconds,
        failures,
    })
}

fn wait_for_frame(
    engine: &SlamEngine,
    frame_id: u64,
    timeout: Duration,
) -> Result<EngineSnapshot, ReplayEvalError> {
    let start = Instant::now();
    loop {
        let snapshot = engine.snapshot();
        if snapshot.tracking.frame_id >= frame_id {
            return Ok(snapshot);
        }

        if start.elapsed() >= timeout {
            return Err(ReplayEvalError::FrameTimeout { frame_id });
        }

        thread::sleep(Duration::from_millis(1));
    }
}

fn synthesize_frame_bgra(
    pattern: &FramePattern,
    width: u32,
    height: u32,
    frame_index: usize,
) -> Arc<[u8]> {
    let mut data = vec![0_u8; width as usize * height as usize * 4];

    for y in 0..height as usize {
        for x in 0..width as usize {
            let luma = match pattern {
                FramePattern::Checkerboard {
                    cell_px,
                    low,
                    high,
                    phase_per_frame_px,
                } => {
                    let cell = (*cell_px as usize).max(1);
                    let phase = frame_index.saturating_mul(*phase_per_frame_px as usize);
                    let checker = ((x + phase) / cell + y / cell) % 2;
                    if checker == 0 { *low } else { *high }
                }
                FramePattern::GradientX { low, high } => {
                    if width <= 1 {
                        *low
                    } else {
                        let ratio = x as f64 / (width as f64 - 1.0);
                        let value = *low as f64 + (*high as f64 - *low as f64) * ratio;
                        value.round().clamp(0.0, 255.0) as u8
                    }
                }
                FramePattern::Flat { value } => *value,
            };

            let idx = (y * width as usize + x) * 4;
            data[idx] = luma;
            data[idx + 1] = luma;
            data[idx + 2] = luma;
            data[idx + 3] = 255;
        }
    }

    data.into()
}

fn compute_ate_rmse(estimates: &[[f64; 16]], frames: &[ReplayFrame]) -> f64 {
    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;

    for (estimated, gt) in estimates.iter().zip(frames.iter()) {
        let est = pose_translation(estimated);
        let gt = pose_translation(&gt.gt_pose_wc);
        let dx = est[0] - gt[0];
        let dy = est[1] - gt[1];
        let dz = est[2] - gt[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn compute_rpe_rmse(estimates: &[[f64; 16]], frames: &[ReplayFrame]) -> f64 {
    if estimates.len() < 2 || frames.len() < 2 {
        return 0.0;
    }

    let mut sum_sq = 0.0_f64;
    let mut count = 0usize;

    for idx in 1..estimates.len().min(frames.len()) {
        let est_prev = pose_translation(&estimates[idx - 1]);
        let est_curr = pose_translation(&estimates[idx]);
        let gt_prev = pose_translation(&frames[idx - 1].gt_pose_wc);
        let gt_curr = pose_translation(&frames[idx].gt_pose_wc);

        let est_delta = [
            est_curr[0] - est_prev[0],
            est_curr[1] - est_prev[1],
            est_curr[2] - est_prev[2],
        ];
        let gt_delta = [
            gt_curr[0] - gt_prev[0],
            gt_curr[1] - gt_prev[1],
            gt_curr[2] - gt_prev[2],
        ];

        let dx = est_delta[0] - gt_delta[0];
        let dy = est_delta[1] - gt_delta[1];
        let dz = est_delta[2] - gt_delta[2];
        sum_sq += dx * dx + dy * dy + dz * dz;
        count += 1;
    }

    if count == 0 {
        0.0
    } else {
        (sum_sq / count as f64).sqrt()
    }
}

fn compute_relocalization_recovery(
    states: &[TrackingState],
    timestamps: &[f64],
    events: &[RelocalizationEvent],
) -> (Vec<f64>, Vec<String>) {
    let mut samples = Vec::new();
    let mut failures = Vec::new();

    for event in events {
        if event.drop_end_frame >= states.len() || event.drop_end_frame >= timestamps.len() {
            failures.push("relocalization event references out-of-range frame".to_string());
            continue;
        }

        let saw_loss = states[event.drop_start_frame..=event.drop_end_frame]
            .iter()
            .any(|state| {
                matches!(
                    state,
                    TrackingState::Lost | TrackingState::Relocalizing | TrackingState::Initializing
                )
            });

        if !saw_loss {
            failures.push(format!(
                "no tracking loss observed during event [{}..={}]",
                event.drop_start_frame, event.drop_end_frame
            ));
            continue;
        }

        let recovered = (event.drop_end_frame..states.len()).find(|index| {
            matches!(states[*index], TrackingState::Tracking)
        });

        match recovered {
            Some(index) => {
                let recovery = (timestamps[index] - timestamps[event.drop_end_frame]).max(0.0);
                samples.push(recovery);
            }
            None => {
                failures.push(format!(
                    "tracking never recovered after event ending at frame {}",
                    event.drop_end_frame
                ));
            }
        }
    }

    (samples, failures)
}

fn pose_translation(pose: &[f64; 16]) -> [f64; 3] {
    [pose[3], pose[7], pose[11]]
}

fn percentile(samples: &[f64], q: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }

    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_returns_expected_quantile() {
        let values = vec![0.1, 0.3, 0.2, 0.4];
        let p95 = percentile(&values, 0.95);
        assert!(p95 >= 0.3);
    }

    #[test]
    fn pose_translation_reads_row_major_offsets() {
        let mut pose = [
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0, //
            0.0, 0.0, 0.0, 1.0, //
        ];
        pose[3] = 1.5;
        pose[7] = -2.0;
        pose[11] = 0.75;

        let t = pose_translation(&pose);
        assert_eq!(t, [1.5, -2.0, 0.75]);
    }
}
