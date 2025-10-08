use crate::output::{FrameStat, ImuStats, Metadata, ProcessingOutput, Summary, TrajectoryPoint};
use crate::types::{ImuSample, SessionData};
use arbit_core::math::se3::TransformSE3;

/// Collects statistics during processing
pub struct AnalysisCollector {
    session: SessionData,
    frame_stats: Vec<FrameStat>,
    processing_times: Vec<f64>,
    imu_samples: Vec<ImuSample>,
    imu_gravity_count: usize,
    poses: Vec<TransformSE3>,
    preintegration_count: usize,
}

impl AnalysisCollector {
    pub fn new(session: SessionData) -> Self {
        Self {
            session,
            frame_stats: Vec::new(),
            processing_times: Vec::new(),
            imu_samples: Vec::new(),
            imu_gravity_count: 0,
            poses: Vec::new(),
            preintegration_count: 0,
        }
    }

    /// Increments the preintegration interval counter.
    /// Call this each time IMU preintegration is completed (when finish() is called).
    pub fn increment_preintegration_count(&mut self) {
        self.preintegration_count += 1;
    }

    pub fn add_frame_stat(&mut self, stat: FrameStat) {
        self.processing_times.push(stat.processing_ms);
        self.frame_stats.push(stat);
    }

    pub fn set_imu_samples(&mut self, samples: Vec<ImuSample>) {
        self.imu_samples = samples;
    }

    pub fn increment_gravity_count(&mut self) {
        self.imu_gravity_count += 1;
    }

    pub fn add_pose(&mut self, pose: TransformSE3) {
        self.poses.push(pose);
    }

    pub fn finalize(
        self,
        _trajectory: Vec<nalgebra::Vector3<f64>>,
        keyframes: u64,
        landmarks: u64,
        duration: f64,
    ) -> ProcessingOutput {
        let AnalysisCollector {
            session,
            frame_stats,
            processing_times,
            imu_samples,
            imu_gravity_count,
            poses,
            preintegration_count,
        } = self;

        let average_processing_ms = if processing_times.is_empty() {
            0.0
        } else {
            processing_times.iter().sum::<f64>() / processing_times.len() as f64
        };

        let frame_count = frame_stats.len();
        let imu_sample_count = imu_samples.len();

        let trajectory_points: Vec<TrajectoryPoint> = poses
            .iter()
            .enumerate()
            .map(|(i, pose)| {
                let timestamp = if !frame_stats.is_empty() && i < frame_stats.len() {
                    frame_stats[i].timestamp
                } else {
                    0.0
                };
                let quat = pose.rotation.quaternion();
                TrajectoryPoint {
                    timestamp,
                    x: pose.translation.x,
                    y: pose.translation.y,
                    z: pose.translation.z,
                    qw: quat.w,
                    qx: quat.i,
                    qy: quat.j,
                    qz: quat.k,
                }
            })
            .collect();

        let imu_stats = if !imu_samples.is_empty() {
            let avg_magnitude = imu_samples
                .iter()
                .map(|s| {
                    let (x, y, z) = s.accelerometer;
                    (x * x + y * y + z * z).sqrt()
                })
                .sum::<f64>()
                / imu_samples.len() as f64;

            // Check if gyroscope data is present (not all zeros)
            let has_gyro = imu_samples.iter().any(|s| {
                let (x, y, z) = s.gyroscope;
                x.abs() > 1e-9 || y.abs() > 1e-9 || z.abs() > 1e-9
            });

            // Compute simple bias estimates (mean of samples)
            let (gyro_bias_x, gyro_bias_y, gyro_bias_z) = if has_gyro {
                let count = imu_samples.len() as f64;
                let (sum_x, sum_y, sum_z) = imu_samples.iter().fold((0.0, 0.0, 0.0), |acc, s| {
                    let (x, y, z) = s.gyroscope;
                    (acc.0 + x, acc.1 + y, acc.2 + z)
                });
                (sum_x / count, sum_y / count, sum_z / count)
            } else {
                (0.0, 0.0, 0.0)
            };

            // Accel bias is deviation from 1g
            let (accel_bias_x, accel_bias_y, accel_bias_z) = {
                let count = imu_samples.len() as f64;
                let (sum_x, sum_y, sum_z) = imu_samples.iter().fold((0.0, 0.0, 0.0), |acc, s| {
                    let (x, y, z) = s.accelerometer;
                    (acc.0 + x, acc.1 + y, acc.2 + z)
                });
                let mean_x = sum_x / count;
                let mean_y = sum_y / count;
                let mean_z = sum_z / count;
                let magnitude = (mean_x * mean_x + mean_y * mean_y + mean_z * mean_z).sqrt();
                let error = magnitude - 9.80665;
                // Simple bias estimate proportional to mean
                (
                    mean_x * error / magnitude,
                    mean_y * error / magnitude,
                    mean_z * error / magnitude,
                )
            };

            // Compute average motion state from frame stats
            let average_motion_state = if !frame_stats.is_empty() {
                let mut stationary_count = 0;
                let mut slow_count = 0;
                let mut fast_count = 0;
                let mut unknown_count = 0;

                for stat in &frame_stats {
                    match stat.motion_state.as_deref() {
                        Some("Stationary") => stationary_count += 1,
                        Some("SlowMotion") => slow_count += 1,
                        Some("FastMotion") => fast_count += 1,
                        _ => unknown_count += 1,
                    }
                }

                let total = frame_stats.len() as f64;
                if unknown_count == frame_stats.len() {
                    "Unknown (motion not tracked)".to_string()
                } else {
                    let stationary_pct = (stationary_count as f64 / total * 100.0) as i32;
                    let slow_pct = (slow_count as f64 / total * 100.0) as i32;
                    let fast_pct = (fast_count as f64 / total * 100.0) as i32;
                    format!(
                        "Stationary: {}%, Slow: {}%, Fast: {}%",
                        stationary_pct, slow_pct, fast_pct
                    )
                }
            } else {
                "No frames".to_string()
            };

            Some(ImuStats {
                total_samples: imu_samples.len(),
                has_gyroscope_data: has_gyro,
                gravity_estimate_count: imu_gravity_count,
                average_gravity_magnitude: avg_magnitude,
                gyro_bias_x,
                gyro_bias_y,
                gyro_bias_z,
                accel_bias_x,
                accel_bias_y,
                accel_bias_z,
                preintegration_intervals: preintegration_count,
                average_motion_state,
            })
        } else {
            None
        };

        ProcessingOutput {
            metadata: Metadata {
                session_name: session.name.clone(),
                video_file: session.video_file.display().to_string(),
                imu_file: session
                    .imu_file
                    .as_ref()
                    .map(|p: &std::path::PathBuf| p.display().to_string()),
                processing_timestamp: chrono::Utc::now().to_rfc3339(),
                frame_count,
                imu_sample_count,
                duration_seconds: duration,
            },
            trajectory: trajectory_points,
            frame_stats,
            imu_stats,
            summary: Summary {
                total_frames: frame_count,
                keyframes_created: keyframes,
                landmarks_created: landmarks,
                average_processing_ms,
                imu_fusion_enabled: !imu_samples.is_empty(),
            },
        }
    }
}
