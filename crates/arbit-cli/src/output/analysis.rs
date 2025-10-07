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
        }
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

            Some(ImuStats {
                accelerometer_samples: imu_samples.len(),
                gravity_estimate_count: imu_gravity_count,
                average_gravity_magnitude: avg_magnitude,
            })
        } else {
            None
        };

        ProcessingOutput {
            metadata: Metadata {
                session_name: session.name.clone(),
                video_file: session.video_file.display().to_string(),
                imu_file: session.imu_file.as_ref().map(|p| p.display().to_string()),
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
