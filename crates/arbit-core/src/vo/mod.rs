use crate::logs::{FrameLogEntry, ReplayLog};
use crate::math::se3::TransformSE3;
use log::{debug, info, warn};
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy)]
pub struct VoLoopConfig {
    pub metrics_window: usize,
    pub min_inlier_ratio: f32,
    pub max_forward_backward_error: f32,
}

impl Default for VoLoopConfig {
    fn default() -> Self {
        Self {
            metrics_window: 30,
            min_inlier_ratio: 0.3,
            max_forward_backward_error: 4.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VoStatus {
    Healthy,
    Degrading,
    Lost,
}

#[derive(Debug, Clone, Copy)]
pub struct FrameObservation {
    pub frame_index: u64,
    pub timestamp_seconds: f64,
    pub pose: TransformSE3,
    pub track_count: usize,
    pub inlier_ratio: f32,
    pub forward_backward_error: f32,
}

#[derive(Debug, Default, Clone)]
pub struct VoMetrics {
    window: VecDeque<FrameObservation>,
    pub average_inlier_ratio: f32,
    pub average_forward_backward_error: f32,
    pub track_count_mean: f32,
}

impl VoMetrics {
    fn update(&mut self, observation: FrameObservation, capacity: usize) {
        if self.window.len() == capacity {
            self.window.pop_front();
        }
        self.window.push_back(observation);

        let mut inlier_sum = 0.0f32;
        let mut fb_sum = 0.0f32;
        let mut track_sum = 0.0f32;
        let len = self.window.len() as f32;

        for obs in &self.window {
            inlier_sum += obs.inlier_ratio;
            fb_sum += obs.forward_backward_error;
            track_sum += obs.track_count as f32;
        }

        self.average_inlier_ratio = if len > 0.0 { inlier_sum / len } else { 0.0 };
        self.average_forward_backward_error = if len > 0.0 { fb_sum / len } else { 0.0 };
        self.track_count_mean = if len > 0.0 { track_sum / len } else { 0.0 };
    }
}

pub struct VoLoop {
    config: VoLoopConfig,
    metrics: VoMetrics,
    log: ReplayLog,
    consecutive_losses: usize,
}

impl VoLoop {
    pub fn new(config: VoLoopConfig) -> Self {
        Self {
            config,
            metrics: VoMetrics::default(),
            log: ReplayLog::default(),
            consecutive_losses: 0,
        }
    }

    pub fn process(&mut self, observation: FrameObservation) -> VoStatus {
        debug!(target: "arbit_core::vo", "Processing frame {}: tracks={}, inliers={:.2}%, fb_error={:.2}",
               observation.frame_index, observation.track_count, observation.inlier_ratio * 100.0, observation.forward_backward_error);

        self.metrics
            .update(observation, self.config.metrics_window.max(1));

        let status = self.evaluate(observation);
        match status {
            VoStatus::Lost => {
                self.consecutive_losses += 1;
                warn!(target: "arbit_core::vo", "VO tracking lost (consecutive: {})", self.consecutive_losses);
            }
            VoStatus::Degrading => {
                warn!(target: "arbit_core::vo", "VO tracking degrading");
            }
            VoStatus::Healthy => {
                if self.consecutive_losses > 0 {
                    info!(target: "arbit_core::vo", "VO tracking recovered after {} lost frames", self.consecutive_losses);
                    self.consecutive_losses = 0;
                }
            }
        }

        self.log.push(FrameLogEntry {
            frame_index: observation.frame_index,
            timestamp_seconds: observation.timestamp_seconds,
            pose: observation.pose,
            track_count: observation.track_count,
            inlier_ratio: observation.inlier_ratio,
            forward_backward_error: observation.forward_backward_error,
        });

        status
    }

    fn evaluate(&self, observation: FrameObservation) -> VoStatus {
        if observation.inlier_ratio < self.config.min_inlier_ratio
            || observation.forward_backward_error > self.config.max_forward_backward_error * 1.5
        {
            return VoStatus::Lost;
        }

        if observation.inlier_ratio
            < self.config.min_inlier_ratio + (self.config.min_inlier_ratio * 0.5)
            || observation.forward_backward_error > self.config.max_forward_backward_error * 0.5
            || self.metrics.average_inlier_ratio < self.config.min_inlier_ratio * 1.25
            || self.metrics.average_forward_backward_error
                > self.config.max_forward_backward_error * 0.75
        {
            return VoStatus::Degrading;
        }

        VoStatus::Healthy
    }

    pub fn metrics(&self) -> &VoMetrics {
        &self.metrics
    }

    pub fn log(&self) -> &ReplayLog {
        &self.log
    }

    pub fn reset(&mut self) {
        self.metrics = VoMetrics::default();
        self.log = ReplayLog::default();
        self.consecutive_losses = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::se3::SE3;
    use nalgebra::{UnitQuaternion, Vector3};

    fn make_pose(t: Vector3<f64>) -> TransformSE3 {
        SE3::from_parts(
            crate::math::so3::SO3::from_unit_quaternion(UnitQuaternion::identity()),
            t,
        )
        .to_isometry()
    }

    #[test]
    fn vo_loop_accumulates_metrics() {
        let mut vo = VoLoop::new(VoLoopConfig {
            metrics_window: 5,
            min_inlier_ratio: 0.2,
            max_forward_backward_error: 5.0,
        });
        for idx in 0..10 {
            let obs = FrameObservation {
                frame_index: idx,
                timestamp_seconds: idx as f64 * 0.016,
                pose: make_pose(Vector3::new(idx as f64 * 0.01, 0.0, 1.0)),
                track_count: 120 - idx as usize,
                inlier_ratio: 0.9 - (idx as f32) * 0.02,
                forward_backward_error: 0.5 + (idx as f32) * 0.05,
            };
            vo.process(obs);
        }
        assert_eq!(vo.log().len(), 10);
        assert!(vo.metrics().average_inlier_ratio < 0.9);
        assert!(vo.metrics().track_count_mean > 100.0);
    }

    #[test]
    fn vo_loop_detects_degradation() {
        let mut vo = VoLoop::new(VoLoopConfig::default());
        let healthy = FrameObservation {
            frame_index: 0,
            timestamp_seconds: 0.0,
            pose: TransformSE3::identity(),
            track_count: 150,
            inlier_ratio: 0.8,
            forward_backward_error: 0.5,
        };
        assert!(matches!(vo.process(healthy), VoStatus::Healthy));

        let degrading = FrameObservation {
            frame_index: 1,
            timestamp_seconds: 0.016,
            pose: TransformSE3::identity(),
            track_count: 50,
            inlier_ratio: 0.35,
            forward_backward_error: 2.0,
        };
        assert!(matches!(vo.process(degrading), VoStatus::Degrading));

        let lost = FrameObservation {
            frame_index: 2,
            timestamp_seconds: 0.032,
            pose: TransformSE3::identity(),
            track_count: 30,
            inlier_ratio: 0.1,
            forward_backward_error: 10.0,
        };
        assert!(matches!(vo.process(lost), VoStatus::Lost));
    }
}
