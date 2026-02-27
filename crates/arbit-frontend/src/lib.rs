use arbit_core::contracts::{
    FramePacket, FrontendOutput, FrontendProcessor, ImuPacket, TrackingState, identity_pose,
};

#[derive(Debug, Clone)]
pub struct CpuFrontendConfig {
    pub keyframe_interval: u64,
    pub min_tracks_for_keyframe: u32,
}

impl Default for CpuFrontendConfig {
    fn default() -> Self {
        Self {
            keyframe_interval: 10,
            min_tracks_for_keyframe: 120,
        }
    }
}

#[derive(Debug)]
pub struct CpuFrontend {
    config: CpuFrontendConfig,
    last_pose_wc: [f64; 16],
    last_frame_id: u64,
}

impl CpuFrontend {
    pub fn new(config: CpuFrontendConfig) -> Self {
        Self {
            config,
            last_pose_wc: identity_pose(),
            last_frame_id: 0,
        }
    }

    fn estimate_track_count(&self, frame: &FramePacket) -> u32 {
        let spatial_budget = (frame
            .intrinsics
            .width
            .saturating_mul(frame.intrinsics.height)
            / 4_096)
            .clamp(80, 1_200);
        spatial_budget as u32
    }

    fn integrate_imu_step(&self, imu_window: &[ImuPacket]) -> f64 {
        if imu_window.is_empty() {
            return 0.0;
        }

        let mut gyro_norm_sum = 0.0;
        for imu in imu_window {
            let [gx, gy, gz] = imu.gyro_rps;
            gyro_norm_sum += (gx * gx + gy * gy + gz * gz).sqrt();
        }

        (gyro_norm_sum / imu_window.len() as f64).min(3.0)
    }
}

impl FrontendProcessor for CpuFrontend {
    fn process_frame(&mut self, frame: &FramePacket, imu_window: &[ImuPacket]) -> FrontendOutput {
        let track_count = self.estimate_track_count(frame);
        let inlier_count = (track_count as f64 * 0.72) as u32;
        let keyframe_candidate = frame.frame_id % self.config.keyframe_interval == 0
            || track_count < self.config.min_tracks_for_keyframe;

        let mut pose_wc = self.last_pose_wc;
        let imu_delta = self.integrate_imu_step(imu_window);
        pose_wc[11] -= 0.005 + imu_delta * 0.0005;

        let tracking_state = if track_count > 0 {
            TrackingState::Tracking
        } else {
            TrackingState::Relocalizing
        };

        self.last_pose_wc = pose_wc;
        self.last_frame_id = frame.frame_id;

        FrontendOutput {
            frame_id: frame.frame_id,
            timestamp_seconds: frame.timestamp_seconds,
            pose_wc,
            track_count,
            inlier_count,
            tracking_state,
            keyframe_candidate,
        }
    }

    fn reset(&mut self) {
        self.last_pose_wc = identity_pose();
        self.last_frame_id = 0;
    }
}

impl Default for CpuFrontend {
    fn default() -> Self {
        Self::new(CpuFrontendConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arbit_core::{
        contracts::{FrontendProcessor, PixelFormat},
        math::{CameraIntrinsics, DistortionModel},
    };

    use super::*;

    fn sample_frame(frame_id: u64) -> FramePacket {
        FramePacket {
            frame_id,
            timestamp_seconds: frame_id as f64 * 0.033,
            intrinsics: CameraIntrinsics::new(
                600.0,
                600.0,
                320.0,
                240.0,
                0.0,
                640,
                480,
                DistortionModel::None,
            ),
            pixel_format: PixelFormat::Bgra8,
            bytes_per_row: 640 * 4,
            data: Arc::from(vec![0_u8; 640 * 480 * 4]),
        }
    }

    #[test]
    fn frontend_emits_keyframe_on_interval() {
        let mut frontend = CpuFrontend::default();
        let out = frontend.process_frame(&sample_frame(10), &[]);
        assert!(out.keyframe_candidate);
        assert_eq!(out.tracking_state, TrackingState::Tracking);
        assert!(out.track_count >= 80);
    }
}
