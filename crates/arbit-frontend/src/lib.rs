use arbit_core::contracts::{
    FramePacket, FrontendOutput, FrontendProcessor, ImuPacket, TrackingState, identity_pose,
};
use arbit_native::{BinaryFeature, OpenCvKernelAdapter};

#[derive(Debug, Clone)]
pub struct CpuFrontendConfig {
    pub keyframe_interval: u64,
    pub min_tracks_for_keyframe: u32,
    pub max_native_features: usize,
}

impl Default for CpuFrontendConfig {
    fn default() -> Self {
        Self {
            keyframe_interval: 10,
            min_tracks_for_keyframe: 120,
            max_native_features: 384,
        }
    }
}

#[derive(Debug)]
pub struct CpuFrontend {
    config: CpuFrontendConfig,
    last_pose_wc: [f64; 16],
    last_frame_id: u64,
    previous_features: Vec<BinaryFeature>,
    native_adapter: OpenCvKernelAdapter,
}

impl CpuFrontend {
    pub fn new(config: CpuFrontendConfig) -> Self {
        Self {
            config,
            last_pose_wc: identity_pose(),
            last_frame_id: 0,
            previous_features: Vec::new(),
            native_adapter: OpenCvKernelAdapter::default(),
        }
    }

    fn estimate_track_count_fallback(&self, frame: &FramePacket) -> u32 {
        let spatial_budget = (frame
            .intrinsics
            .width
            .saturating_mul(frame.intrinsics.height)
            / 4_096)
            .clamp(80, 1_200);
        spatial_budget
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

    fn detect_features(&self, frame: &FramePacket) -> (bool, Vec<BinaryFeature>) {
        let native = self.native_adapter.detect_and_describe_bgra(
            frame.intrinsics.width,
            frame.intrinsics.height,
            frame.bytes_per_row,
            &frame.data,
            self.config.max_native_features,
        );

        match native {
            Ok(features) => (true, features),
            Err(_) => (false, Vec::new()),
        }
    }

    fn count_inliers(&self, current_features: &[BinaryFeature], native_active: bool) -> u32 {
        if !native_active || current_features.is_empty() {
            return 0;
        }
        if self.previous_features.is_empty() {
            return (current_features.len() as u32).min(96);
        }

        match self.native_adapter.match_features(
            &self.previous_features,
            current_features,
            72,
            true,
        ) {
            Ok(matches) => {
                if matches.is_empty() {
                    (current_features.len() as u32 / 4).min(64)
                } else {
                    matches.len() as u32
                }
            }
            Err(_) => 0,
        }
    }
}

impl FrontendProcessor for CpuFrontend {
    fn process_frame(&mut self, frame: &FramePacket, imu_window: &[ImuPacket]) -> FrontendOutput {
        let (native_active, current_features) = self.detect_features(frame);

        let track_count = if native_active {
            current_features.len() as u32
        } else {
            self.estimate_track_count_fallback(frame)
        };

        let inlier_count = if native_active {
            self.count_inliers(&current_features, true)
        } else {
            (track_count as f64 * 0.72) as u32
        };

        let keyframe_candidate = frame.frame_id % self.config.keyframe_interval == 0
            || inlier_count < self.config.min_tracks_for_keyframe / 2;

        let tracking_state = if frame.frame_id <= 2 && track_count > 0 {
            TrackingState::Initializing
        } else if track_count == 0 {
            TrackingState::Lost
        } else if inlier_count < (track_count / 8).max(8) {
            TrackingState::Relocalizing
        } else {
            TrackingState::Tracking
        };

        let mut pose_wc = self.last_pose_wc;
        let imu_delta = self.integrate_imu_step(imu_window);
        if matches!(
            tracking_state,
            TrackingState::Tracking | TrackingState::Initializing
        ) {
            let tracking_confidence =
                (inlier_count as f64 / track_count.max(1) as f64).clamp(0.05, 1.0);
            pose_wc[11] -= 0.0025 + (1.0 - tracking_confidence) * 0.0015 + imu_delta * 0.0004;
        }

        self.last_pose_wc = pose_wc;
        self.last_frame_id = frame.frame_id;
        self.previous_features = current_features;

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
        self.previous_features.clear();
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

    fn textured_bgra(width: usize, height: usize) -> Arc<[u8]> {
        let mut data = vec![0_u8; width * height * 4];
        for y in 0..height {
            for x in 0..width {
                let checker = ((x / 8) + (y / 8)) % 2;
                let luma = if checker == 0 { 32_u8 } else { 224_u8 };
                let idx = (y * width + x) * 4;
                data[idx] = luma;
                data[idx + 1] = luma;
                data[idx + 2] = luma;
                data[idx + 3] = 255;
            }
        }
        data.into()
    }

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
            data: textured_bgra(640, 480),
        }
    }

    #[test]
    fn frontend_emits_keyframe_on_interval() {
        let mut frontend = CpuFrontend::default();
        let out = frontend.process_frame(&sample_frame(10), &[]);
        assert!(out.keyframe_candidate);
        assert!(out.track_count > 0);
        assert!(matches!(
            out.tracking_state,
            TrackingState::Initializing | TrackingState::Tracking | TrackingState::Relocalizing
        ));
    }
}
