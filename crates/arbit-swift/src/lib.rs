//! Swift-focused packaging for the Arbit FFI layer.
//!
//! This crate compiles the lower-level `arbit-ffi` symbols into a single static
//! library that can be wrapped inside an `.xcframework` or linked directly from
//! Xcode. All exported C-compatible data structures and functions are
//! re-exported here so that consumers only need to depend on one compiled
//! artifact when integrating with Swift. Platform-specific helpers (like the
//! SceneKit adapter) live here so that the pure C layer stays neutral.

#![deny(unused_must_use)]

pub use arbit_ffi::{
    ArbitAccelerometerSample, ArbitCameraFrame, ArbitCameraIntrinsics, ArbitCameraSample,
    ArbitCaptureContextHandle, ArbitFrameTimestamps, ArbitGravityEstimate, ArbitImuSample,
    ArbitMotionState, ArbitPixelFormat, ArbitPoseSample, ArbitPyramidLevelView,
    ArbitRelocalizationSummary, ArbitTrackStatus, ArbitTrackedPoint, ArbitTransform,
    ArbitTwoViewSummary, arbit_capture_context_create_anchor, arbit_capture_context_free,
    arbit_capture_context_gravity, arbit_capture_context_last_relocalization,
    arbit_capture_context_list_anchors, arbit_capture_context_load_map,
    arbit_capture_context_map_stats, arbit_capture_context_new,
    arbit_capture_context_pyramid_levels, arbit_capture_context_resolve_anchor,
    arbit_capture_context_save_map, arbit_capture_context_tracked_points,
    arbit_capture_context_trajectory, arbit_capture_context_two_view,
    arbit_capture_context_update_anchor, arbit_finish_imu_preintegration,
    arbit_ingest_accelerometer_sample, arbit_ingest_camera_frame, arbit_ingest_imu_sample,
    arbit_last_imu_rotation_prior, arbit_last_motion_state,
};

use arbit_core::adapters::world_from_scenekit;
use arbit_core::math::se3::TransformSE3;
use nalgebra::{Matrix4, Translation3, UnitQuaternion, Vector3};

/// Returns the semantic version of the compiled Arbit library as a
/// null-terminated C string.
///
/// The pointer remains valid for the lifetime of the process and must not be
/// freed by the caller.
#[unsafe(no_mangle)]
pub extern "C" fn arbit_swift_version() -> *const core::ffi::c_char {
    static VERSION: &[u8] = concat!(env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr().cast()
}

/// Converts a column-major SceneKit transform (f32) into a right-handed,
/// world-aligned transform expressed in f64 homogeneous coordinates.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn arbit_world_from_scenekit(
    transform_columns: *const f32,
    out_transform: *mut ArbitTransform,
) -> bool {
    if transform_columns.is_null() || out_transform.is_null() {
        return false;
    }

    let input = unsafe { std::slice::from_raw_parts(transform_columns, 16) };
    let mut raw = [0.0f64; 16];
    for (dst, src) in raw.iter_mut().zip(input.iter()) {
        *dst = f64::from(*src);
    }

    let matrix = Matrix4::from_column_slice(&raw);
    let rotation_matrix = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    let translation_vec = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
    let rotation = UnitQuaternion::from_matrix(&rotation_matrix);
    let translation = Translation3::from(translation_vec);
    let iso = TransformSE3::from_parts(translation, rotation);

    let world = world_from_scenekit(&iso);
    let homogeneous = world.to_homogeneous();

    let mut output = ArbitTransform::default();
    for (idx, value) in homogeneous.iter().enumerate() {
        output.elements[idx] = *value;
    }

    unsafe {
        *out_transform = output;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr;

    #[test]
    fn exported_symbols_are_linked() {
        let handle = arbit_capture_context_new();
        assert!(!handle.is_null());

        let width = 8u32;
        let height = 8u32;
        let bytes_per_row = (width as usize) * 4;
        let pixels = vec![0u8; bytes_per_row * (height as usize)];

        let frame = ArbitCameraFrame {
            timestamp_seconds: 0.0,
            intrinsics: ArbitCameraIntrinsics {
                fx: 500.0,
                fy: 500.0,
                cx: (width as f64) / 2.0,
                cy: (height as f64) / 2.0,
                skew: 0.0,
                width,
                height,
                distortion_len: 0,
                distortion: ptr::null(),
            },
            pixel_format: ArbitPixelFormat::Bgra8,
            bytes_per_row,
            data: pixels.as_ptr(),
            data_len: pixels.len(),
        };

        let mut sample = ArbitCameraSample {
            timestamps: ArbitFrameTimestamps {
                capture_seconds: 0.0,
                pipeline_seconds: 0.0,
                latency_seconds: 0.0,
            },
            intrinsics: ArbitCameraIntrinsics::default(),
            pixel_format: ArbitPixelFormat::Bgra8,
            bytes_per_row: 0,
        };

        unsafe {
            assert!(arbit_ingest_camera_frame(handle, &frame, &mut sample));
            arbit_capture_context_free(handle);
        }
    }

    #[test]
    fn scenekit_identity_round_trip() {
        let identity: [f32; 16] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut transform = ArbitTransform::default();
        unsafe {
            assert!(arbit_world_from_scenekit(
                identity.as_ptr(),
                &mut transform as *mut _
            ));
        }
        for (idx, value) in identity.iter().enumerate() {
            assert!((transform.elements[idx] - f64::from(*value)).abs() < 1e-12);
        }
    }
}
