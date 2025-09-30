#ifndef ARBIT_SWIFT_H
#define ARBIT_SWIFT_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Mirrors arbit_ffi::ArbitPixelFormat
typedef enum {
    ARBIT_PIXEL_FORMAT_BGRA8 = 0,
    ARBIT_PIXEL_FORMAT_RGBA8 = 1,
    ARBIT_PIXEL_FORMAT_NV12 = 2,
    ARBIT_PIXEL_FORMAT_YV12 = 3,
    ARBIT_PIXEL_FORMAT_DEPTH16 = 4,
} ArbitPixelFormat;

// Mirrors arbit_ffi::ArbitCameraIntrinsics
typedef struct {
    double fx;
    double fy;
    double cx;
    double cy;
    double skew;
    uint32_t width;
    uint32_t height;
    size_t distortion_len;
    const double *distortion;
} ArbitCameraIntrinsics;

// Mirrors arbit_ffi::ArbitCameraFrame
typedef struct {
    double timestamp_seconds;
    ArbitCameraIntrinsics intrinsics;
    ArbitPixelFormat pixel_format;
    size_t bytes_per_row;
    const uint8_t *data;
    size_t data_len;
} ArbitCameraFrame;

// Mirrors arbit_ffi::ArbitFrameTimestamps
typedef struct {
    double capture_seconds;
    double pipeline_seconds;
    double latency_seconds;
} ArbitFrameTimestamps;

// Mirrors arbit_ffi::ArbitCameraSample
typedef struct {
    ArbitFrameTimestamps timestamps;
    ArbitCameraIntrinsics intrinsics;
    ArbitPixelFormat pixel_format;
    size_t bytes_per_row;
} ArbitCameraSample;

// Mirrors arbit_ffi::ArbitTransform
typedef struct {
    double elements[16];
} ArbitTransform;

// Mirrors arbit_ffi::ArbitPyramidLevelView
typedef struct {
    uint32_t octave;
    float scale;
    uint32_t width;
    uint32_t height;
    size_t bytes_per_row;
    const uint8_t *pixels;
    size_t pixels_len;
} ArbitPyramidLevelView;

// Mirrors arbit_ffi::ArbitTrackStatus
typedef enum {
    ARBIT_TRACK_STATUS_CONVERGED = 0,
    ARBIT_TRACK_STATUS_DIVERGED = 1,
    ARBIT_TRACK_STATUS_OUT_OF_BOUNDS = 2,
} ArbitTrackStatus;

// Mirrors arbit_ffi::ArbitTrackedPoint
typedef struct {
    float initial_x;
    float initial_y;
    float refined_x;
    float refined_y;
    float residual;
    uint32_t iterations;
    ArbitTrackStatus status;
} ArbitTrackedPoint;

// Mirrors arbit_ffi::ArbitTwoViewSummary
typedef struct {
    uint32_t inliers;
    double average_error;
    double rotation[9];
    double translation[3];
} ArbitTwoViewSummary;

// Mirrors arbit_ffi::ArbitPoseSample
typedef struct {
    double x;
    double y;
    double z;
} ArbitPoseSample;

// Mirrors arbit_ffi::ArbitAccelerometerSample
typedef struct {
    double ax;
    double ay;
    double az;
    double dt_seconds;
} ArbitAccelerometerSample;

// Mirrors arbit_ffi::ArbitGravityEstimate
typedef struct {
    double down[3];
    uint32_t samples;
} ArbitGravityEstimate;

// Opaque handle around the core timestamp policy context
typedef struct ArbitCaptureContextHandle ArbitCaptureContextHandle;

ArbitCaptureContextHandle *arbit_capture_context_new(void);
void arbit_capture_context_free(ArbitCaptureContextHandle *handle);

bool arbit_ingest_camera_frame(
    ArbitCaptureContextHandle *handle,
    const ArbitCameraFrame *frame,
    ArbitCameraSample *out_sample
);

size_t arbit_capture_context_pyramid_levels(
    ArbitCaptureContextHandle *handle,
    ArbitPyramidLevelView *out_levels,
    size_t max_levels
);

size_t arbit_capture_context_tracked_points(
    ArbitCaptureContextHandle *handle,
    ArbitTrackedPoint *out_points,
    size_t max_points
);

bool arbit_capture_context_two_view(
    ArbitCaptureContextHandle *handle,
    ArbitTwoViewSummary *out_summary
);

size_t arbit_capture_context_trajectory(
    ArbitCaptureContextHandle *handle,
    ArbitPoseSample *out_points,
    size_t max_points
);

bool arbit_ingest_accelerometer_sample(
    ArbitCaptureContextHandle *handle,
    ArbitAccelerometerSample sample
);

bool arbit_capture_context_gravity(
    ArbitCaptureContextHandle *handle,
    ArbitGravityEstimate *out_estimate
);

bool arbit_world_from_scenekit(
    const float *transform_columns,
    ArbitTransform *out_transform
);

const char *arbit_swift_version(void);

#ifdef __cplusplus
}
#endif

#endif // ARBIT_SWIFT_H
