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

// Opaque handle around the core timestamp policy context
typedef struct ArbitCaptureContextHandle ArbitCaptureContextHandle;

ArbitCaptureContextHandle *arbit_capture_context_new(void);
void arbit_capture_context_free(ArbitCaptureContextHandle *handle);

bool arbit_ingest_camera_frame(
    ArbitCaptureContextHandle *handle,
    const ArbitCameraFrame *frame,
    ArbitCameraSample *out_sample
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
