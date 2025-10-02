# arbit-arkit-example (Milestone 1 AVFoundation Demo)

This sample app implements the original ARKit-based prototype using plain AVFoundation while wiring frames through the Arbit FFI bridge. It demonstrates : clean camera timestamps, world-aligned orientation, and a HUD that visualises intrinsics and latency.

## High-Level Architecture

The SwiftUI app is split across lightweight components so the capture path mirrors the Rust core contracts:

- **CameraCaptureManager** (`CapturePipeline.swift`) owns `AVCaptureSession` setup and feeds frames into `arbit-swift-lib`. It extracts per-frame intrinsics, wraps buffers as `CameraFrame`, and emits processed `CameraSample` structs surfaced to SwiftUI.
- **arbit-swift-lib** (`crates/arbit-swift/swift-package/`) exposes the Rust FFI (`arbit_ffi`) as a Swift Package, providing safe wrappers such as `ArbitCaptureContext`, `CameraFrame`, and `CameraSample` with monotonic timestamp handling in Rust.
- **DeviceOrientationProvider** (`DeviceOrientationProvider.swift`) streams CoreMotion quaternions using the `xArbitraryCorrectedZVertical` reference frame so the demo has a gravity-aligned world basis without ARKit.
- **AxesSceneView** (`AxesSceneView.swift`) hosts a SceneKit scene that renders XYZ cylinders and applies the device-to-world quaternion from CoreMotion.
- **CameraPreviewView** (`CameraPreviewView.swift`) bridges `AVCaptureVideoPreviewLayer` into SwiftUI.
- **ContentView** (`ContentView.swift`) composes the HUD, displays Arbit-derived metrics (capture/pipeline timestamps, latency, intrinsics, FPS), and embeds both the camera preview and axes overlay.

```
AVCaptureVideoDataOutput ─▶ CameraCaptureManager ─▶ CameraFrame ─▶ arbit-swift-lib (Rust FFI)
        │                                                         │
        └─▶ CameraPreviewView (Preview layer)                     └─▶ CameraSample (timestamps, K)
                                                                    │
CoreMotion ─▶ DeviceOrientationProvider ────────────────────────────┘
```

## Data Flow Details

1. **Session bootstrap** – On `ContentView` appearance, `CameraCaptureManager.start()` requests permission (if needed), configures a 720p back-camera session, enables intrinsics delivery, and begins streaming frames on a dedicated queue.
2. **Frame ingestion** – Each `CMSampleBuffer` locks the pixel buffer, copies BGRA bytes into a `Data` payload, and builds a `CameraFrame` with the intrinsic matrix parsed from the attachment (falling back to a centred pinhole model).
3. **FFI bridge** – `ArbitCaptureContext.ingest(_:)` forwards the frame to the Rust timestamp policy (`TimestampPolicy<SystemClock>`). It returns a `CameraSample` containing monotonic capture/pipeline times, calculated latency, and normalised intrinsics.
4. **UI updates** – The processed sample is published on the main thread. `ContentView` derives live metrics and estimates FPS from successive pipeline timestamps, while SceneKit applies the CoreMotion quaternion to the axis gizmo.
5. **Orientation sync** – CoreMotion delivers device motion in a gravity-stabilised reference frame. The quaternion is inverted so the axes display stays world-aligned (Y-up) as the phone moves.

## Running the Demo

1. Open `examples/arbit-arkit-example/arbit-arkit-example.xcodeproj` in Xcode 16 or later.
2. Ensure the `arbit-arkit-example` target links the local Swift package `arbit-swift-lib` located at `crates/arbit-swift/swift-package` (already configured in the project).
3. Select an iOS device (simulator cannot provide camera frames) and run. Grant camera access on first launch.
4. Observe:
   - Live BGRA preview from AVFoundation.
   - HUD metrics showing capture/pipeline timestamps, Arbit-derived latency, focal lengths, principal point, and resolution.
   - A SceneKit axes widget that rotates with the phone following gravity, proving the world frame alignment.

## Notes & Next Steps

- The capture manager copies frame bytes into `Data` before handing them to the FFI for safety; if throughput becomes an issue we can expose a zero-copy API that accepts `CVPixelBuffer` base addresses directly.
- Intrinsics parsing currently assumes the matrix is provided as 32-bit floats (`kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix`). When the attachment is absent we fallback to a naive centred pinhole—it may be worth exposing warnings in the HUD.
- Milestone 1 stops at timestamp sanity and orientation; future milestones can hook into pyramid generation, IMU fusion, and VO once those Rust crates solidify.
