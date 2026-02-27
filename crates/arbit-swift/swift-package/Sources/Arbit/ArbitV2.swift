import Foundation
import ArbitFFI

@_silgen_name("arbit_v2_context_create")
private func ffi_v2_context_create() -> OpaquePointer?

@_silgen_name("arbit_v2_context_destroy")
private func ffi_v2_context_destroy(_ handle: OpaquePointer?)

@_silgen_name("arbit_v2_ingest_frame")
private func ffi_v2_ingest_frame(
    _ handle: OpaquePointer?,
    _ frame: UnsafePointer<ArbitFFI.ArbitV2CameraFrame>?
) -> Bool

@_silgen_name("arbit_v2_ingest_imu")
private func ffi_v2_ingest_imu(
    _ handle: OpaquePointer?,
    _ imu: UnsafePointer<ArbitFFI.ArbitV2ImuSample>?
) -> Bool

@_silgen_name("arbit_v2_get_snapshot")
private func ffi_v2_get_snapshot(
    _ handle: OpaquePointer?,
    _ outSnapshot: UnsafeMutablePointer<ArbitFFI.ArbitV2Snapshot>?
) -> Bool

@_silgen_name("arbit_v2_create_anchor")
private func ffi_v2_create_anchor(
    _ handle: OpaquePointer?,
    _ pose: UnsafePointer<Double>?,
    _ hasHint: Bool,
    _ keyframeHint: UInt64,
    _ outAnchorId: UnsafeMutablePointer<UInt64>?
) -> Bool

@_silgen_name("arbit_v2_query_anchor")
private func ffi_v2_query_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64,
    _ outAnchor: UnsafeMutablePointer<ArbitFFI.ArbitV2Anchor>?
) -> Bool

@_silgen_name("arbit_v2_reset_session")
private func ffi_v2_reset_session(_ handle: OpaquePointer?) -> Bool

public struct V2CameraFrame {
    public var timestamp: Double
    public var intrinsics: CameraIntrinsics
    public var pixelFormat: CameraPixelFormat
    public var bytesPerRow: Int
    public var data: Data

    public init(
        timestamp: Double,
        intrinsics: CameraIntrinsics,
        pixelFormat: CameraPixelFormat,
        bytesPerRow: Int,
        data: Data
    ) {
        self.timestamp = timestamp
        self.intrinsics = intrinsics
        self.pixelFormat = pixelFormat
        self.bytesPerRow = bytesPerRow
        self.data = data
    }

    fileprivate func withFFI<Result>(
        _ body: (ArbitFFI.ArbitV2CameraFrame) -> Result
    ) -> Result {
        intrinsics.withFFI { ffiIntrinsics in
            data.withUnsafeBytes { rawBuffer in
                let ptr = rawBuffer.bindMemory(to: UInt8.self).baseAddress
                let ffiFrame = ArbitFFI.ArbitV2CameraFrame(
                    timestamp_seconds: timestamp,
                    intrinsics: ArbitFFI.ArbitV2CameraIntrinsics(
                        fx: ffiIntrinsics.fx,
                        fy: ffiIntrinsics.fy,
                        cx: ffiIntrinsics.cx,
                        cy: ffiIntrinsics.cy,
                        skew: ffiIntrinsics.skew,
                        width: ffiIntrinsics.width,
                        height: ffiIntrinsics.height,
                        distortion_len: ffiIntrinsics.distortion_len,
                        distortion: ffiIntrinsics.distortion
                    ),
                    pixel_format: ArbitFFI.ArbitV2PixelFormat(rawValue: pixelFormat.rawValue),
                    bytes_per_row: bytesPerRow,
                    data: ptr,
                    data_len: data.count
                )
                return body(ffiFrame)
            }
        }
    }
}

public struct V2ImuSample {
    public var timestamp: Double
    public var accel: SIMD3<Double>
    public var gyro: SIMD3<Double>

    public init(timestamp: Double, accel: SIMD3<Double>, gyro: SIMD3<Double>) {
        self.timestamp = timestamp
        self.accel = accel
        self.gyro = gyro
    }

    fileprivate var ffiValue: ArbitFFI.ArbitV2ImuSample {
        ArbitFFI.ArbitV2ImuSample(
            timestamp_seconds: timestamp,
            accel_x: accel.x,
            accel_y: accel.y,
            accel_z: accel.z,
            gyro_x: gyro.x,
            gyro_y: gyro.y,
            gyro_z: gyro.z
        )
    }
}

public struct V2TrackingSnapshot {
    public var state: UInt32
    public var frameId: UInt64
    public var trackCount: UInt32
    public var inlierCount: UInt32
    public var poseWC: [Double]

    init(ffi: ArbitFFI.ArbitV2TrackingSnapshot) {
        state = ffi.state
        frameId = ffi.frame_id
        trackCount = ffi.track_count
        inlierCount = ffi.inlier_count
        poseWC = Array(ffi.pose_wc)
    }
}

public struct V2BackendSnapshot {
    public var keyframeCount: UInt64
    public var loopClosureEvents: UInt64
    public var relocalizationReady: Bool

    init(ffi: ArbitFFI.ArbitV2BackendSnapshot) {
        keyframeCount = ffi.keyframe_count
        loopClosureEvents = ffi.loop_closure_events
        relocalizationReady = ffi.relocalization_ready
    }
}

public struct V2MapSnapshot {
    public var landmarkCount: UInt64
    public var anchorCount: UInt64

    init(ffi: ArbitFFI.ArbitV2MapSnapshot) {
        landmarkCount = ffi.landmark_count
        anchorCount = ffi.anchor_count
    }
}

public struct V2RuntimeMetricsSnapshot {
    public var frameQueueDepth: Int
    public var imuQueueDepth: Int
    public var keyframeQueueDepth: Int
    public var backendQueueDepth: Int
    public var droppedFrames: UInt64
    public var frontendMsMedian: Double
    public var frontendMsP95: Double
    public var endToEndMsP95: Double

    init(ffi: ArbitFFI.ArbitV2RuntimeMetricsSnapshot) {
        frameQueueDepth = ffi.frame_queue_depth
        imuQueueDepth = ffi.imu_queue_depth
        keyframeQueueDepth = ffi.keyframe_queue_depth
        backendQueueDepth = ffi.backend_queue_depth
        droppedFrames = ffi.dropped_frames
        frontendMsMedian = ffi.frontend_ms_median
        frontendMsP95 = ffi.frontend_ms_p95
        endToEndMsP95 = ffi.end_to_end_ms_p95
    }
}

public struct V2Snapshot {
    public var timestampSeconds: Double
    public var tracking: V2TrackingSnapshot
    public var backend: V2BackendSnapshot
    public var map: V2MapSnapshot
    public var metrics: V2RuntimeMetricsSnapshot

    init(ffi: ArbitFFI.ArbitV2Snapshot) {
        timestampSeconds = ffi.timestamp_seconds
        tracking = V2TrackingSnapshot(ffi: ffi.tracking)
        backend = V2BackendSnapshot(ffi: ffi.backend)
        map = V2MapSnapshot(ffi: ffi.map)
        metrics = V2RuntimeMetricsSnapshot(ffi: ffi.metrics)
    }
}

public struct V2Anchor {
    public var anchorId: UInt64
    public var poseWC: [Double]
    public var createdFromKeyframe: UInt64?
    public var lastObservedFrame: UInt64

    init(ffi: ArbitFFI.ArbitV2Anchor) {
        anchorId = ffi.anchor_id
        poseWC = Array(ffi.pose_wc)
        createdFromKeyframe = ffi.has_keyframe ? ffi.created_from_keyframe : nil
        lastObservedFrame = ffi.last_observed_frame
    }
}

public final class ArbitEngineV2: @unchecked Sendable {
    private var handle: OpaquePointer

    public init() throws {
        guard let ptr = ffi_v2_context_create() else {
            throw ArbitCaptureError.allocationFailed
        }
        handle = ptr
    }

    deinit {
        ffi_v2_context_destroy(handle)
    }

    @discardableResult
    public func ingest(_ frame: V2CameraFrame) -> Bool {
        frame.withFFI { ffiFrame in
            var mutableFrame = ffiFrame
            return withUnsafePointer(to: &mutableFrame) { ptr in
                ffi_v2_ingest_frame(handle, ptr)
            }
        }
    }

    @discardableResult
    public func ingest(imu sample: V2ImuSample) -> Bool {
        var ffiSample = sample.ffiValue
        return withUnsafePointer(to: &ffiSample) { ptr in
            ffi_v2_ingest_imu(handle, ptr)
        }
    }

    public func snapshot() -> V2Snapshot? {
        var ffiSnapshot = ArbitFFI.ArbitV2Snapshot()
        guard ffi_v2_get_snapshot(handle, &ffiSnapshot) else {
            return nil
        }
        return V2Snapshot(ffi: ffiSnapshot)
    }

    public func createAnchor(
        poseWC: [Double]? = nil,
        keyframeHint: UInt64? = nil
    ) -> UInt64? {
        var out: UInt64 = 0
        let hasHint = keyframeHint != nil
        let hint = keyframeHint ?? 0

        if let poseWC {
            guard poseWC.count == 16 else { return nil }
            let ok = poseWC.withUnsafeBufferPointer { buffer in
                ffi_v2_create_anchor(handle, buffer.baseAddress, hasHint, hint, &out)
            }
            return ok ? out : nil
        }

        let ok = ffi_v2_create_anchor(handle, nil, hasHint, hint, &out)
        return ok ? out : nil
    }

    public func queryAnchor(_ anchorId: UInt64) -> V2Anchor? {
        var ffiAnchor = ArbitFFI.ArbitV2Anchor()
        guard ffi_v2_query_anchor(handle, anchorId, &ffiAnchor) else {
            return nil
        }
        return V2Anchor(ffi: ffiAnchor)
    }

    @discardableResult
    public func resetSession() -> Bool {
        ffi_v2_reset_session(handle)
    }
}
