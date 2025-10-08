import Foundation
import ArbitFFI
#if canImport(CoreVideo)
import CoreVideo
#endif
#if canImport(simd)
import simd
#endif
#if canImport(ARKit)
import ARKit
#endif

// MARK: - Public IMU Types

/// IMU sample containing 6DOF sensor data (accelerometer + gyroscope)
public struct ImuSample {
    public let timestampSeconds: Double
    public let accelX: Double
    public let accelY: Double
    public let accelZ: Double
    public let gyroX: Double
    public let gyroY: Double
    public let gyroZ: Double
    
    public init(
        timestampSeconds: Double,
        accelX: Double, accelY: Double, accelZ: Double,
        gyroX: Double, gyroY: Double, gyroZ: Double
    ) {
        self.timestampSeconds = timestampSeconds
        self.accelX = accelX
        self.accelY = accelY
        self.accelZ = accelZ
        self.gyroX = gyroX
        self.gyroY = gyroY
        self.gyroZ = gyroZ
    }
    
    /// Convert to FFI representation
    fileprivate func toFFI() -> ArbitFFI.ArbitImuSample {
        ArbitFFI.ArbitImuSample(
            timestamp_seconds: timestampSeconds,
            accel_x: accelX,
            accel_y: accelY,
            accel_z: accelZ,
            gyro_x: gyroX,
            gyro_y: gyroY,
            gyro_z: gyroZ
        )
    }
}

/// Motion state classification
public enum MotionState: String {
    case stationary = "Stationary"
    case slowMotion = "SlowMotion"
    case fastMotion = "FastMotion"
    
    fileprivate init?(ffiState: ArbitFFI.ArbitMotionState) {
        switch ffiState.state {
        case 0: self = .stationary
        case 1: self = .slowMotion
        case 2: self = .fastMotion
        default: return nil
        }
    }
}

@_silgen_name("arbit_capture_context_new")
private func ffi_capture_context_new() -> OpaquePointer?

@_silgen_name("arbit_capture_context_free")
private func ffi_capture_context_free(_ handle: OpaquePointer?)

@_silgen_name("arbit_init_logging")
private func ffi_init_logging()

@_silgen_name("arbit_ingest_camera_frame")
private func ffi_ingest_camera_frame(
    _ handle: OpaquePointer?,
    _ frame: UnsafePointer<ArbitCameraFrame>?,
    _ sample: UnsafeMutablePointer<ArbitCameraSample>?
) -> Bool

@_silgen_name("arbit_capture_context_pyramid_levels")
private func ffi_capture_context_pyramid_levels(
    _ handle: OpaquePointer?,
    _ levels: UnsafeMutablePointer<ArbitPyramidLevelView>?,
    _ maxLevels: Int
) -> Int

@_silgen_name("arbit_capture_context_tracked_points")
private func ffi_capture_context_tracked_points(
    _ handle: OpaquePointer?,
    _ points: UnsafeMutablePointer<ArbitTrackedPoint>?,
    _ maxPoints: Int
) -> Int

@_silgen_name("arbit_capture_context_two_view")
private func ffi_capture_context_two_view(
    _ handle: OpaquePointer?,
    _ summary: UnsafeMutablePointer<ArbitTwoViewSummary>?
) -> Bool

@_silgen_name("arbit_capture_context_trajectory")
private func ffi_capture_context_trajectory(
    _ handle: OpaquePointer?,
    _ samples: UnsafeMutablePointer<ArbitPoseSample>?,
    _ maxPoints: Int
) -> Int

@_silgen_name("arbit_ingest_accelerometer_sample")
private func ffi_ingest_accelerometer_sample(
    _ handle: OpaquePointer?,
    _ sample: ArbitAccelerometerSample
) -> Bool

@_silgen_name("arbit_ingest_imu_sample")
private func ffi_ingest_imu_sample(
    _ handle: OpaquePointer?,
    _ sample: ArbitImuSample
) -> Bool

@_silgen_name("arbit_finish_imu_preintegration")
private func ffi_finish_imu_preintegration(
    _ handle: OpaquePointer?
) -> Bool

@_silgen_name("arbit_last_imu_rotation_prior")
private func ffi_last_imu_rotation_prior(
    _ handle: OpaquePointer?,
    _ rotation: UnsafeMutablePointer<Double>?
) -> Bool

@_silgen_name("arbit_last_motion_state")
private func ffi_last_motion_state(
    _ handle: OpaquePointer?,
    _ state: UnsafeMutablePointer<ArbitMotionState>?
) -> Bool

@_silgen_name("arbit_capture_context_gravity")
private func ffi_capture_context_gravity(
    _ handle: OpaquePointer?,
    _ estimate: UnsafeMutablePointer<ArbitGravityEstimate>?
) -> Bool

@_silgen_name("arbit_capture_context_map_stats")
private func ffi_capture_context_map_stats(
    _ handle: OpaquePointer?,
    _ keyframes: UnsafeMutablePointer<UInt64>?,
    _ landmarks: UnsafeMutablePointer<UInt64>?,
    _ anchors: UnsafeMutablePointer<UInt64>?
) -> Bool

@_silgen_name("arbit_capture_context_list_anchors")
private func ffi_capture_context_list_anchors(
    _ handle: OpaquePointer?,
    _ ids: UnsafeMutablePointer<UInt64>?,
    _ maxIds: Int
) -> Int

@_silgen_name("arbit_capture_context_create_anchor")
private func ffi_capture_context_create_anchor(
    _ handle: OpaquePointer?,
    _ pose: UnsafePointer<ArbitTransform>?,
    _ outId: UnsafeMutablePointer<UInt64>?
) -> Bool

@_silgen_name("arbit_capture_context_resolve_anchor")
private func ffi_capture_context_resolve_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64,
    _ outPose: UnsafeMutablePointer<ArbitTransform>?
) -> Bool

@_silgen_name("arbit_capture_context_update_anchor")
private func ffi_capture_context_update_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64,
    _ pose: UnsafePointer<ArbitTransform>?
) -> Bool

@_silgen_name("arbit_capture_context_last_relocalization")
private func ffi_capture_context_last_relocalization(
    _ handle: OpaquePointer?,
    _ summary: UnsafeMutablePointer<ArbitRelocalizationSummary>?
) -> Bool

@_silgen_name("arbit_capture_context_save_map")
private func ffi_capture_context_save_map(
    _ handle: OpaquePointer?,
    _ buffer: UnsafeMutablePointer<UInt8>?,
    _ bufferLen: Int,
    _ written: UnsafeMutablePointer<Int>?
) -> Bool

@_silgen_name("arbit_capture_context_load_map")
private func ffi_capture_context_load_map(
    _ handle: OpaquePointer?,
    _ data: UnsafePointer<UInt8>?,
    _ dataLen: Int
) -> Bool

public enum CameraPixelFormat: UInt32, CaseIterable, Sendable {
    case bgra8 = 0
    case rgba8 = 1
    case nv12 = 2
    case yv12 = 3
    case depth16 = 4

    var ffiValue: ArbitPixelFormat {
        switch self {
        case .bgra8: return ARBIT_PIXEL_FORMAT_BGRA8
        case .rgba8: return ARBIT_PIXEL_FORMAT_RGBA8
        case .nv12: return ARBIT_PIXEL_FORMAT_NV12
        case .yv12: return ARBIT_PIXEL_FORMAT_YV12
        case .depth16: return ARBIT_PIXEL_FORMAT_DEPTH16
        }
    }

    init?(ffiValue: ArbitPixelFormat) {
        switch ffiValue {
        case ARBIT_PIXEL_FORMAT_BGRA8: self = .bgra8
        case ARBIT_PIXEL_FORMAT_RGBA8: self = .rgba8
        case ARBIT_PIXEL_FORMAT_NV12: self = .nv12
        case ARBIT_PIXEL_FORMAT_YV12: self = .yv12
        case ARBIT_PIXEL_FORMAT_DEPTH16: self = .depth16
        default: return nil
        }
    }
}

public struct CameraIntrinsics: Sendable {
    public var fx: Double
    public var fy: Double
    public var cx: Double
    public var cy: Double
    public var skew: Double
    public var width: UInt32
    public var height: UInt32
    public var distortion: [Double]?

    public init(
        fx: Double,
        fy: Double,
        cx: Double,
        cy: Double,
        skew: Double,
        width: UInt32,
        height: UInt32,
        distortion: [Double]? = nil
    ) {
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.skew = skew
        self.width = width
        self.height = height
        self.distortion = distortion
    }

    init(ffiValue: ArbitCameraIntrinsics) {
        fx = ffiValue.fx
        fy = ffiValue.fy
        cx = ffiValue.cx
        cy = ffiValue.cy
        skew = ffiValue.skew
        width = ffiValue.width
        height = ffiValue.height

        if ffiValue.distortion_len > 0, let pointer = ffiValue.distortion {
            let buffer = UnsafeBufferPointer(start: pointer, count: ffiValue.distortion_len)
            distortion = Array(buffer)
        } else {
            distortion = nil
        }
    }

    func withFFI<Result>(_ body: (ArbitCameraIntrinsics) -> Result) -> Result {
        if let distortion, !distortion.isEmpty {
            return distortion.withUnsafeBufferPointer { buffer in
                let ffiIntrinsics = ArbitCameraIntrinsics(
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    skew: skew,
                    width: width,
                    height: height,
                    distortion_len: buffer.count,
                    distortion: buffer.baseAddress
                )
                return body(ffiIntrinsics)
            }
        } else {
            let ffiIntrinsics = ArbitCameraIntrinsics(
                fx: fx,
                fy: fy,
                cx: cx,
                cy: cy,
                skew: skew,
                width: width,
                height: height,
                distortion_len: 0,
                distortion: nil
            )
            return body(ffiIntrinsics)
        }
    }
}

public struct CameraFrame: Sendable {
    public var timestamp: TimeInterval
    public var intrinsics: CameraIntrinsics
    public var pixelFormat: CameraPixelFormat
    public var bytesPerRow: Int
    public var data: Data?

    public init(
        timestamp: TimeInterval,
        intrinsics: CameraIntrinsics,
        pixelFormat: CameraPixelFormat,
        bytesPerRow: Int,
        data: Data? = nil
    ) {
        self.timestamp = timestamp
        self.intrinsics = intrinsics
        self.pixelFormat = pixelFormat
        self.bytesPerRow = bytesPerRow
        self.data = data
    }

    func withFFI<Result>(_ body: (ArbitCameraFrame) -> Result) -> Result {
        intrinsics.withFFI { ffiIntrinsics in
            if let data, !data.isEmpty {
                return data.withUnsafeBytes { rawBuffer -> Result in
                    let byteBuffer = rawBuffer.bindMemory(to: UInt8.self)
                    let frame = ArbitCameraFrame(
                        timestamp_seconds: timestamp,
                        intrinsics: ffiIntrinsics,
                        pixel_format: pixelFormat.ffiValue,
                        bytes_per_row: bytesPerRow,
                        data: byteBuffer.baseAddress,
                        data_len: data.count
                    )
                    return body(frame)
                }
            } else {
                let frame = ArbitCameraFrame(
                    timestamp_seconds: timestamp,
                    intrinsics: ffiIntrinsics,
                    pixel_format: pixelFormat.ffiValue,
                    bytes_per_row: bytesPerRow,
                    data: nil,
                    data_len: 0
                )
                return body(frame)
            }
        }
    }
}

public struct FrameTimestamps: Sendable {
    public var capture: TimeInterval
    public var pipeline: TimeInterval
    public var latency: TimeInterval

    init(ffiValue: ArbitFrameTimestamps) {
        capture = ffiValue.capture_seconds
        pipeline = ffiValue.pipeline_seconds
        latency = ffiValue.latency_seconds
    }
}

public struct CameraSample: Sendable {
    public var timestamps: FrameTimestamps
    public var intrinsics: CameraIntrinsics
    public var pixelFormat: CameraPixelFormat
    public var bytesPerRow: Int

    init?(ffiValue: ArbitCameraSample) {
        guard let pixelFormat = CameraPixelFormat(ffiValue: ffiValue.pixel_format) else {
            return nil
        }

        timestamps = FrameTimestamps(ffiValue: ffiValue.timestamps)
        intrinsics = CameraIntrinsics(ffiValue: ffiValue.intrinsics)
        self.pixelFormat = pixelFormat
        bytesPerRow = ffiValue.bytes_per_row
    }
}

public enum TrackStatus: UInt32, Sendable {
    case converged = 0
    case diverged = 1
    case outOfBounds = 2

    init(ffiValue: ArbitTrackStatus) {
        switch UInt32(ffiValue.rawValue) {
        case 0:
            self = .converged
        case 1:
            self = .diverged
        case 2:
            self = .outOfBounds
        default:
            self = .diverged
        }
    }
}

public struct TrackedPoint: Sendable {
    public var initial: SIMD2<Float>
    public var refined: SIMD2<Float>
    public var residual: Float
    public var iterations: UInt32
    public var status: TrackStatus

    init(ffiValue: ArbitTrackedPoint) {
        initial = SIMD2(x: ffiValue.initial_x, y: ffiValue.initial_y)
        refined = SIMD2(x: ffiValue.refined_x, y: ffiValue.refined_y)
        residual = ffiValue.residual
        iterations = ffiValue.iterations
        status = TrackStatus(ffiValue: ffiValue.status)
    }
}

public struct PyramidLevelView: Sendable {
    public var octave: Int
    public var scale: Float
    public var width: Int
    public var height: Int
    public var bytesPerRow: Int
    public var data: Data

    init?(ffiValue: ArbitPyramidLevelView) {
        guard let pixels = ffiValue.pixels, ffiValue.pixels_len > 0 else {
            return nil
        }
        octave = Int(ffiValue.octave)
        scale = ffiValue.scale
        width = Int(ffiValue.width)
        height = Int(ffiValue.height)
        bytesPerRow = Int(ffiValue.bytes_per_row)
        data = Data(bytes: pixels, count: Int(ffiValue.pixels_len))
    }
}

public struct TwoViewSummary: Sendable {
    public var inliers: Int
    public var averageError: Double
    public var rotation: [Double]
    public var translation: SIMD3<Double>

    init(ffiValue: ArbitTwoViewSummary) {
        inliers = Int(ffiValue.inliers)
        averageError = ffiValue.average_error
        var rotationElements = withUnsafeBytes(of: ffiValue.rotation) { buffer -> [Double] in
            Array(buffer.bindMemory(to: Double.self).prefix(9))
        }
        if rotationElements.count < 9 {
            rotationElements.append(contentsOf: Array(repeating: 0.0, count: 9 - rotationElements.count))
        }
        rotation = rotationElements
        let translationElements = withUnsafeBytes(of: ffiValue.translation) { buffer -> [Double] in
            Array(buffer.bindMemory(to: Double.self).prefix(3))
        }
        if translationElements.count >= 3 {
            translation = SIMD3(translationElements[0], translationElements[1], translationElements[2])
        } else {
            translation = SIMD3.zero
        }
    }
}

public struct PoseSample: Sendable {
    public var position: SIMD3<Double>

    init(ffiValue: ArbitPoseSample) {
        position = SIMD3(ffiValue.x, ffiValue.y, ffiValue.z)
    }
}

public struct GravityVector: Sendable {
    public var down: SIMD3<Double>
    public var samples: UInt32

    init(ffiValue: ArbitGravityEstimate) {
        let components = withUnsafeBytes(of: ffiValue.down) { buffer -> [Double] in
            Array(buffer.bindMemory(to: Double.self).prefix(3))
        }
        if components.count >= 3 {
            down = SIMD3(components[0], components[1], components[2])
        } else {
            down = SIMD3.zero
        }
        samples = ffiValue.samples
    }
}

public struct MapStats: Sendable {
    public var keyframes: UInt64
    public var landmarks: UInt64
    public var anchors: UInt64

    public init(keyframes: UInt64, landmarks: UInt64, anchors: UInt64) {
        self.keyframes = keyframes
        self.landmarks = landmarks
        self.anchors = anchors
    }
}

public struct RelocalizationSummary: Sendable {
    public var pose: simd_double4x4
    public var inliers: UInt32
    public var averageError: Double

    init(ffiValue: ArbitRelocalizationSummary) {
        pose = matrixFromArbitTransform(ffiValue.pose)
        inliers = ffiValue.inliers
        averageError = ffiValue.average_error
    }
}

private func makeArbitTransform(from matrix: simd_double4x4) -> ArbitTransform {
    var transform = ArbitTransform()
    withUnsafeMutablePointer(to: &transform) { pointer in
        pointer.withMemoryRebound(to: Double.self, capacity: 16) { buffer in
            for column in 0..<4 {
                for row in 0..<4 {
                    buffer[column * 4 + row] = matrix[column][row]
                }
            }
        }
    }
    return transform
}

private func matrixFromArbitTransform(_ transform: ArbitTransform) -> simd_double4x4 {
    var matrix = simd_double4x4()
    withUnsafePointer(to: transform) { pointer in
        pointer.withMemoryRebound(to: Double.self, capacity: 16) { buffer in
            for column in 0..<4 {
                for row in 0..<4 {
                    matrix[column][row] = buffer[column * 4 + row]
                }
            }
        }
    }
    return matrix
}

public enum ArbitCaptureError: Error {
    case allocationFailed
    case ingestionFailed
    case mapSaveFailed
    case mapLoadFailed
}

public final class ArbitCaptureContext: @unchecked Sendable {
    private var handle: OpaquePointer

    public init() throws {
        guard let pointer = ffi_capture_context_new() else {
            throw ArbitCaptureError.allocationFailed
        }
        handle = pointer
    }

    deinit {
        ffi_capture_context_free(handle)
    }

    /// Initialize logging for the Rust library
    public static func initLogging() {
        ffi_init_logging()
    }

    public func ingest(_ frame: CameraFrame) throws -> CameraSample {
        let result = frame.withFFI { ffiFrame -> CameraSample? in
            var frameCopy = ffiFrame
            var rawSample = ArbitCameraSample(
                timestamps: ArbitFrameTimestamps(
                    capture_seconds: 0,
                    pipeline_seconds: 0,
                    latency_seconds: 0
                ),
                intrinsics: ArbitCameraIntrinsics(
                    fx: 0,
                    fy: 0,
                    cx: 0,
                    cy: 0,
                    skew: 0,
                    width: 0,
                    height: 0,
                    distortion_len: 0,
                    distortion: nil
                ),
                pixel_format: frame.pixelFormat.ffiValue,
                bytes_per_row: 0
            )

            let success = withUnsafePointer(to: &frameCopy) { framePtr in
                ffi_ingest_camera_frame(handle, framePtr, &rawSample)
            }

            if success {
                return CameraSample(ffiValue: rawSample)
            } else {
                return nil
            }
        }

        guard let sample = result else {
            throw ArbitCaptureError.ingestionFailed
        }

        return sample
    }

    /// Ingest a full 6DOF IMU sample (accelerometer + gyroscope)
    /// This is the preferred method for feeding IMU data as it enables preintegration
    public func ingestIMUSample(_ sample: ImuSample) throws {
        guard ffi_ingest_imu_sample(handle, sample.toFFI()) else {
            throw ArbitCaptureError.ingestionFailed
        }
    }
    
    /// Finish the current IMU preintegration interval
    /// 
    /// **DEPRECATED**: This is now called automatically when you ingest a camera frame,
    /// so you no longer need to call this manually. It remains for backwards compatibility.
    @available(*, deprecated, message: "IMU preintegration is now finished automatically when ingesting camera frames")
    public func finishIMUPreintegration() {
        _ = ffi_finish_imu_preintegration(handle)
    }
    
    /// Get the last IMU rotation prior (in radians)
    public func lastIMURotationPrior() -> Double? {
        var rotation: Double = 0
        if ffi_last_imu_rotation_prior(handle, &rotation) {
            return rotation
        }
        return nil
    }
    
    /// Get the last motion state
    public func lastMotionState() -> MotionState? {
        var ffiState = ArbitFFI.ArbitMotionState(state: 0)
        if ffi_last_motion_state(handle, &ffiState) {
            return MotionState(ffiState: ffiState)
        }
        return nil
    }

    /// Validates that a frame can be processed before ingestion
    public func validateFrame(_ frame: CameraFrame) -> Bool {
        return frame.withFFI { ffiFrame -> Bool in
            var frameCopy = ffiFrame
            var rawSample = ArbitCameraSample(
                timestamps: ArbitFrameTimestamps(
                    capture_seconds: 0,
                    pipeline_seconds: 0,
                    latency_seconds: 0
                ),
                intrinsics: ArbitCameraIntrinsics(
                    fx: 0,
                    fy: 0,
                    cx: 0,
                    cy: 0,
                    skew: 0,
                    width: 0,
                    height: 0,
                    distortion_len: 0,
                    distortion: nil
                ),
                pixel_format: frame.pixelFormat.ffiValue,
                bytes_per_row: 0
            )

            return withUnsafePointer(to: &frameCopy) { framePtr in
                ffi_ingest_camera_frame(handle, framePtr, &rawSample)
            }
        }
    }

    public func pyramidLevels(maxLevels: Int = 3) -> [PyramidLevelView] {
        guard maxLevels > 0 else { return [] }
        let buffer = UnsafeMutablePointer<ArbitPyramidLevelView>.allocate(capacity: maxLevels)
        defer { buffer.deallocate() }
        let written = ffi_capture_context_pyramid_levels(handle, buffer, maxLevels)
        guard written > 0 else { return [] }
        var result: [PyramidLevelView] = []
        result.reserveCapacity(written)
        for index in 0..<written {
            if let view = PyramidLevelView(ffiValue: buffer[index]) {
                result.append(view)
            }
        }
        return result
    }

    public func trackedPoints(maxPoints: Int = 256) -> [TrackedPoint] {
        guard maxPoints > 0 else { return [] }
        let buffer = UnsafeMutablePointer<ArbitTrackedPoint>.allocate(capacity: maxPoints)
        defer { buffer.deallocate() }
        let written = ffi_capture_context_tracked_points(handle, buffer, maxPoints)
        guard written > 0 else { return [] }
        var points: [TrackedPoint] = []
        points.reserveCapacity(written)
        for index in 0..<written {
            points.append(TrackedPoint(ffiValue: buffer[index]))
        }
        return points
    }

    public func latestTwoViewSummary() -> TwoViewSummary? {
        let summaryPtr = UnsafeMutablePointer<ArbitTwoViewSummary>.allocate(capacity: 1)
        defer { summaryPtr.deallocate() }
        let hasResult = ffi_capture_context_two_view(handle, summaryPtr)
        guard hasResult else {
            return nil
        }
        return TwoViewSummary(ffiValue: summaryPtr.pointee)
    }

    public func trajectory(maxPoints: Int = 512) -> [PoseSample] {
        guard maxPoints > 0 else { return [] }
        let buffer = UnsafeMutablePointer<ArbitPoseSample>.allocate(capacity: maxPoints)
        defer { buffer.deallocate() }
        let written = ffi_capture_context_trajectory(handle, buffer, maxPoints)
        guard written > 0 else { return [] }
        var samples: [PoseSample] = []
        samples.reserveCapacity(written)
        for index in 0..<written {
            samples.append(PoseSample(ffiValue: buffer[index]))
        }
        return samples
    }

    public func ingestAccelerometer(ax: Double, ay: Double, az: Double, dt: Double) {
        let sample = ArbitAccelerometerSample(ax: ax, ay: ay, az: az, dt_seconds: dt)
        _ = ffi_ingest_accelerometer_sample(handle, sample)
    }

    public func gravityEstimate() -> GravityVector? {
        let estimatePtr = UnsafeMutablePointer<ArbitGravityEstimate>.allocate(capacity: 1)
        defer { estimatePtr.deallocate() }
        let hasEstimate = ffi_capture_context_gravity(handle, estimatePtr)
        guard hasEstimate else { return nil }
        return GravityVector(ffiValue: estimatePtr.pointee)
    }

    public func mapStats() -> MapStats {
        var keyframes: UInt64 = 0
        var landmarks: UInt64 = 0
        var anchors: UInt64 = 0
        _ = ffi_capture_context_map_stats(handle, &keyframes, &landmarks, &anchors)
        return MapStats(keyframes: keyframes, landmarks: landmarks, anchors: anchors)
    }

    public func anchorIds(maxCount: Int = 16) -> [UInt64] {
        guard maxCount > 0 else { return [] }
        var buffer = Array(repeating: UInt64(0), count: maxCount)
        let written = ffi_capture_context_list_anchors(handle, &buffer, maxCount)
        guard written > 0 else { return [] }
        return Array(buffer.prefix(written))
    }

    public func createAnchor(pose: simd_double4x4) -> UInt64? {
        var anchorId: UInt64 = 0
        var transform = makeArbitTransform(from: pose)
        let success = withUnsafePointer(to: &transform) { pointer in
            ffi_capture_context_create_anchor(handle, pointer, &anchorId)
        }
        return success ? anchorId : nil
    }

    public func resolveAnchor(_ anchorId: UInt64) -> simd_double4x4? {
        var transform = ArbitTransform()
        let success = withUnsafeMutablePointer(to: &transform) { pointer in
            ffi_capture_context_resolve_anchor(handle, anchorId, pointer)
        }
        return success ? matrixFromArbitTransform(transform) : nil
    }

    public func updateAnchor(_ anchorId: UInt64, pose: simd_double4x4) -> Bool {
        var transform = makeArbitTransform(from: pose)
        return withUnsafePointer(to: &transform) { pointer in
            ffi_capture_context_update_anchor(handle, anchorId, pointer)
        }
    }

    public func lastRelocalizationSummary() -> RelocalizationSummary? {
        var summary = ArbitRelocalizationSummary(
            pose: ArbitTransform(),
            inliers: 0,
            average_error: 0.0
        )
        let success = withUnsafeMutablePointer(to: &summary) { pointer in
            ffi_capture_context_last_relocalization(handle, pointer)
        }
        return success ? RelocalizationSummary(ffiValue: summary) : nil
    }

    public func saveMap() throws -> Data {
        var required = 0
        let query = ffi_capture_context_save_map(handle, nil, 0, &required)
        if query && required == 0 {
            return Data()
        }
        guard required >= 0 else {
            throw ArbitCaptureError.mapSaveFailed
        }

        var data = Data(count: required)
        let saved = data.withUnsafeMutableBytes { buffer -> Bool in
            guard let baseAddress = buffer.bindMemory(to: UInt8.self).baseAddress else {
                return false
            }
            return ffi_capture_context_save_map(handle, baseAddress, required, &required)
        }
        guard saved else {
            throw ArbitCaptureError.mapSaveFailed
        }

        data.count = required
        return data
    }

    public func loadMap(_ data: Data) throws {
        guard !data.isEmpty else {
            throw ArbitCaptureError.mapLoadFailed
        }
        let loaded = data.withUnsafeBytes { buffer -> Bool in
            guard let baseAddress = buffer.bindMemory(to: UInt8.self).baseAddress else {
                return false
            }
            return ffi_capture_context_load_map(handle, baseAddress, buffer.count)
        }
        guard loaded else {
            throw ArbitCaptureError.mapLoadFailed
        }
    }
}

#if canImport(ARKit)
public extension CameraIntrinsics {
    init(camera: ARCamera) {
        let intrinsics = camera.intrinsics
        let fx = Double(intrinsics.columns.0.x)
        let fy = Double(intrinsics.columns.1.y)
        let cx = Double(intrinsics.columns.2.x)
        let cy = Double(intrinsics.columns.2.y)
        let width = UInt32(camera.imageResolution.width.rounded())
        let height = UInt32(camera.imageResolution.height.rounded())

        self.init(
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            skew: Double(intrinsics.columns.0.y),
            width: width,
            height: height,
            distortion: nil
        )
    }
}

public extension CameraFrame {
    init?(arFrame: ARFrame, pixelFormat: CameraPixelFormat) {
        let buffer = arFrame.capturedImage

        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }

        let bytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let data = Data(bytes: baseAddress, count: bytesPerRow * height)

        self.init(
            timestamp: arFrame.timestamp,
            intrinsics: CameraIntrinsics(camera: arFrame.camera),
            pixelFormat: pixelFormat,
            bytesPerRow: bytesPerRow,
            data: data
        )
    }
}
#endif
