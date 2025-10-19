import Foundation
import ArbitFFI
#if canImport(CoreVideo)
import CoreVideo
#endif
#if canImport(simd)
import simd
#endif
#if canImport(ARKit) && !os(macOS)
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
public enum MotionState: String, Sendable {
    case stationary = "Stationary"
    case slowMotion = "SlowMotion"
    case fastMotion = "FastMotion"
}

/// Unified IMU state containing all IMU-related information
public struct ImuState: Sendable {
    public var hasRotationPrior: Bool
    public var rotationPriorRadians: Double
    public var hasMotionState: Bool
    public var motionState: MotionState?
    public var hasGravity: Bool
    public var gravityDown: SIMD3<Double>
    public var gravitySamples: UInt32
    public var preintegrationCount: UInt32
    
    init(ffiValue: ArbitFFI.ArbitImuState) {
        hasRotationPrior = ffiValue.has_rotation_prior
        rotationPriorRadians = ffiValue.rotation_prior_radians
        hasMotionState = ffiValue.has_motion_state
        
        if hasMotionState {
            switch ffiValue.motion_state {
            case 0: motionState = .stationary
            case 1: motionState = .slowMotion
            case 2: motionState = .fastMotion
            default: motionState = nil
            }
        } else {
            motionState = nil
        }
        
        hasGravity = ffiValue.has_gravity
        let gravityComponents = withUnsafeBytes(of: ffiValue.gravity_down) { buffer -> [Double] in
            Array(buffer.bindMemory(to: Double.self).prefix(3))
        }
        if gravityComponents.count >= 3 {
            gravityDown = SIMD3(gravityComponents[0], gravityComponents[1], gravityComponents[2])
        } else {
            gravityDown = SIMD3.zero
        }
        gravitySamples = ffiValue.gravity_samples
        preintegrationCount = ffiValue.preintegration_count
    }
}

/// Comprehensive frame state containing all common query results
public struct FrameState: Sendable {
    // Tracking state
    public var trackCount: UInt32
    public var hasTwoView: Bool
    public var twoView: TwoViewSummary?
    public var hasRelocalization: Bool
    public var relocalization: RelocalizationSummary?
    
    // Map state
    public var keyframeCount: UInt64
    public var landmarkCount: UInt64
    public var anchorCount: UInt64
    
    // IMU state
    public var imu: ImuState
    
    init(ffiValue: ArbitFFI.ArbitFrameState) {
        trackCount = ffiValue.track_count
        hasTwoView = ffiValue.has_two_view
        twoView = hasTwoView ? TwoViewSummary(ffiValue: ffiValue.two_view) : nil
        hasRelocalization = ffiValue.has_relocalization
        relocalization = hasRelocalization ? RelocalizationSummary(ffiValue: ffiValue.relocalization) : nil
        keyframeCount = ffiValue.keyframe_count
        landmarkCount = ffiValue.landmark_count
        anchorCount = ffiValue.anchor_count
        imu = ImuState(ffiValue: ffiValue.imu)
    }
}

// MARK: - FFI Function Declarations

@_silgen_name("arbit_context_create")
private func ffi_context_create() -> OpaquePointer?

@_silgen_name("arbit_context_destroy")
private func ffi_context_destroy(_ handle: OpaquePointer?)

@_silgen_name("arbit_init_logging")
private func ffi_init_logging()

@_silgen_name("arbit_ingest_frame")
private func ffi_ingest_frame(
    _ handle: OpaquePointer?,
    _ frame: UnsafePointer<ArbitFFI.ArbitCameraFrame>?,
    _ sample: UnsafeMutablePointer<ArbitFFI.ArbitCameraSample>?
) -> Bool

@_silgen_name("arbit_ingest_imu")
private func ffi_ingest_imu(
    _ handle: OpaquePointer?,
    _ sample: ArbitFFI.ArbitImuSample
) -> Bool

@_silgen_name("arbit_get_imu_state")
private func ffi_get_imu_state(
    _ handle: OpaquePointer?,
    _ state: UnsafeMutablePointer<ArbitFFI.ArbitImuState>?
) -> Bool

@_silgen_name("arbit_get_frame_state")
private func ffi_get_frame_state(
    _ handle: OpaquePointer?,
    _ state: UnsafeMutablePointer<ArbitFFI.ArbitFrameState>?
) -> Bool

@_silgen_name("arbit_get_pyramid_levels")
private func ffi_get_pyramid_levels(
    _ handle: OpaquePointer?,
    _ levels: UnsafeMutablePointer<ArbitFFI.ArbitPyramidLevelView>?,
    _ maxLevels: Int
) -> Int

@_silgen_name("arbit_get_tracked_points")
private func ffi_get_tracked_points(
    _ handle: OpaquePointer?,
    _ points: UnsafeMutablePointer<ArbitFFI.ArbitTrackedPoint>?,
    _ maxPoints: Int
) -> Int

@_silgen_name("arbit_get_trajectory")
private func ffi_get_trajectory(
    _ handle: OpaquePointer?,
    _ samples: UnsafeMutablePointer<ArbitFFI.ArbitPoseSample>?,
    _ maxPoints: Int
) -> Int

@_silgen_name("arbit_list_anchors")
private func ffi_list_anchors(
    _ handle: OpaquePointer?,
    _ ids: UnsafeMutablePointer<UInt64>?,
    _ maxIds: Int
) -> Int

@_silgen_name("arbit_create_anchor")
private func ffi_create_anchor(
    _ handle: OpaquePointer?,
    _ pose: UnsafePointer<ArbitFFI.ArbitTransform>?,
    _ outId: UnsafeMutablePointer<UInt64>?
) -> Bool

@_silgen_name("arbit_get_anchor")
private func ffi_get_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64,
    _ outPose: UnsafeMutablePointer<ArbitFFI.ArbitTransform>?
) -> Bool

@_silgen_name("arbit_update_anchor")
private func ffi_update_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64,
    _ pose: UnsafePointer<ArbitFFI.ArbitTransform>?
) -> Bool

@_silgen_name("arbit_remove_anchor")
private func ffi_remove_anchor(
    _ handle: OpaquePointer?,
    _ anchorId: UInt64
) -> Bool

@_silgen_name("arbit_place_anchor_at_screen_point")
private func ffi_place_anchor_at_screen_point(
    _ handle: OpaquePointer?,
    _ normalizedU: Double,
    _ normalizedV: Double,
    _ depth: Double,
    _ outAnchorId: UnsafeMutablePointer<UInt64>?
) -> Bool

@_silgen_name("arbit_get_visible_anchors")
private func ffi_get_visible_anchors(
    _ handle: OpaquePointer?,
    _ outAnchors: UnsafeMutablePointer<ArbitFFI.ArbitProjectedAnchor>?,
    _ maxAnchors: Int
) -> Int

@_silgen_name("arbit_get_visible_landmarks")
private func ffi_get_visible_landmarks(
    _ handle: OpaquePointer?,
    _ outLandmarks: UnsafeMutablePointer<ArbitFFI.ArbitProjectedLandmark>?,
    _ maxLandmarks: Int
) -> Int

@_silgen_name("arbit_get_map_debug_snapshot")
private func ffi_get_map_debug_snapshot(
    _ handle: OpaquePointer?,
    _ outSnapshot: UnsafeMutablePointer<ArbitFFI.ArbitMapDebugSnapshot>?
) -> Bool

@_silgen_name("arbit_save_map")
private func ffi_save_map(
    _ handle: OpaquePointer?,
    _ buffer: UnsafeMutablePointer<UInt8>?,
    _ bufferLen: Int,
    _ written: UnsafeMutablePointer<Int>?
) -> Bool

@_silgen_name("arbit_load_map")
private func ffi_load_map(
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

    var ffiValue: ArbitFFI.ArbitPixelFormat {
        ArbitFFI.ArbitPixelFormat(rawValue: self.rawValue)
    }

    init?(ffiValue: ArbitFFI.ArbitPixelFormat) {
        self.init(rawValue: ffiValue.rawValue)
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

    init(ffiValue: ArbitFFI.ArbitCameraIntrinsics) {
        fx = ffiValue.fx
        fy = ffiValue.fy
        cx = ffiValue.cx
        cy = ffiValue.cy
        skew = ffiValue.skew
        width = ffiValue.width
        height = ffiValue.height

        if ffiValue.distortion_len > 0, let pointer = ffiValue.distortion {
            let buffer = UnsafeBufferPointer(start: pointer, count: Int(ffiValue.distortion_len))
            distortion = Array(buffer)
        } else {
            distortion = nil
        }
    }

    func withFFI<Result>(_ body: (ArbitFFI.ArbitCameraIntrinsics) -> Result) -> Result {
        if let distortion, !distortion.isEmpty {
            return distortion.withUnsafeBufferPointer { buffer in
                let ffiIntrinsics = ArbitFFI.ArbitCameraIntrinsics(
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    skew: skew,
                    width: width,
                    height: height,
                    distortion_len: UInt(buffer.count),
                    distortion: buffer.baseAddress
                )
                return body(ffiIntrinsics)
            }
        } else {
            let ffiIntrinsics = ArbitFFI.ArbitCameraIntrinsics(
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

    func withFFI<Result>(_ body: (ArbitFFI.ArbitCameraFrame) -> Result) -> Result {
        intrinsics.withFFI { ffiIntrinsics in
            if let data, !data.isEmpty {
                return data.withUnsafeBytes { (rawBuffer: UnsafeRawBufferPointer) -> Result in
                    let byteBuffer = rawBuffer.bindMemory(to: UInt8.self)
                    let frame = ArbitFFI.ArbitCameraFrame(
                        timestamp_seconds: timestamp,
                        intrinsics: ffiIntrinsics,
                        pixel_format: pixelFormat.ffiValue,
                        bytes_per_row: UInt(bytesPerRow),
                        data: byteBuffer.baseAddress,
                        data_len: UInt(data.count)
                    )
                    return body(frame)
                }
            } else {
                let frame = ArbitFFI.ArbitCameraFrame(
                    timestamp_seconds: timestamp,
                    intrinsics: ffiIntrinsics,
                    pixel_format: pixelFormat.ffiValue,
                    bytes_per_row: UInt(bytesPerRow),
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

    init(ffiValue: ArbitFFI.ArbitFrameTimestamps) {
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

    init?(ffiValue: ArbitFFI.ArbitCameraSample) {
        guard let pixelFormat = CameraPixelFormat(ffiValue: ffiValue.pixel_format) else {
            return nil
        }

        timestamps = FrameTimestamps(ffiValue: ffiValue.timestamps)
        intrinsics = CameraIntrinsics(ffiValue: ffiValue.intrinsics)
        self.pixelFormat = pixelFormat
        bytesPerRow = Int(ffiValue.bytes_per_row)
    }
}

public enum TrackStatus: UInt32, Sendable {
    case converged = 0
    case diverged = 1
    case outOfBounds = 2

    init(ffiValue: ArbitFFI.ArbitTrackStatus) {
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

    init(ffiValue: ArbitFFI.ArbitTrackedPoint) {
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

    init?(ffiValue: ArbitFFI.ArbitPyramidLevelView) {
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

    init(ffiValue: ArbitFFI.ArbitTwoViewSummary) {
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

    init(ffiValue: ArbitFFI.ArbitPoseSample) {
        position = SIMD3(ffiValue.x, ffiValue.y, ffiValue.z)
    }
}

// GravityVector removed - gravity information is now part of ImuState

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

/// An anchor projected into screen space with visibility information
public struct ProjectedAnchor: Sendable {
    /// The anchor identifier
    public let anchorId: UInt64
    /// The anchor's world pose (4x4 transformation matrix)
    public let pose: simd_double4x4
    /// Optional keyframe ID this anchor was created from
    public let createdFromKeyframe: UInt64?
    /// Normalized image coordinates in range [0, 1]
    public let normalizedU: Double
    public let normalizedV: Double
    /// Pixel coordinates in the current frame
    public let pixelX: Float
    public let pixelY: Float
    /// Depth from camera in meters
    public let depth: Double
    
    init(ffiValue: ArbitFFI.ArbitProjectedAnchor) {
        anchorId = ffiValue.anchor_id
        pose = matrixFromArbitTransform(ffiValue.pose)
        createdFromKeyframe = ffiValue.has_keyframe ? ffiValue.created_from_keyframe : nil
        normalizedU = ffiValue.normalized_u
        normalizedV = ffiValue.normalized_v
        pixelX = ffiValue.pixel_x
        pixelY = ffiValue.pixel_y
        depth = ffiValue.depth
    }
}

/// A landmark projected into screen space for debugging
public struct ProjectedLandmark: Sendable {
    public let landmarkId: UInt64
    public let worldPosition: SIMD3<Double>
    public let normalizedU: Double
    public let normalizedV: Double
    public let pixelX: Float
    public let pixelY: Float
    public let depth: Double
    
    init(ffiValue: ArbitFFI.ArbitProjectedLandmark) {
        landmarkId = ffiValue.landmark_id
        worldPosition = SIMD3(ffiValue.world_x, ffiValue.world_y, ffiValue.world_z)
        normalizedU = ffiValue.normalized_u
        normalizedV = ffiValue.normalized_v
        pixelX = ffiValue.pixel_x
        pixelY = ffiValue.pixel_y
        depth = ffiValue.depth
    }
}

/// Debug snapshot of the map state
public struct MapDebugSnapshot: Sendable {
    public let cameraPosition: SIMD3<Double>
    public let cameraRotation: simd_double3x3
    public let landmarkCount: UInt64
    public let keyframeCount: UInt64
    public let anchorCount: UInt64
    
    init(ffiValue: ArbitFFI.ArbitMapDebugSnapshot) {
        cameraPosition = SIMD3(ffiValue.camera_x, ffiValue.camera_y, ffiValue.camera_z)
        
        let r = ffiValue.camera_rotation
        cameraRotation = simd_double3x3(
            SIMD3(r.0, r.1, r.2),
            SIMD3(r.3, r.4, r.5),
            SIMD3(r.6, r.7, r.8)
        )
        
        landmarkCount = ffiValue.landmark_count
        keyframeCount = ffiValue.keyframe_count
        anchorCount = ffiValue.anchor_count
    }
}

public struct RelocalizationSummary: Sendable {
    public var pose: simd_double4x4
    public var inliers: UInt32
    public var averageError: Double

    init(ffiValue: ArbitFFI.ArbitRelocalizationSummary) {
        pose = matrixFromArbitTransform(ffiValue.pose)
        inliers = ffiValue.inliers
        averageError = ffiValue.average_error
    }
}

private func makeArbitTransform(from matrix: simd_double4x4) -> ArbitTransform {
    var transform = ArbitFFI.ArbitTransform()
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

private func matrixFromArbitTransform(_ transform: ArbitFFI.ArbitTransform) -> simd_double4x4 {
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
        guard let pointer = ffi_context_create() else {
            throw ArbitCaptureError.allocationFailed
        }
        handle = pointer
    }

    deinit {
        ffi_context_destroy(handle)
    }

    /// Initialize logging for the Rust library
    public static func initLogging() {
        ffi_init_logging()
    }

    public func ingest(_ frame: CameraFrame) throws -> CameraSample {
        let result = frame.withFFI { ffiFrame -> CameraSample? in
            var frameCopy = ffiFrame
            var rawSample = ArbitFFI.ArbitCameraSample(
                timestamps: ArbitFFI.ArbitFrameTimestamps(
                    capture_seconds: 0,
                    pipeline_seconds: 0,
                    latency_seconds: 0
                ),
                intrinsics: ArbitFFI.ArbitCameraIntrinsics(
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
                ffi_ingest_frame(handle, framePtr, &rawSample)
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
        guard ffi_ingest_imu(handle, sample.toFFI()) else {
            throw ArbitCaptureError.ingestionFailed
        }
    }
    
    // MARK: - Simplified API Methods
    
    /// Get unified IMU state in a single call (replaces multiple IMU queries)
    public func getImuState() -> ImuState? {
        var ffiState = ArbitFFI.ArbitImuState()
        if ffi_get_imu_state(handle, &ffiState) {
            return ImuState(ffiValue: ffiState)
        }
        return nil
    }
    
    /// Get comprehensive frame state in a single call (replaces 7+ separate queries)
    /// This is the preferred method for getting all frame-related state efficiently.
    public func getFrameState() -> FrameState? {
        var ffiState = ArbitFFI.ArbitFrameState()
        if ffi_get_frame_state(handle, &ffiState) {
            return FrameState(ffiValue: ffiState)
        }
        return nil
    }

    /// Validates that a frame can be processed before ingestion
    public func validateFrame(_ frame: CameraFrame) -> Bool {
        return frame.withFFI { ffiFrame -> Bool in
            var frameCopy = ffiFrame
            var rawSample = ArbitFFI.ArbitCameraSample(
                timestamps: ArbitFFI.ArbitFrameTimestamps(
                    capture_seconds: 0,
                    pipeline_seconds: 0,
                    latency_seconds: 0
                ),
                intrinsics: ArbitFFI.ArbitCameraIntrinsics(
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
                ffi_ingest_frame(handle, framePtr, &rawSample)
            }
        }
    }

    public func pyramidLevels(maxLevels: Int = 3) -> [PyramidLevelView] {
        guard maxLevels > 0 else { return [] }
        let buffer = UnsafeMutablePointer<ArbitFFI.ArbitPyramidLevelView>.allocate(capacity: maxLevels)
        defer { buffer.deallocate() }
        let written = ffi_get_pyramid_levels(handle, buffer, maxLevels)
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
        let buffer = UnsafeMutablePointer<ArbitFFI.ArbitTrackedPoint>.allocate(capacity: maxPoints)
        defer { buffer.deallocate() }
        let written = ffi_get_tracked_points(handle, buffer, maxPoints)
        guard written > 0 else { return [] }
        var points: [TrackedPoint] = []
        points.reserveCapacity(written)
        for index in 0..<written {
            points.append(TrackedPoint(ffiValue: buffer[index]))
        }
        return points
    }

    public func trajectory(maxPoints: Int = 512) -> [PoseSample] {
        guard maxPoints > 0 else { return [] }
        let buffer = UnsafeMutablePointer<ArbitFFI.ArbitPoseSample>.allocate(capacity: maxPoints)
        defer { buffer.deallocate() }
        let written = ffi_get_trajectory(handle, buffer, maxPoints)
        guard written > 0 else { return [] }
        var samples: [PoseSample] = []
        samples.reserveCapacity(written)
        for index in 0..<written {
            samples.append(PoseSample(ffiValue: buffer[index]))
        }
        return samples
    }

    public func anchorIds(maxCount: Int = 16) -> [UInt64] {
        guard maxCount > 0 else { return [] }
        var buffer = Array(repeating: UInt64(0), count: maxCount)
        let written = ffi_list_anchors(handle, &buffer, maxCount)
        guard written > 0 else { return [] }
        return Array(buffer.prefix(written))
    }

    public func createAnchor(pose: simd_double4x4) -> UInt64? {
        var anchorId: UInt64 = 0
        var transform = makeArbitTransform(from: pose)
        let success = withUnsafePointer(to: &transform) { pointer in
            ffi_create_anchor(handle, pointer, &anchorId)
        }
        return success ? anchorId : nil
    }

    public func resolveAnchor(_ anchorId: UInt64) -> simd_double4x4? {
        var transform = ArbitFFI.ArbitTransform()
        let success = withUnsafeMutablePointer(to: &transform) { pointer in
            ffi_get_anchor(handle, anchorId, pointer)
        }
        return success ? matrixFromArbitTransform(transform) : nil
    }

    public func updateAnchor(_ anchorId: UInt64, pose: simd_double4x4) -> Bool {
        var transform = makeArbitTransform(from: pose)
        return withUnsafePointer(to: &transform) { pointer in
            ffi_update_anchor(handle, anchorId, pointer)
        }
    }
    
    /// Remove an anchor by its identifier
    /// - Parameter anchorId: The identifier of the anchor to remove
    /// - Returns: `true` if the anchor was removed, `false` if it didn't exist
    public func removeAnchor(_ anchorId: UInt64) -> Bool {
        ffi_remove_anchor(handle, anchorId)
    }
    
    /// Place an anchor by raycasting from a screen point into the scene
    /// The engine uses its current camera pose and intrinsics to compute the world position
    /// - Parameters:
    ///   - normalizedU: Horizontal screen coordinate in range [0, 1]
    ///   - normalizedV: Vertical screen coordinate in range [0, 1]
    ///   - depth: Distance along the ray in meters (default: 1.0)
    /// - Returns: The new anchor ID, or `nil` if placement failed
    public func placeAnchorAtScreenPoint(
        normalizedU: Double,
        normalizedV: Double,
        depth: Double = 1.0
    ) -> UInt64? {
        var anchorId: UInt64 = 0
        let success = ffi_place_anchor_at_screen_point(
            handle,
            normalizedU,
            normalizedV,
            depth,
            &anchorId
        )
        return success ? anchorId : nil
    }
    
    /// Get all anchors visible in the current camera frame with their projected screen coordinates
    /// - Parameter maxCount: Maximum number of anchors to return (default: 32)
    /// - Returns: Array of projected anchors that are visible (in front of camera and within frame bounds)
    public func getVisibleAnchors(maxCount: Int = 32) -> [ProjectedAnchor] {
        var buffer = [ArbitFFI.ArbitProjectedAnchor](
            repeating: ArbitFFI.ArbitProjectedAnchor(),
            count: maxCount
        )
        let written = ffi_get_visible_anchors(handle, &buffer, maxCount)
        return buffer.prefix(written).map { ProjectedAnchor(ffiValue: $0) }
    }
    
    /// Get all visible landmarks in the current camera frame (for debugging visualization)
    /// - Parameter maxCount: Maximum number of landmarks to return (default: 200)
    /// - Returns: Array of projected landmarks with 3D world positions and screen coordinates
    public func getVisibleLandmarks(maxCount: Int = 200) -> [ProjectedLandmark] {
        var buffer = [ArbitFFI.ArbitProjectedLandmark](
            repeating: ArbitFFI.ArbitProjectedLandmark(),
            count: maxCount
        )
        let written = ffi_get_visible_landmarks(handle, &buffer, maxCount)
        return buffer.prefix(written).map { ProjectedLandmark(ffiValue: $0) }
    }
    
    /// Get a debug snapshot of the current map state
    /// - Returns: Snapshot containing camera pose and map statistics
    public func getMapDebugSnapshot() -> MapDebugSnapshot? {
        var snapshot = ArbitFFI.ArbitMapDebugSnapshot()
        if ffi_get_map_debug_snapshot(handle, &snapshot) {
            return MapDebugSnapshot(ffiValue: snapshot)
        }
        return nil
    }

    public func saveMap() throws -> Data {
        var required = 0
        let query = ffi_save_map(handle, nil, 0, &required)
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
            return ffi_save_map(handle, baseAddress, required, &required)
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
            return ffi_load_map(handle, baseAddress, buffer.count)
        }
        guard loaded else {
            throw ArbitCaptureError.mapLoadFailed
        }
    }
}

#if canImport(ARKit) && !os(macOS)
@available(iOS 11.0, *)
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

@available(iOS 11.0, *)
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
