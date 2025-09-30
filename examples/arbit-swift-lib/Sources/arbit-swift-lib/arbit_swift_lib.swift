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

@_silgen_name("arbit_capture_context_new")
private func ffi_capture_context_new() -> OpaquePointer?

@_silgen_name("arbit_capture_context_free")
private func ffi_capture_context_free(_ handle: OpaquePointer?)

@_silgen_name("arbit_ingest_camera_frame")
private func ffi_ingest_camera_frame(
    _ handle: OpaquePointer?,
    _ frame: UnsafePointer<ArbitCameraFrame>?,
    _ sample: UnsafeMutablePointer<ArbitCameraSample>?
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

public enum ArbitCaptureError: Error {
    case allocationFailed
    case ingestionFailed
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
