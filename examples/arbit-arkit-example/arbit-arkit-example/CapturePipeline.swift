//
//  CapturePipeline.swift
//  arbit-arkit-example
//
//  Recreated to drive Milestone 1 demo using AVFoundation.
//

import AVFoundation
import Combine
import CoreMotion
import CoreVideo
import Foundation
import os.log
import simd
import Arbit

struct IntrinsicsSummary {
    let fx: Double
    let fy: Double
    let cx: Double
    let cy: Double
    let skew: Double
    let width: Int
    let height: Int
}

struct CapturedSample: Identifiable {
    let id = UUID()
    let captureSeconds: Double
    let pipelineSeconds: Double
    let latencySeconds: Double
    let rawTimestampSeconds: Double
    let intrinsics: IntrinsicsSummary
    let pixelFormat: CameraPixelFormat
    let bytesPerRow: Int
    let frameIndex: Int
    let receivedAt: Date
}

final class CameraCaptureManager: NSObject, ObservableObject {
    @Published private(set) var authorizationStatus: AVAuthorizationStatus
    @Published private(set) var lastSample: CapturedSample?
    @Published private(set) var isRunning: Bool = false
    @Published private(set) var pyramidLevels: [PyramidLevelView] = []
    @Published private(set) var trackedPoints: [TrackedPoint] = []
    @Published private(set) var twoViewSummary: TwoViewSummary?
    @Published private(set) var trajectory: [PoseSample] = []
    @Published private(set) var imu: ImuState?
    @Published private(set) var mapStats: MapStats = MapStats(keyframes: 0, landmarks: 0, anchors: 0)
    @Published private(set) var anchorPoses: [UInt64: simd_double4x4] = [:]
//    @Published private(set) var visibleAnchors: [ProjectedAnchor] = []
//    @Published private(set) var visibleLandmarks: [ProjectedLandmark] = []
//    @Published private(set) var mapDebugSnapshot: MapDebugSnapshot?
    @Published private(set) var relocalizationSummary: RelocalizationSummary?
    @Published private(set) var lastPoseMatrix: simd_double4x4?
    @Published private(set) var mapStatusMessage: String?
    private var deviceIntrinsics: CameraIntrinsics?

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "arbit.camera.session")
    private let sampleQueue = DispatchQueue(label: "arbit.camera.frames")
    private let logger = Logger(subsystem: "arbit.camera", category: "capture")

    private var context: ArbitCaptureContext?
    private var sessionConfigured = false
    private var frameCounter: Int = 0
    private var savedMapData: Data?
    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()

    var captureContext: ArbitCaptureContext? {
        context
    }

    override init() {
        authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
        super.init()

        // Initialize Rust logging with debug level
        ArbitCaptureContext.initLogging()

        do {
            context = try ArbitCaptureContext()
        } catch {
            logger.error("Failed to initialize Arbit capture context: \(error.localizedDescription)")
            context = nil
        }
    }

    func start() {
        switch authorizationStatus {
        case .authorized:
            configureAndStartSession()
        case .notDetermined:
            requestAuthorization()
        case .denied, .restricted:
            logger.error("Camera access denied or restricted")
        @unknown default:
            logger.error("Unhandled authorization status")
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            guard let self else { return }
            if self.session.isRunning {
                self.session.stopRunning()
            }
            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }

    private func poseMatrix(from sample: PoseSample) -> simd_double4x4 {
        var matrix = matrix_identity_double4x4
        matrix.columns.3 = SIMD4(sample.position.x, sample.position.y, sample.position.z, 1.0)
        return matrix
    }

    private func requestAuthorization() {
        AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
            DispatchQueue.main.async {
                guard let self else { return }
                self.authorizationStatus = granted ? .authorized : .denied
                if granted {
                    self.configureAndStartSession()
                } else {
                    self.logger.error("Camera permission not granted")
                }
            }
        }
    }

    private func configureAndStartSession() {
        sessionQueue.async { [weak self] in
            guard let self else { return }

            if !self.sessionConfigured {
                self.configureSession()
            }

            self.deviceIntrinsics = self.getDeviceIntrinsics()

            guard self.sessionConfigured else { return }

            if !self.session.isRunning {
                self.session.startRunning()
            }

            DispatchQueue.main.async {
                self.isRunning = self.session.isRunning
            }
        }
    }

    private func getDeviceIntrinsics() -> CameraIntrinsics? {
        print("DEBUG: getDeviceIntrinsics() called")
        
        guard let videoDevice = AVCaptureDevice.default(for: .video) else {
            print("WARNING: No back camera device found")
            return nil
        }
        
        print("DEBUG: Using camera device: \(videoDevice.localizedName)")
        print("DEBUG: Camera device type: \(videoDevice.deviceType.rawValue)")

        let format = videoDevice.activeFormat
        let dimensions = CMVideoFormatDescriptionGetDimensions(format.formatDescription)
        let width = Int(dimensions.width)
        let height = Int(dimensions.height)

        print("DEBUG: Format dimensions: \(width) x \(height)")
        print("DEBUG: Video field of view: \(format.videoFieldOfView)°")
        
        // Try to get extensions - bridge from Core Foundation to Swift
        if let extensionsCF = CMFormatDescriptionGetExtensions(format.formatDescription) {
            print("DEBUG: Got CF extensions")
            
            // Bridge CFDictionary to NSDictionary to Swift Dictionary
            let extensionsNS = extensionsCF as NSDictionary
            if let extensions = extensionsNS as? [String: Any] {
                print("DEBUG: Successfully bridged to [String: Any]")
                print("DEBUG: Available extension keys: \(extensions.keys.sorted())")
            
                if let data = extensions["CameraIntrinsicMatrix"] as? Data {
                    return data.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> CameraIntrinsics? in
                        // The matrix is stored as 9 floats in column-major order
                        let floats = ptr.bindMemory(to: Float.self)
                        guard floats.count >= 9 else {
                            print("WARNING: Insufficient matrix data")
                            return nil
                        }
                        
                        // Camera intrinsic matrix (column-major):
                        // [fx,  0,  0,  0, fy,  0, cx, cy,  1]
                        //  0   1   2   3   4   5   6   7   8
                        let fx = Double(floats[0])
                        let fy = Double(floats[4])
                        let cx = Double(floats[6])
                        let cy = Double(floats[7])
                        
                        print("DEBUG: Device intrinsics extracted - fx:\(fx), fy:\(fy), cx:\(cx), cy:\(cy)")
                        
                        // Validate intrinsics
                        guard fx > 1.0 && fy > 1.0 else {
                            print("WARNING: Invalid focal lengths - fx:\(fx), fy:\(fy)")
                            return nil
                        }
                        
                        return CameraIntrinsics(
                            fx: fx,
                            fy: fy,
                            cx: cx,
                            cy: cy,
                            skew: 0.0,
                            width: UInt32(width),
                            height: UInt32(height),
                            distortion: nil
                        )
                    }
                }
            } else {
                print("WARNING: Failed to cast CF extensions to [String: Any]")
            }
        } else {
            print("WARNING: CMFormatDescriptionGetExtensions returned nil")
        }
        
        // Fallback: compute from field of view (works even without extensions)
        let fovX = format.videoFieldOfView
        print("DEBUG: Video field of view: \(fovX)°")
        
        if fovX > 0 {
            let fovRadians = Double(fovX) * .pi / 180.0
            let fx = Double(width) / (2.0 * tan(fovRadians / 2.0))
            let fy = fx // Assume square pixels
            
            print("DEBUG: Computed intrinsics from FOV(\(fovX)°) - fx:\(fx), fy:\(fy), cx:\(Double(width) / 2.0), cy:\(Double(height) / 2.0)")
            print("DEBUG: Final intrinsics - width:\(width), height:\(height)")
            
            return CameraIntrinsics(
                fx: fx,
                fy: fy,
                cx: Double(width) / 2.0,
                cy: Double(height) / 2.0,
                skew: 0.0,
                width: UInt32(width),
                height: UInt32(height),
                distortion: nil
            )
        } else {
            print("WARNING: FOV is zero or negative: \(fovX)")
        }

        print("WARNING: No camera intrinsic matrix found and no valid FOV")
        return nil
}

    private func configureSession() {
        session.beginConfiguration()
        session.sessionPreset = .hd1280x720

        do {
            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back) else {
                logger.error("No back camera available")
                session.commitConfiguration()
                return
            }

            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }

            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
            ]
            output.alwaysDiscardsLateVideoFrames = false
            output.setSampleBufferDelegate(self, queue: sampleQueue)

            if session.canAddOutput(output) {
                session.addOutput(output)
            }

            if let connection = output.connection(with: .video) {
                if connection.isCameraIntrinsicMatrixDeliverySupported {
                    connection.isCameraIntrinsicMatrixDeliveryEnabled = true
                }
                if connection.isVideoRotationAngleSupported(90) {
                    connection.videoRotationAngle = 90
                }
                // No rotation needed for default camera - it should match preview orientation
            }

        } catch {
            logger.error("Failed to configure session: \(error.localizedDescription)")
            session.commitConfiguration()
            return
        }

        session.commitConfiguration()
        sessionConfigured = true
    }
    
}

extension CameraCaptureManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let context = context else { return }
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }

        guard let baseAddress = CVPixelBufferGetBaseAddress(pixelBuffer) else { return }

        let height = CVPixelBufferGetHeight(pixelBuffer)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
        let dataLength = bytesPerRow * height

        let timestampSeconds = CMSampleBufferGetPresentationTimeStamp(sampleBuffer).seconds
        let intrinsics = makeIntrinsics(from: sampleBuffer, pixelBuffer: pixelBuffer)
        let frameData = Data(bytes: baseAddress, count: dataLength)

        let frame = CameraFrame(
            timestamp: timestampSeconds,
            intrinsics: intrinsics,
            pixelFormat: .bgra8,
            bytesPerRow: bytesPerRow,
            data: frameData
        )

        var sample: CameraSample?
        do {
            // Ingest the camera frame (IMU preintegration is finished automatically)
            sample = try context.ingest(frame)
        } catch {
            logger.error("Failed to ingest frame into Arbit core: \(error.localizedDescription)")
            // Continue processing - don't return early as this might be a temporary issue
            // The pyramid and tracking will be empty for this frame but processing continues
        }

        frameCounter &+= 1

        // Create sample data even if ingestion failed (for UI consistency)
        let summary: IntrinsicsSummary
        let captured: CapturedSample

        if let sample = sample {
            summary = IntrinsicsSummary(
                fx: sample.intrinsics.fx,
                fy: sample.intrinsics.fy,
                cx: sample.intrinsics.cx,
                cy: sample.intrinsics.cy,
                skew: sample.intrinsics.skew,
                width: Int(sample.intrinsics.width),
                height: Int(sample.intrinsics.height)
            )

            captured = CapturedSample(
                captureSeconds: sample.timestamps.capture,
                pipelineSeconds: sample.timestamps.pipeline,
                latencySeconds: sample.timestamps.latency,
                rawTimestampSeconds: timestampSeconds,
                intrinsics: summary,
                pixelFormat: sample.pixelFormat,
                bytesPerRow: sample.bytesPerRow,
                frameIndex: frameCounter,
                receivedAt: Date()
            )
        } else {
            // Fallback sample when ingestion fails
            let fallbackIntrinsics = makeIntrinsics(from: sampleBuffer, pixelBuffer: pixelBuffer)
            summary = IntrinsicsSummary(
                fx: fallbackIntrinsics.fx,
                fy: fallbackIntrinsics.fy,
                cx: fallbackIntrinsics.cx,
                cy: fallbackIntrinsics.cy,
                skew: fallbackIntrinsics.skew,
                width: Int(fallbackIntrinsics.width),
                height: Int(fallbackIntrinsics.height)
            )
            captured = CapturedSample(
                captureSeconds: timestampSeconds,
                pipelineSeconds: timestampSeconds + 0.001, // Small offset to show processing happened
                latencySeconds: 0.001,
                rawTimestampSeconds: timestampSeconds,
                intrinsics: summary,
                pixelFormat: .bgra8,
                bytesPerRow: bytesPerRow,
                frameIndex: frameCounter,
                receivedAt: Date()
            )
        }

        let pyramid = context.pyramidLevels(maxLevels: 3)
        let tracks = context.trackedPoints(maxPoints: 200)
        let frameState = context.getFrameState()
        let twoView = frameState?.twoView
        let trajectory = context.trajectory(maxPoints: 256)
        var anchorDictionary: [UInt64: simd_double4x4] = [:]
        let relocalization = frameState?.relocalization
        let latestPoseMatrix = trajectory.last.map { self.poseMatrix(from: $0) }
        
        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lastSample = captured
            self.pyramidLevels = pyramid
            self.trackedPoints = tracks
            self.twoViewSummary = twoView
            self.trajectory = trajectory
            self.mapStats = MapStats(keyframes: frameState?.keyframeCount ?? 0,
                                     landmarks: frameState?.landmarkCount ?? 0,
                                     anchors: frameState?.anchorCount ?? 0)
            self.anchorPoses = anchorDictionary
            self.relocalizationSummary = relocalization
            self.lastPoseMatrix = latestPoseMatrix
        }
    }

    private func makeIntrinsics(from sampleBuffer: CMSampleBuffer, pixelBuffer: CVPixelBuffer) -> CameraIntrinsics {
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // PRIORITY 1: Try to extract from sample buffer attachment (most accurate - reflects actual capture)
        // This works because we enabled isCameraIntrinsicMatrixDeliveryEnabled in configureSession()
        
        if let attachment = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
            attachmentModeOut: nil
        ) as? Data {
            
            let intrinsics = attachment.withUnsafeBytes { (ptr: UnsafeRawBufferPointer) -> CameraIntrinsics? in
                let floats = ptr.bindMemory(to: Float.self)
                guard floats.count >= 9 else { return nil }
                
                // Column-major: [fx, 0, 0, 0, fy, 0, cx, cy, 1]
                let fx = Double(floats[0])
                let fy = Double(floats[4])
                let cx = Double(floats[6])
                let cy = Double(floats[7])
                
                guard fx > 1.0 && fy > 1.0 else { return nil }
                
                return CameraIntrinsics(
                    fx: fx,
                    fy: fy,
                    cx: cx,
                    cy: cy,
                    skew: 0.0,
                    width: UInt32(width),
                    height: UInt32(height),
                    distortion: nil
                )
            }
            
            if let intrinsics = intrinsics {
                print("DEBUG: ✓ Using sample buffer intrinsics - fx:\(intrinsics.fx), fy:\(intrinsics.fy)")
                return intrinsics
            }
        }
        
        // PRIORITY 2: Use cached device intrinsics, adjusting for 90° rotation
        if let cached = deviceIntrinsics {
            // Check if dimensions are swapped (90° rotation applied)
            if Int(cached.width) == height && Int(cached.height) == width {
                // Apply 90° clockwise rotation transformation to intrinsics
                // For 90° rotation: swap fx/fy and transform principal point
                let rotatedCx = cached.cy
                let rotatedCy = Double(cached.width) - cached.cx
                
                print("DEBUG: ✓ Using cached intrinsics with 90° rotation")
                print("DEBUG:   Before: \(cached.width)×\(cached.height), fx:\(cached.fx), fy:\(cached.fy), cx:\(cached.cx), cy:\(cached.cy)")
                print("DEBUG:   After:  \(width)×\(height), fx:\(cached.fy), fy:\(cached.fx), cx:\(rotatedCx), cy:\(rotatedCy)")
                
                return CameraIntrinsics(
                    fx: cached.fy,        // Swap
                    fy: cached.fx,        // Swap
                    cx: rotatedCx,        // Transform
                    cy: rotatedCy,        // Transform
                    skew: 0.0,
                    width: UInt32(width),
                    height: UInt32(height),
                    distortion: nil)
            } else {
                // Dimensions match exactly - no rotation needed
                print("DEBUG: ✓ Using cached device intrinsics (no rotation)")
                return cached
            }
        }
        
        // PRIORITY 3: Ultimate fallback - approximate based on typical iPhone FOV
        print("WARNING: Using fallback intrinsics (approximate 60° FOV)")
        let fx = Double(width) * 1.2
        let fy = Double(height) * 1.2
        return CameraIntrinsics(
            fx: fx,
            fy: fy,
            cx: Double(width) / 2.0,
            cy: Double(height) / 2.0,
            skew: 0.0,
            width: UInt32(width),
            height: UInt32(height),
            distortion: nil
        )
    }
}

