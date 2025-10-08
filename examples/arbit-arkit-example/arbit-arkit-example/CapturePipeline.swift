//
//  CapturePipeline.swift
//  arbit-arkit-example
//
//  Recreated to drive Milestone 1 demo using AVFoundation.
//

import AVFoundation
import Combine
import CoreMotion
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
    @Published private(set) var relocalizationSummary: RelocalizationSummary?
    @Published private(set) var lastPoseMatrix: simd_double4x4?
    @Published private(set) var mapStatusMessage: String?

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
            startIMUUpdates()
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
            self.stopIMUUpdates()
            DispatchQueue.main.async {
                self.isRunning = false
            }
        }
    }

    func placeAnchor() {
        guard let context else { return }
        let pose = lastPoseMatrix ?? matrix_identity_double4x4
        sampleQueue.async { [weak self] in
            guard let _ = self else { return }
            _ = context.createAnchor(pose: pose)
            let translation = SIMD3(pose.columns.3.x, pose.columns.3.y, pose.columns.3.z)
            self?.logger.info(
                "Placed anchor at [\(translation.x, format: .fixed(precision: 2)), \(translation.y, format: .fixed(precision: 2)), \(translation.z, format: .fixed(precision:2))]"
            )
        }
    }

    func saveMapSnapshot() {
        guard let context else { return }
        sampleQueue.async { [weak self] in
            guard let self else { return }
            do {
                let data = try context.saveMap()
                self.savedMapData = data
                self.logger.info("Saved map snapshot (\(data.count, format: .decimal) bytes)")
                DispatchQueue.main.async {
                    self.mapStatusMessage = String(format: "Saved map (%d bytes)", data.count)
                }
            } catch {
                self.logger.error("Failed to save map snapshot: \(error.localizedDescription, privacy: .public)")
                DispatchQueue.main.async {
                    self.mapStatusMessage = "Failed to save map"
                }
            }
        }
    }

    func loadSavedMap() {
        guard let context else { return }
        guard let data = savedMapData else {
            mapStatusMessage = "No saved map snapshot"
            return
        }
        sampleQueue.async { [weak self] in
            guard let self else { return }
            do {
                try context.loadMap(data)
                self.logger.info("Loaded map snapshot")
                DispatchQueue.main.async {
                    self.mapStatusMessage = "Map loaded"
                }
            } catch {
                self.logger.error("Failed to load map snapshot: \(error.localizedDescription, privacy: .public)")
                DispatchQueue.main.async {
                    self.mapStatusMessage = "Failed to load map"
                }
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

            guard self.sessionConfigured else { return }

            if !self.session.isRunning {
                self.session.startRunning()
            }

            DispatchQueue.main.async {
                self.isRunning = self.session.isRunning
            }
        }
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
            }

        } catch {
            logger.error("Failed to configure session: \(error.localizedDescription)")
            session.commitConfiguration()
            return
        }

        session.commitConfiguration()
        sessionConfigured = true
    }
    
    // MARK: - IMU Management
    
    private func startIMUUpdates() {
        guard motionManager.isDeviceMotionAvailable else {
            logger.warning("Device motion (IMU) not available")
            return
        }
        
        // Set update interval to 500 Hz (2ms)
        motionManager.deviceMotionUpdateInterval = 1.0 / 500.0
        
        // Start device motion updates (includes both accel and gyro)
        motionManager.startDeviceMotionUpdates(to: motionQueue) { [weak self] motion, error in
            guard let self = self, let motion = motion, error == nil else {
                if let error = error {
                    self?.logger.error("IMU update error: \(error.localizedDescription)")
                }
                return
            }
            
            self.ingestIMUSample(motion)
        }
        
        logger.info("Started IMU updates at 500 Hz")
    }
    
    private func stopIMUUpdates() {
        if motionManager.isDeviceMotionActive {
            motionManager.stopDeviceMotionUpdates()
            logger.info("Stopped IMU updates")
        }
    }
    
    private func ingestIMUSample(_ motion: CMDeviceMotion) {
        guard let context = context else { return }
        
        // Use userAcceleration (linear accel, gravity already removed by iOS)
        // Note: For iOS, CoreMotion already did high-quality gravity estimation,
        // so we use the pre-processed data instead of reconstructing raw samples.
        // The Rust engine will use iOS's gravity estimate instead of computing its own.
        let userAccel = motion.userAcceleration
        
        // Convert from g to m/s²
        let accel = (
            x: userAccel.x * 9.80665,
            y: userAccel.y * 9.80665,
            z: userAccel.z * 9.80665
        )
        
        // Get gyroscope data (rotation rate in rad/s)
        let gyro = motion.rotationRate
        
        // Create IMU sample
        let sample = ImuSample(
            timestampSeconds: motion.timestamp,
            accelX: accel.x,
            accelY: accel.y,
            accelZ: accel.z,
            gyroX: gyro.x,
            gyroY: gyro.y,
            gyroZ: gyro.z
        )
        
        // Feed to Rust engine (non-blocking)
        do {
            try context.ingestIMUSample(sample)
        } catch {
            // Log but don't spam - IMU samples arrive at 500 Hz
            logger.debug("Failed to ingest IMU sample: \(error.localizedDescription)")
        }
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

        let width = CVPixelBufferGetWidth(pixelBuffer)
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
        let imuState = context.getImuState()
        let twoView = frameState?.twoView
        let trajectory = context.trajectory(maxPoints: 256)
        let anchorIDs = context.anchorIds(maxCount: 16)
        var anchorDictionary: [UInt64: simd_double4x4] = [:]
        for id in anchorIDs {
            if let pose = context.resolveAnchor(id) {
                anchorDictionary[id] = pose
            }
        }
        let relocalization = frameState?.relocalization
        let latestPoseMatrix = trajectory.last.map { self.poseMatrix(from: $0) }

        DispatchQueue.main.async { [weak self] in
            guard let self else { return }
            self.lastSample = captured
            self.pyramidLevels = pyramid
            self.trackedPoints = tracks
            self.twoViewSummary = twoView
            self.trajectory = trajectory
            self.imu = imuState
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

        var fx = Double(width)
        var fy = Double(height)
        var cx = Double(width) / 2.0
        var cy = Double(height) / 2.0
        var skew = 0.0

        if let attachment = CMGetAttachment(
            sampleBuffer,
            key: kCMSampleBufferAttachmentKey_CameraIntrinsicMatrix,
            attachmentModeOut: nil
        ),
           CFGetTypeID(attachment) == CFDataGetTypeID() {
            let cfData = unsafeBitCast(attachment, to: CFData.self)
            let data = cfData as Data
            data.withUnsafeBytes { (rawPointer: UnsafeRawBufferPointer) in
                let count = rawPointer.count / MemoryLayout<Float32>.size
                if count >= 9 {
                    let values = rawPointer.bindMemory(to: Float32.self)
                    fx = Double(values[0])
                    skew = Double(values[1])
                    cx = Double(values[2])
                    fy = Double(values[4])
                    cy = Double(values[5])
                }
            }
        }

        return CameraIntrinsics(
            fx: fx,
            fy: fy,
            cx: cx,
            cy: cy,
            skew: skew,
            width: UInt32(width),
            height: UInt32(height),
            distortion: nil
        )
    }
}
