//
//  CapturePipeline.swift
//  arbit-arkit-example
//
//  Recreated to drive Milestone 1 demo using AVFoundation.
//

import AVFoundation
import Combine
import Foundation
import os.log
import arbit_swift_lib

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

    let session = AVCaptureSession()

    private let sessionQueue = DispatchQueue(label: "arbit.camera.session")
    private let sampleQueue = DispatchQueue(label: "arbit.camera.frames")
    private let logger = Logger(subsystem: "arbit.camera", category: "capture")

    private var context: ArbitCaptureContext?
    private var sessionConfigured = false
    private var frameCounter: Int = 0

    override init() {
        authorizationStatus = AVCaptureDevice.authorizationStatus(for: .video)
        super.init()
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
                if connection.isVideoOrientationSupported {
                    connection.videoOrientation = .portrait
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

        let sample: CameraSample
        do {
            sample = try context.ingest(frame)
        } catch {
            logger.error("Failed to ingest frame into Arbit core: \(error.localizedDescription)")
            return
        }

        frameCounter &+= 1
        let summary = IntrinsicsSummary(
            fx: sample.intrinsics.fx,
            fy: sample.intrinsics.fy,
            cx: sample.intrinsics.cx,
            cy: sample.intrinsics.cy,
            skew: sample.intrinsics.skew,
            width: Int(sample.intrinsics.width),
            height: Int(sample.intrinsics.height)
        )

        let captured = CapturedSample(
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

        DispatchQueue.main.async { [weak self] in
            self?.lastSample = captured
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
