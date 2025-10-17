//
//  ContentView.swift
//  arbit-arkit-example
//
//  Milestones 1–3 interactive overlays plus placeholders for 4–7.
//

import SwiftUI
import Combine
import SceneKit
import AVFoundation
import CoreGraphics
import simd
import Arbit

private enum Milestone: Int, CaseIterable, Identifiable {
    case m1 = 1, m2, m3, m4, m5, m6, m7, m8

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .m1: "M1 Axes"
        case .m2: "M2 Pyramid"
        case .m3: "M3 Gravity"
        case .m4: "M4 Tracker"
        case .m5: "M5 Init"
        case .m6: "M6 VO"
        case .m7: "M7 Reloc"
        case .m8: "M8 Anchors"
        }
    }

    var headline: String {
        switch self {
        case .m1: "Milestone 1 — Timestamp HUD"
        case .m2: "Milestone 2 — Pyramid & Back-Projection"
        case .m3: "Milestone 3 — Gravity Arrow"
        case .m4: "Milestone 4 — Point Tracker"
        case .m5: "Milestone 5 — Two-View Init"
        case .m6: "Milestone 6 — VO Loop"
        case .m7: "Milestone 7 — Relocalization"
        case .m8: "Milestone 8 — Anchors"
        }
    }
}

struct ContentView: View {
    @StateObject private var cameraManager = CameraCaptureManager()
    @StateObject private var orientationProvider = DeviceOrientationProvider()

    @State private var selectedMilestone: Milestone = .m1
    @State private var previousPipelineSeconds: Double?
    @State private var estimatedFPS: Double?
    @State private var backProjection: BackProjectionResult?

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            if cameraManager.authorizationStatus != .authorized {
                PermissionOverlay()
            }

            if selectedMilestone == .m2 {
                MilestoneTwoTouchOverlay(cameraManager: cameraManager, result: $backProjection)
            }

            if selectedMilestone == .m4 {
                TrackedPointsOverlay(
                    trackedPoints: cameraManager.trackedPoints,
                    intrinsics: cameraManager.lastSample?.intrinsics
                )
                .allowsHitTesting(false)
            }

            VStack {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 12) {
                        MilestoneSelector(selectedMilestone: $selectedMilestone)
                        topLeftPanel
                    }
                    Spacer()
                    topRightPanel
                }
                Spacer()
                footerPanel
            }
            .padding()
        }
        .onAppear {
            cameraManager.start()
            orientationProvider.start()
        }
        .onDisappear {
            cameraManager.stop()
            orientationProvider.onAccelerometer = nil
            orientationProvider.stop()
        }
        .onReceive(cameraManager.$lastSample.compactMap { $0 }) { sample in
            if let previous = previousPipelineSeconds {
                let delta = sample.pipelineSeconds - previous
                if delta > 0 {
                    estimatedFPS = 1.0 / delta
                }
            }
            previousPipelineSeconds = sample.pipelineSeconds
        }
    }

    @ViewBuilder
    private var topLeftPanel: some View {
        switch selectedMilestone {
        case .m1:
            MilestoneOnePanel(sample: cameraManager.lastSample, estimatedFPS: estimatedFPS)
        case .m2:
            MilestoneTwoPanel(
                sample: cameraManager.lastSample,
                levels: cameraManager.pyramidLevels,
                projection: backProjection
            )
        case .m3:
            MilestoneThreePanel(
                orientation: orientationProvider.deviceToWorld,
                gravityDown: cameraManager.imu?.gravityDown
            )
        case .m4:
            MilestoneFourPanel(trackedPoints: cameraManager.trackedPoints)
        case .m5:
            MilestoneFivePanel(summary: cameraManager.twoViewSummary)
        case .m6:
            MilestoneSixPanel(trajectory: cameraManager.trajectory)
        case .m7:
            MilestoneSevenPanel(
                stats: cameraManager.mapStats,
                summary: cameraManager.relocalizationSummary
            )
        case .m8:
            MilestoneEightPanel(
                stats: cameraManager.mapStats,
                anchors: cameraManager.anchorPoses,
                lastPose: cameraManager.lastPoseMatrix,
                placeAnchor: cameraManager.placeAnchor,
                statusMessage: cameraManager.mapStatusMessage,
                saveMap: cameraManager.saveMapSnapshot,
                loadMap: cameraManager.loadSavedMap
            )
        }
    }

    @ViewBuilder
    private var topRightPanel: some View {
        switch selectedMilestone {
        case .m1:
            AxesSceneView(orientation: orientationProvider.deviceToWorld)
                .frame(width: 180, height: 180)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
        case .m3:
            AxesSceneView(orientation: orientationProvider.deviceToWorld)
                .frame(width: 160, height: 160)
                .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
        default:
            EmptyView()
        }
    }

    private var footerPanel: some View {
        HStack {
            Spacer()
            Text(cameraManager.isRunning ? "Session Running" : "Session Idle")
                .font(.caption)
                .foregroundStyle(cameraManager.isRunning ? .green : .yellow)
                .padding(10)
                .background(.black.opacity(0.55), in: Capsule())
        }
    }
}

private func formatDouble(_ value: Double, decimals: Int = 3) -> String {
    String(format: "%0.*f", decimals, value)
}

private func translationVector(from matrix: simd_double4x4) -> SIMD3<Double> {
    let column = matrix.columns.3
    return SIMD3(column.x, column.y, column.z)
}

private struct PermissionOverlay: View {
    var body: some View {
        ZStack {
            Color.black.opacity(0.7)
                .ignoresSafeArea()
            Text("Camera access is required to run the demo.")
                .font(.headline)
                .foregroundStyle(.white)
                .padding()
        }
    }
}

private struct MilestoneSelector: View {
    @Binding var selectedMilestone: Milestone

    var body: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 8) {
                ForEach(Milestone.allCases) { milestone in
                    Button(action: { selectedMilestone = milestone }) {
                        Text(milestone.label)
                            .font(.caption)
                            .padding(.vertical, 6)
                            .padding(.horizontal, 12)
                            .background(
                                RoundedRectangle(cornerRadius: 12)
                                    .fill(selectedMilestone == milestone ? Color.blue.opacity(0.85) : Color.black.opacity(0.6))
                            )
                            .overlay(
                                RoundedRectangle(cornerRadius: 12)
                                    .stroke(Color.white.opacity(0.3), lineWidth: 1)
                            )
                            .foregroundStyle(.white)
                    }
                }
            }
        }
        .padding(8)
        .background(.black.opacity(0.45), in: RoundedRectangle(cornerRadius: 14))
    }
}

private struct MilestoneOnePanel: View {
    let sample: CapturedSample?
    let estimatedFPS: Double?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Milestone 1 — Timestamp HUD")
                .font(.headline)
                .padding(.bottom, 4)

            if let sample {
                Text("Capture (s): \(formatDouble(sample.captureSeconds))")
                Text("Pipeline (s): \(formatDouble(sample.pipelineSeconds))")
                Text(String(format: "Latency (ms): %.2f", sample.latencySeconds * 1_000))
                Text("Frame #: \(sample.frameIndex)")
                if let fps = estimatedFPS {
                    Text(String(format: "Approx FPS: %.1f", fps))
                }
                Text("Intrinsics fx/fy: \(formatDouble(sample.intrinsics.fx)), \(formatDouble(sample.intrinsics.fy))")
                Text("Principal cx/cy: \(formatDouble(sample.intrinsics.cx)), \(formatDouble(sample.intrinsics.cy))")
                Text("Resolution: \(sample.intrinsics.width) × \(sample.intrinsics.height)")
            } else {
                Text("Waiting for camera frames…")
            }
        }
        .font(.system(.body, design: .monospaced))
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct BackProjectionResult {
    let tapPoint: CGPoint
    let pixel: CGPoint
    let ray: SIMD3<Double>
}

private struct MilestoneTwoPanel: View {
    let sample: CapturedSample?
    let levels: [PyramidLevelView]
    let projection: BackProjectionResult?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Milestone 2 — Pyramid & Back-Projection")
                .font(.headline)

            if levels.isEmpty {
                Text("Waiting for frame pyramid…")
            } else {
                Text("Levels: \(levels.count)")
                HStack(spacing: 8) {
                    ForEach(Array(levels.prefix(3).enumerated()), id: \.0) { index, level in
                        if let image = makeImage(from: level) {
                            let referenceWidth = CGFloat(levels.first?.width ?? level.width)
                            let ratio = max(CGFloat(level.width) / max(referenceWidth, 1), 0.05)
                            PyramidLevelThumbnail(
                                cgImage: image,
                                label: "L\(index)",
                                scale: ratio
                            )
                        }
                    }
                }
            }

            if let projection {
                VStack(alignment: .leading, spacing: 4) {
                    Text(String(format: "Pixel: (%.0f, %.0f)", projection.pixel.x, projection.pixel.y))
                    Text(String(format: "NDC: (%.3f, %.3f)", projection.ray.x, projection.ray.y))
                    Text(String(format: "Ray: [%.3f, %.3f, %.3f]", projection.ray.x, projection.ray.y, projection.ray.z))
                }
                .font(.system(.footnote, design: .monospaced))
            } else {
                Text("Tap the preview to inspect a ray.")
                    .font(.system(.footnote, design: .monospaced))
            }
        }
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }

    private func makeImage(from level: PyramidLevelView) -> CGImage? {
        guard let provider = CGDataProvider(data: level.data as CFData) else {
            return nil
        }

        return CGImage(
            width: level.width,
            height: level.height,
            bitsPerComponent: 8,
            bitsPerPixel: 8,
            bytesPerRow: level.bytesPerRow,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: true,
            intent: .defaultIntent
        )
    }
}

private struct PyramidLevelThumbnail: View {
    let cgImage: CGImage
    let label: String
    let scale: CGFloat

    var body: some View {
        VStack {
            Image(decorative: cgImage, scale: 1.0, orientation: .up)
                .resizable()
                .aspectRatio(contentMode: .fit)
                .frame(width: 140 * scale, height: 90 * scale)
                .blur(radius: scale == 1.0 ? 0 : Double(1.5 / max(scale, 0.01)))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.white.opacity(0.3), lineWidth: 1)
                )
            Text(label)
                .font(.caption2)
        }
    }
}

private struct MilestoneThreePanel: View {
    let orientation: simd_quatf
    let gravityDown: SIMD3<Double>?

    private var deviceDown: simd_float3 {
        if let gravityDown {
            return simd_float3(
                Float(gravityDown.x),
                Float(gravityDown.y),
                Float(gravityDown.z)
            )
        }
        return orientation.inverse.act(simd_float3(0, -1, 0))
    }

    private var eulerAngles: (roll: Double, pitch: Double, yaw: Double) {
        let q = orientation
        let qw = Double(q.real)
        let qx = Double(q.imag.x)
        let qy = Double(q.imag.y)
        let qz = Double(q.imag.z)

        let sinr = 2.0 * (qw * qx + qy * qz)
        let cosr = 1.0 - 2.0 * (qx * qx + qy * qy)
        let roll = atan2(sinr, cosr)

        let sinp = 2.0 * (qw * qy - qz * qx)
        let pitch: Double
        if abs(sinp) >= 1.0 {
            pitch = copysign(Double.pi / 2.0, sinp)
        } else {
            pitch = asin(sinp)
        }

        let siny = 2.0 * (qw * qz + qx * qy)
        let cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
        let yaw = atan2(siny, cosy)

        return (roll, pitch, yaw)
    }

    var body: some View {
        HStack(alignment: .center, spacing: 12) {
            GravityIndicatorView(downVector: deviceDown)
                .frame(width: 140, height: 140)

            VStack(alignment: .leading, spacing: 4) {
                Text("Milestone 3 — Gravity Arrow")
                    .font(.headline)
                Text(String(format: "Down (device): [%.2f, %.2f, %.2f]", deviceDown.x, deviceDown.y, deviceDown.z))
                let angles = eulerAngles
                Text(String(format: "Roll: %.1f°", angles.roll * 180.0 / .pi))
                Text(String(format: "Pitch: %.1f°", angles.pitch * 180.0 / .pi))
                Text(String(format: "Yaw: %.1f°", angles.yaw * 180.0 / .pi))
                if let gravityDown {
                    Text("Samples: \(gravityDown.debugDescription)")
                }
            }
            .font(.system(.footnote, design: .monospaced))
        }
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct GravityIndicatorView: View {
    let downVector: simd_float3

    var body: some View {
        GeometryReader { geo in
            let size = min(geo.size.width, geo.size.height)
            let radius = size / 2.0 - 6.0
            let center = CGPoint(x: geo.size.width / 2.0, y: geo.size.height / 2.0)
            let dx = CGFloat(downVector.x)
            let dy = CGFloat(downVector.y)
            let length = max(sqrt(dx * dx + dy * dy), 0.001)
            let direction = CGPoint(x: dx / length, y: dy / length)
            let arrowEnd = CGPoint(
                x: center.x + direction.x * radius,
                y: center.y + direction.y * radius
            )

            ZStack {
                Circle()
                    .stroke(Color.white.opacity(0.7), lineWidth: 1.5)
                    .frame(width: size - 8.0, height: size - 8.0)
                    .position(center)

                Path { path in
                    path.move(to: center)
                    path.addLine(to: arrowEnd)
                }
                .stroke(Color.yellow, style: StrokeStyle(lineWidth: 3.0, lineCap: .round))

                ArrowHead()
                    .fill(Color.yellow)
                    .frame(width: 18, height: 18)
                    .position(arrowEnd)

                Circle()
                    .fill(Color.white)
                    .frame(width: 6, height: 6)
                    .position(center)
            }
        }
    }

    private struct ArrowHead: Shape {
        func path(in rect: CGRect) -> Path {
            Path { path in
                path.move(to: CGPoint(x: rect.midX, y: rect.minY))
                path.addLine(to: CGPoint(x: rect.maxX, y: rect.maxY))
                path.addLine(to: CGPoint(x: rect.minX, y: rect.maxY))
                path.closeSubpath()
            }
        }
    }
}

private struct MilestoneFourPanel: View {
    let trackedPoints: [TrackedPoint]

    private var convergedCount: Int {
        trackedPoints.filter { $0.status == .converged }.count
    }

    private var divergedCount: Int {
        trackedPoints.filter { $0.status == .diverged }.count
    }

    private var outOfBoundsCount: Int {
        trackedPoints.filter { $0.status == .outOfBounds }.count
    }

    private var averageResidual: Float {
        guard !trackedPoints.isEmpty else { return 0 }
        let sum = trackedPoints.reduce(Float(0)) { $0 + $1.residual }
        return sum / Float(trackedPoints.count)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Milestone 4 — PyrLK Tracker")
                .font(.headline)
            if trackedPoints.isEmpty {
                Text("Waiting for feature seeds…")
            } else {
                Text("Tracks: \(trackedPoints.count)")
                Text("Converged: \(convergedCount) · Diverged: \(divergedCount) · OOB: \(outOfBoundsCount)")
                Text(String(format: "Avg Residual: %.4f", averageResidual))
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct MilestoneFivePanel: View {
    let summary: TwoViewSummary?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Milestone 5 — Two-View Init")
                .font(.headline)
            if let summary {
                Text("Inliers: \(summary.inliers)")
                Text(String(format: "Avg Error: %.5f", summary.averageError))
                Text(String(format: "Translation: [%.2f, %.2f, %.2f]", summary.translation.x, summary.translation.y, summary.translation.z))
                Text("Rotation Matrix:")
                ForEach(0..<3, id: \.self) { row in
                    let start = row * 3
                    let slice = summary.rotation[start..<start + 3]
                    let values = slice.map { String(format: "%.3f", $0) }.joined(separator: "  ")
                    Text(values)
                }
            } else {
                Text("Waiting for robust essential matrix…")
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct MilestoneSixPanel: View {
    let trajectory: [PoseSample]

    private var pathLength: Double {
        guard trajectory.count > 1 else { return 0 }
        var length = 0.0
        for index in 1..<trajectory.count {
            let prev = trajectory[index - 1].position
            let curr = trajectory[index].position
            let delta = curr - prev
            length += sqrt(delta.x * delta.x + delta.y * delta.y + delta.z * delta.z)
        }
        return length
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Milestone 6 — VO Trajectory")
                .font(.headline)
            if trajectory.count <= 1 {
                Text("Waiting for pose estimates…")
            } else {
                Text("Samples: \(trajectory.count)")
                Text(String(format: "Approx Path Length: %.3f", pathLength))
                if let last = trajectory.last?.position {
                    Text(String(format: "Last Pose: [%.2f, %.2f, %.2f]", last.x, last.y, last.z))
                }
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct MilestoneSevenPanel: View {
    let stats: MapStats
    let summary: RelocalizationSummary?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Milestone 7 — Relocalization")
                .font(.headline)
            Text("Keyframes: \(stats.keyframes) · Landmarks: \(stats.landmarks)")
            if let summary {
                Text("Inliers: \(summary.inliers)")
                Text(String(format: "Avg Error: %.4f", summary.averageError))
                let translation = translationVector(from: summary.pose)
                Text(String(format: "Recovered Pose: [%.2f, %.2f, %.2f]", translation.x, translation.y, translation.z))
            } else {
                Text("Waiting for relocalization attempt…")
            }
        }
        .font(.system(.footnote, design: .monospaced))
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct MilestoneEightPanel: View {
    let stats: MapStats
    let anchors: [UInt64: simd_double4x4]
    let lastPose: simd_double4x4?
    let placeAnchor: () -> Void
    let statusMessage: String?
    let saveMap: () -> Void
    let loadMap: () -> Void

    private var sortedAnchors: [(UInt64, simd_double4x4)] {
        anchors.sorted { $0.key < $1.key }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Milestone 8 — Anchors")
                .font(.headline)
            Text("Anchors: \(stats.anchors) · Keyframes: \(stats.keyframes)")

            if let lastPose {
                let translation = translationVector(from: lastPose)
                Text(String(format: "Last Pose: [%.2f, %.2f, %.2f]", translation.x, translation.y, translation.z))
            } else {
                Text("Last Pose: pending…")
            }

            HStack(spacing: 8) {
                Button(action: placeAnchor) {
                    Text("Place Anchor")
                        .font(.system(.footnote, design: .monospaced).bold())
                        .padding(.vertical, 6)
                        .padding(.horizontal, 16)
                        .background(Color.blue.opacity(0.75), in: Capsule())
                        .foregroundStyle(.white)
                }

                Button(action: saveMap) {
                    Text("Save Map")
                        .font(.system(.footnote, design: .monospaced))
                        .padding(.vertical, 6)
                        .padding(.horizontal, 16)
                        .background(Color.green.opacity(0.7), in: Capsule())
                        .foregroundStyle(.white)
                }

                Button(action: loadMap) {
                    Text("Load Map")
                        .font(.system(.footnote, design: .monospaced))
                        .padding(.vertical, 6)
                        .padding(.horizontal, 16)
                        .background(Color.orange.opacity(0.7), in: Capsule())
                        .foregroundStyle(.white)
                }
            }

            if sortedAnchors.isEmpty {
                Text("No anchors recorded yet.")
                    .font(.system(.footnote, design: .monospaced))
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(sortedAnchors, id: \.0) { entry in
                        let translation = translationVector(from: entry.1)
                        Text(String(format: "#%llu → [%.2f, %.2f, %.2f]", entry.0, translation.x, translation.y, translation.z))
                    }
                }
                .font(.system(.footnote, design: .monospaced))
            }

            if let statusMessage {
                Text(statusMessage)
                    .font(.system(.footnote, design: .monospaced))
                    .foregroundStyle(.yellow)
            }
        }
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
    }
}

private struct MilestoneTwoTouchOverlay: View {
    @ObservedObject var cameraManager: CameraCaptureManager
    @Binding var result: BackProjectionResult?

    var body: some View {
        GeometryReader { geo in
            Color.clear
                .contentShape(Rectangle())
                .gesture(
                    DragGesture(minimumDistance: 0)
                        .onChanged { value in
                            updateResult(location: value.location, size: geo.size)
                        }
                        .onEnded { value in
                            updateResult(location: value.location, size: geo.size)
                        }
                )
                .overlay(alignment: .topLeading) {
                    if let result {
                        Crosshair()
                            .stroke(Color.yellow, lineWidth: 1.5)
                            .frame(width: 40, height: 40)
                            .position(result.tapPoint)
                    }
                }
        }
    }

    private func updateResult(location: CGPoint, size: CGSize) {
        guard let sample = cameraManager.lastSample,
              size.width > 0,
              size.height > 0 else {
            result = nil
            return
        }

        let scaleX = Double(location.x / size.width)
        let scaleY = Double(location.y / size.height)
        let pixelX = scaleX * Double(sample.intrinsics.width)
        let pixelY = scaleY * Double(sample.intrinsics.height)

        let ndcX = (pixelX - sample.intrinsics.cx) / sample.intrinsics.fx
        let ndcY = (pixelY - sample.intrinsics.cy) / sample.intrinsics.fy
        let ray = simd_normalize(SIMD3<Double>(ndcX, ndcY, 1.0))

        result = BackProjectionResult(
            tapPoint: location,
            pixel: CGPoint(x: pixelX, y: pixelY),
            ray: ray
        )
    }

    private struct Crosshair: Shape {
        func path(in rect: CGRect) -> Path {
            var path = Path()
            let centerX = rect.midX
            let centerY = rect.midY
            path.move(to: CGPoint(x: centerX - rect.width / 2, y: centerY))
            path.addLine(to: CGPoint(x: centerX + rect.width / 2, y: centerY))
            path.move(to: CGPoint(x: centerX, y: centerY - rect.height / 2))
            path.addLine(to: CGPoint(x: centerX, y: centerY + rect.height / 2))
            return path
        }
    }
}

private struct TrackedPointsOverlay: View {
    let trackedPoints: [TrackedPoint]
    let intrinsics: IntrinsicsSummary?

    var body: some View {
        GeometryReader { geo in
            let width = CGFloat(intrinsics?.width ?? 0)
            let height = CGFloat(intrinsics?.height ?? 0)
            if width <= 0 || height <= 0 {
                Color.clear
            } else {
                Canvas { context, size in
                    for point in trackedPoints.prefix(200) where point.status == .converged {
                        // Convert initial position to screen coordinates
                        let u0 = CGFloat(point.initial.x) / width
                        let v0 = CGFloat(point.initial.y) / height
                        let x0 = u0 * size.width
                        let y0 = v0 * size.height
                        
                        // Convert refined position to screen coordinates
                        let u1 = CGFloat(point.refined.x) / width
                        let v1 = CGFloat(point.refined.y) / height
                        let x1 = u1 * size.width
                        let y1 = v1 * size.height
                        
                        // Calculate flow magnitude for color coding
                        let dx = x1 - x0
                        let dy = y1 - y0
                        let magnitude = sqrt(dx * dx + dy * dy)
                        
                        // Color based on magnitude: green for small, yellow for medium, red for large
                        let color: Color
                        if magnitude < 5.0 {
                            color = .green
                        } else if magnitude < 15.0 {
                            color = .yellow
                        } else {
                            color = .red
                        }
                        
                        // Draw initial position dot (start of flow)
                        let startDot = CGRect(x: x0 - 2.0, y: y0 - 2.0, width: 4.0, height: 4.0)
                        context.fill(Path(ellipseIn: startDot), with: .color(color.opacity(0.5)))
                        
                        // Draw flow arrow
                        if magnitude > 1.0 {
                            // Draw line from initial to refined
                            var path = Path()
                            path.move(to: CGPoint(x: x0, y: y0))
                            path.addLine(to: CGPoint(x: x1, y: y1))
                            context.stroke(path, with: .color(color), lineWidth: 2.0)
                            
                            // Draw arrowhead (elongated for clear directionality)
                            let arrowLength: CGFloat = 10.0
                            let angle = atan2(dy, dx)
                            let arrowAngle: CGFloat = .pi / 8.0  // Narrower angle for sharper arrow
                            
                            let p1 = CGPoint(
                                x: x1 - arrowLength * cos(angle - arrowAngle),
                                y: y1 - arrowLength * sin(angle - arrowAngle)
                            )
                            let p2 = CGPoint(
                                x: x1 - arrowLength * cos(angle + arrowAngle),
                                y: y1 - arrowLength * sin(angle + arrowAngle)
                            )
                            
                            var arrowPath = Path()
                            arrowPath.move(to: CGPoint(x: x1, y: y1))
                            arrowPath.addLine(to: p1)
                            arrowPath.addLine(to: p2)
                            arrowPath.closeSubpath()
                            context.fill(arrowPath, with: .color(color))
                        } else {
                            // For very small motion, just draw a brighter dot
                            let endDot = CGRect(x: x1 - 2.5, y: y1 - 2.5, width: 5.0, height: 5.0)
                            context.fill(Path(ellipseIn: endDot), with: .color(color))
                        }
                    }
                }
            }
        }
    }
}

#Preview {
    ContentView()
}
