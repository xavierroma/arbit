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
    case pyramid = 1, tracking, twoView, vo, anchors

    var id: Int { rawValue }

    var label: String {
        switch self {
        case .pyramid: "Pyramid"
        case .tracking: "Tracker"
        case .twoView: "Init"
        case .vo: "VO"
        case .anchors: "Anchors"
        }
    }

    var headline: String {
        switch self {
        case .pyramid: "Pyramid"
        case .tracking: "Tracker"
        case .twoView: "Init"
        case .vo: "VO"
        case .anchors: "Anchors"
        }
    }
}

struct ContentView: View {
    @StateObject private var cameraManager = CameraCaptureManager()
    @StateObject private var orientationProvider = DeviceOrientationProvider()

    @State private var selectedMilestone: Milestone = .pyramid
    @State private var previousPipelineSeconds: Double?
    @State private var estimatedFPS: Double?
    @State private var backProjection: BackProjectionResult?

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)

            if cameraManager.authorizationStatus != .authorized {
                PermissionOverlay()
            }

            if selectedMilestone == .tracking {
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
        case .pyramid:
            MilestoneTwoPanel(
                sample: cameraManager.lastSample,
                levels: cameraManager.pyramidLevels,
                projection: backProjection
            )
        case .tracking:
            MilestoneFourPanel(trackedPoints: cameraManager.trackedPoints)
        case .twoView:
            MilestoneFivePanel(summary: cameraManager.twoViewSummary)
        case .vo:
            MilestoneSixPanel(trajectory: cameraManager.trajectory)
        case .anchors:
            MilestoneEightPanel(
                stats: cameraManager.mapStats,
                anchors: cameraManager.anchorPoses,
                lastPose: cameraManager.lastPoseMatrix,
                statusMessage: cameraManager.mapStatusMessage,
                )
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
            Text("Pyramid")
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
            Text("Shi Tomasi")
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
            Text("Two-View Init")
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
            Text("VO Trajectory")
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

private struct MilestoneEightPanel: View {
    let stats: MapStats
    let anchors: [UInt64: simd_double4x4]
//    let visibleAnchors: [ProjectedAnchor]
    let lastPose: simd_double4x4?
    let statusMessage: String?
    
    private var sortedAnchors: [(UInt64, simd_double4x4)] {
        anchors.sorted { $0.key < $1.key }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("AR Anchors")
                .font(.headline)
            
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Map:")
                    Text("  Landmarks: \(stats.landmarks)")
                    Text("  Keyframes: \(stats.keyframes)")
                    Text("  Anchors: \(stats.anchors)")
                }
                
                if let lastPose {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Camera:")
                        let trans = translationVector(from: lastPose)
                        Text(String(format: "  X: %.2fm", trans.x))
                        Text(String(format: "  Y: %.2fm", trans.y))
                        Text(String(format: "  Z: %.2fm", trans.z))
                    }
                }
            }
            .font(.system(.caption, design: .monospaced))

            Text("Tap screen to place anchors at 1m depth")
                .font(.system(.caption, design: .monospaced))
                .foregroundStyle(.cyan)
            
            Text("Green dots = map landmarks")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.green)
            
            Text("Cyan cubes = placed anchors")
                .font(.system(.caption2, design: .monospaced))
                .foregroundStyle(.cyan)

           

            if let statusMessage {
                Text(statusMessage)
                    .font(.system(.caption2, design: .monospaced))
                    .foregroundStyle(.yellow)
            }
        }
        .foregroundStyle(.white)
        .padding()
        .background(.black.opacity(0.58), in: RoundedRectangle(cornerRadius: 12))
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
                        let u0 = CGFloat(point.initial.x - 0.5) / width
                        let v0 = CGFloat(point.initial.y - 0.5) / height
                        let x0 = u0 * size.width
                        let y0 = v0 * size.height
                        
                        // Convert refined position to screen coordinates
                        let u1 = CGFloat(point.refined.x - 0.5) / width
                        let v1 = CGFloat(point.refined.y - 0.5) / height
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

                        // Label the feature with its persistent track identifier
                        let labelPosition = CGPoint(x: x1 + 6.0, y: y1 - 6.0)
                        let labelText = Text("#\(point.trackID)")
                            .font(.system(size: 9, weight: .medium, design: .monospaced))
                            .foregroundStyle(Color.white.opacity(0.9))
                        context.draw(labelText, at: labelPosition, anchor: .topLeading)
                    }
                }
            }
        }
    }
}


//private struct MapLandmarksOverlay: View {
////    let landmarks: [ProjectedLandmark]
//    let intrinsics: IntrinsicsSummary?
//    
//    var body: some View {
//        GeometryReader { geo in
//            let width = CGFloat(intrinsics?.width ?? 0)
//            let height = CGFloat(intrinsics?.height ?? 0)
//            
//            if width <= 0 || height <= 0 {
//                Color.clear
//            } else {
//                Canvas { context, size in
//                    for landmark in landmarks {
//                        // Convert pixel to screen coordinates
//                        let u = CGFloat(landmark.pixelX) / width
//                        let v = CGFloat(landmark.pixelY) / height
//                        let x = u * size.width
//                        let y = v * size.height
//                        
//                        // Color based on depth
//                        let color: Color
//                        if landmark.depth < 1.0 {
//                            color = .yellow  // Very close
//                        } else if landmark.depth < 3.0 {
//                            color = .green   // Medium distance
//                        } else {
//                            color = .blue    // Far
//                        }
//                        
//                        // Draw small dot for each landmark
//                        let dotSize: CGFloat = 3.0
//                        let dot = Path(ellipseIn: CGRect(
//                            x: x - dotSize / 2,
//                            y: y - dotSize / 2,
//                            width: dotSize,
//                            height: dotSize
//                        ))
//                        context.fill(dot, with: .color(color.opacity(0.7)))
//                    }
//                    
//                    // Draw count in top-right
//                    let countText = Text("\(landmarks.count) landmarks")
//                        .font(.system(size: 12, weight: .bold, design: .monospaced))
//                        .foregroundStyle(.green)
//                    
//                    context.draw(countText, at: CGPoint(x: size.width - 80, y: 20))
//                }
//            }
//        }
//    }
//}

#Preview {
    ContentView()
}
