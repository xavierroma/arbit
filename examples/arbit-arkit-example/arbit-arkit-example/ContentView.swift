//
//  ContentView.swift
//  arbit-arkit-example
//
//  Milestone 1 demo powered by AVFoundation + Arbit FFI.
//

import SwiftUI
import Combine
import SceneKit
import AVFoundation
import arbit_swift_lib

struct ContentView: View {
    @StateObject private var cameraManager = CameraCaptureManager()
    @StateObject private var orientationProvider = DeviceOrientationProvider()

    @State private var previousPipelineSeconds: Double?
    @State private var estimatedFPS: Double?

    var body: some View {
        ZStack {
            CameraPreviewView(session: cameraManager.session)
                .ignoresSafeArea()

            if cameraManager.authorizationStatus != .authorized {
                Color.black.opacity(0.7)
                    .ignoresSafeArea()
                Text("Camera access is required to run the demo.")
                    .font(.headline)
                    .foregroundStyle(.white)
                    .padding()
            }

            VStack {
                HStack(alignment: .top) {
                    metricsPanel
                    Spacer()
                    AxesSceneView(orientation: orientationProvider.deviceToWorld)
                        .frame(width: 180, height: 180)
                        .background(.thinMaterial, in: RoundedRectangle(cornerRadius: 12))
                        .padding([.top, .trailing])
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

    private var metricsPanel: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Arbit Milestone 1")
                .font(.headline)
                .padding(.bottom, 4)

            if let sample = cameraManager.lastSample {
                Text("Capture (s): \(format(sample.captureSeconds))")
                Text("Pipeline (s): \(format(sample.pipelineSeconds))")
                Text(String(format: "Latency (ms): %.2f", sample.latencySeconds * 1_000))
                Text("Frame #: \(sample.frameIndex)")
                if let fps = estimatedFPS {
                    Text(String(format: "Approx FPS: %.1f", fps))
                }

                Text("Intrinsics fx/fy: \(format(sample.intrinsics.fx)), \(format(sample.intrinsics.fy))")
                Text("Principal cx/cy: \(format(sample.intrinsics.cx)), \(format(sample.intrinsics.cy))")
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

    private var footerPanel: some View {
        HStack {
            Spacer()
            if cameraManager.isRunning {
                Text("Session Running")
                    .foregroundStyle(.green)
            } else {
                Text("Session Idle")
                    .foregroundStyle(.yellow)
            }
        }
        .font(.caption)
        .foregroundStyle(.white)
        .padding(10)
        .background(.black.opacity(0.55), in: Capsule())
    }

    private func format(_ value: Double) -> String {
        String(format: "%.3f", value)
    }
}

#Preview {
    ContentView()
}
