//
//  DeviceOrientationProvider.swift
//  arbit-arkit-example
//
//  Provides live device orientation using CoreMotion.
//

import Combine
import CoreMotion
import Foundation
import simd

final class DeviceOrientationProvider: ObservableObject {
    @Published var deviceToWorld: simd_quatf = simd_quatf(ix: 0, iy: 0, iz: 0, r: 1)

    private let motionManager = CMMotionManager()
    private let queue: OperationQueue = {
        let queue = OperationQueue()
        queue.name = "arbit.motion.queue"
        queue.qualityOfService = .userInteractive
        return queue
    }()

    func start() {
        guard motionManager.isDeviceMotionAvailable else {
            return
        }

        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        motionManager.startDeviceMotionUpdates(using: .xArbitraryCorrectedZVertical, to: queue) { [weak self] motion, _ in
            guard let motion else { return }
            let q = motion.attitude.quaternion
            let referenceToDevice = simd_quatf(ix: Float(q.x), iy: Float(q.y), iz: Float(q.z), r: Float(q.w))
            let deviceToReference = simd_normalize(referenceToDevice).inverse
            DispatchQueue.main.async {
                self?.deviceToWorld = deviceToReference
            }
        }
    }

    func stop() {
        motionManager.stopDeviceMotionUpdates()
    }
}
