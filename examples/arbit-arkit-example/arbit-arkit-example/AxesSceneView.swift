//
//  AxesSceneView.swift
//  arbit-arkit-example
//
//  Renders world-aligned axes driven by device motion.
//

import SceneKit
import SwiftUI
import UIKit
import simd

private func buildAxesNode() -> SCNNode {
    let root = SCNNode()

    let axisLength: CGFloat = 1.0
    let radius: CGFloat = 0.01

    let xAxis = SCNCylinder(radius: radius, height: axisLength)
    xAxis.firstMaterial?.diffuse.contents = UIColor.systemRed
    let xNode = SCNNode(geometry: xAxis)
    xNode.eulerAngles = SCNVector3(0, 0, Float.pi / 2)
    xNode.position = SCNVector3(axisLength / 2, 0, 0)

    let yAxis = SCNCylinder(radius: radius, height: axisLength)
    yAxis.firstMaterial?.diffuse.contents = UIColor.systemGreen
    let yNode = SCNNode(geometry: yAxis)
    yNode.position = SCNVector3(0, axisLength / 2, 0)

    let zAxis = SCNCylinder(radius: radius, height: axisLength)
    zAxis.firstMaterial?.diffuse.contents = UIColor.systemBlue
    let zNode = SCNNode(geometry: zAxis)
    zNode.eulerAngles = SCNVector3(Float.pi / 2, 0, 0)
    zNode.position = SCNVector3(0, 0, axisLength / 2)

    root.addChildNode(xNode)
    root.addChildNode(yNode)
    root.addChildNode(zNode)

    return root
}

struct AxesSceneView: UIViewRepresentable {
    var orientation: simd_quatf

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    func makeUIView(context: Context) -> SCNView {
        let view = SCNView(frame: .zero, options: nil)
        view.scene = context.coordinator.scene
        view.backgroundColor = .clear
        view.allowsCameraControl = false
        view.isJitteringEnabled = true
        view.pointOfView = context.coordinator.cameraNode
        return view
    }

    func updateUIView(_ view: SCNView, context: Context) {
        context.coordinator.update(orientation: orientation)
    }

    final class Coordinator {
        let scene = SCNScene()
        let cameraNode = SCNNode()
        private let axesNode = buildAxesNode()

        init() {
            let camera = SCNCamera()
            camera.usesOrthographicProjection = true
            camera.orthographicScale = 1.5
            camera.zNear = 0.1
            camera.zFar = 10.0
            cameraNode.camera = camera
            cameraNode.position = SCNVector3(0, 0, 3)

            let lightNode = SCNNode()
            let light = SCNLight()
            light.type = .ambient
            light.color = UIColor(white: 0.8, alpha: 1.0)
            lightNode.light = light

            scene.rootNode.addChildNode(cameraNode)
            scene.rootNode.addChildNode(lightNode)
            scene.rootNode.addChildNode(axesNode)
        }

        func update(orientation: simd_quatf) {
            axesNode.simdOrientation = orientation
        }
    }
}
