// swift-tools-version: 6.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Arbit",
    products: [
        // Products define the executables and libraries a package produces, making them visible to other packages.
        .library(
            name: "Arbit",
            targets: ["Arbit"]
        ),
    ],
    targets: [
        // Targets are the basic building blocks of a package, defining a module or a test suite.
        // Targets can depend on other targets in this package and products from dependencies.
        .binaryTarget(
            name: "ArbitFFI",
              path: "ArbitFFI.xcframework"
        ),
        .target(
            name: "Arbit",
            dependencies: ["ArbitFFI"]
        ),

    ]
)
