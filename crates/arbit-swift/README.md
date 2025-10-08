# arbit-swift

Swift package wrapper for the Arbit FFI layer, providing iOS/macOS-specific integration.

## Overview

This crate packages the `arbit-ffi` C bindings into a Swift-friendly XCFramework with additional platform-specific helpers for iOS/macOS development.

## Architecture

```
arbit-swift/
├── scripts/
│   └── build-xcframework.sh  # Builds the XCFramework
├── swift-package/
│   ├── ArbitFFI.xcframework/  # Generated XCFramework
│   ├── include/               # C module headers
│   │   ├── arbit_swift.h      # Copied from arbit-ffi
│   │   └── dummy.c
│   ├── Package.swift          # Swift Package Manager manifest
│   └── Sources/Arbit/         # Swift wrapper code
└── src/
    └── lib.rs                 # Re-exports arbit-ffi + Swift helpers
```

## Relationship to arbit-ffi

This crate:

1. **Re-exports** all `arbit-ffi` symbols
2. **Packages** everything into a single static library (`libarbit_swift.a`)
3. **Wraps** the library in an XCFramework for iOS consumption

The separation ensures:
- `arbit-ffi` remains language-agnostic (pure C FFI)
- `arbit-swift` is a pure packaging layer for iOS
- Platform-specific helpers are implemented in Swift (not Rust)
- Other language bindings can reuse `arbit-ffi` directly

## Building the XCFramework

```bash
./scripts/build-xcframework.sh
```

This script:

1. Generates the C header from `arbit-ffi` (via `arbit-ffi/scripts/generate-header.sh`)
2. Copies the header to `swift-package/include/arbit_swift.h`
3. Builds static libraries for:
   - `aarch64-apple-ios` (iPhone/iPad)
   - `aarch64-apple-ios-sim` (Simulator)
4. Creates `ArbitFFI.xcframework` with bundled headers

## Usage in Xcode

Add the Swift package to your Xcode project:

```swift
dependencies: [
    .package(path: "path/to/arbit-swift/swift-package")
]
```

Or use the local package in the Xcode UI.

## Swift API

The Swift wrapper (in `Sources/Arbit/`) provides idiomatic Swift interfaces on top of the C FFI:

- Memory-safe types (automatic cleanup via deinit)
- Swift enums for C enums
- Error handling via Result types
- Collection types instead of raw pointers

See `swift-package/Sources/Arbit/arbit.swift` for the full API.

## Platform-Specific Helpers

Platform-specific helpers (such as SceneKit transform conversions) are implemented in pure Swift in the `Sources/Arbit/` directory. This keeps the Rust layer minimal and focused on packaging.

## Testing

Test the FFI integration:

```bash
cargo test -p arbit-swift
```

Test the Swift package:

```bash
cd swift-package
swift test
```

## Development Workflow

When the `arbit-ffi` interface changes:

1. `arbit-ffi` maintainer updates the FFI and regenerates header
2. Run `./scripts/build-xcframework.sh` to pick up changes
3. Update Swift wrapper code if needed
4. Bump version in `Package.swift`

## Build Requirements

- Rust toolchain with iOS targets:
  ```bash
  rustup target add aarch64-apple-ios
  rustup target add aarch64-apple-ios-sim
  ```
- Xcode command-line tools
- `cbindgen` (installed automatically when running the build script)
