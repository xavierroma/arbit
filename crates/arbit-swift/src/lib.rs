//! Swift packaging layer for the Arbit FFI.
//!
//! This crate provides a static library packaging of `arbit-ffi` for iOS/macOS
//! consumption via XCFramework. It re-exports all arbit-ffi symbols so that
//! iOS applications can link against a single compiled artifact.
//!
//! Platform-specific helpers and Swift-native APIs are implemented in the
//! Swift wrapper layer (`swift-package/Sources/Arbit/`), keeping this crate
//! as a pure packaging layer.

#![deny(unused_must_use)]

// Re-export all arbit-ffi symbols for iOS consumption
pub use arbit_ffi::*;
