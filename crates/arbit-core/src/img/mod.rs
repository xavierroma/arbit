pub mod image_utils;
pub mod pyramid;
use image::{ImageBuffer, Luma, Rgb, Rgba};
pub use pyramid::{Pyramid, PyramidLevel, build_pyramid};

/// Sendable Rgb image buffer
pub type RgbImage = ImageBuffer<Rgb<u8>, Vec<u8>>;
/// Sendable Rgb + alpha channel image buffer
pub type RgbaImage = ImageBuffer<Rgba<u8>, Vec<u8>>;
/// Sendable grayscale image buffer
pub type GrayImage = ImageBuffer<Luma<u8>, Vec<u8>>;

/// An image buffer for 32-bit float RGB pixels,
/// where the backing container is a flattened vector of floats.
pub type Rgb32FImage = ImageBuffer<Rgb<f32>, Vec<f32>>;

/// An image buffer for 32-bit float RGBA pixels,
/// where the backing container is a flattened vector of floats.
pub type Rgba32FImage = ImageBuffer<Rgba<f32>, Vec<f32>>;
