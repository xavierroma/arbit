use log::{debug, trace};
use std::ops::{Index, IndexMut};

const GAUSS_KERNEL: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
const GAUSS_KERNEL_SUM: f32 = 16.0; // Sum of GAUSS_KERNEL

/// Simple single-channel image buffer backed by contiguous `f32` pixels.
#[derive(Debug, Clone, PartialEq)]
pub struct ImageBuffer {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl ImageBuffer {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Self {
        assert_eq!(
            width * height,
            data.len(),
            "Image data does not match dimensions."
        );
        Self {
            width,
            height,
            data,
        }
    }

    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.data[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        let idx = y * self.width + x;
        self.data[idx] = value;
    }

    pub fn sample(&self, x: f32, y: f32) -> f32 {
        if x < 0.0 || y < 0.0 {
            return 0.0;
        }
        if x > (self.width - 1) as f32 || y > (self.height - 1) as f32 {
            return 0.0;
        }

        let x0 = x.floor() as usize;
        let y0 = y.floor() as usize;
        let x1 = (x0 + 1).min(self.width - 1);
        let y1 = (y0 + 1).min(self.height - 1);
        let dx = x - x0 as f32;
        let dy = y - y0 as f32;

        let v00 = self.get(x0, y0);
        let v10 = self.get(x1, y0);
        let v01 = self.get(x0, y1);
        let v11 = self.get(x1, y1);

        let v0 = v00 + dx * (v10 - v00);
        let v1 = v01 + dx * (v11 - v01);
        v0 + dy * (v1 - v0)
    }

    pub fn from_bgra8(bytes: &[u8], width: usize, height: usize, bytes_per_row: usize) -> Self {
        debug!(
            "Converting BGRA8 buffer to grayscale: {}x{}, {} bytes per row, {} total bytes",
            width,
            height,
            bytes_per_row,
            bytes.len()
        );

        assert!(
            bytes.len() >= bytes_per_row * height,
            "Byte buffer shorter than expected rows."
        );
        let mut data = Vec::with_capacity(width * height);
        for row in 0..height {
            let row_start = row * bytes_per_row;
            for col in 0..width {
                let offset = row_start + col * 4;
                let b = bytes[offset] as f32;
                let g = bytes[offset + 1] as f32;
                let r = bytes[offset + 2] as f32;
                // Simple luma approximation (Rec. 709)
                let gray = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                data.push(gray);
            }
        }
        debug!("Converted BGRA8 to grayscale image: {}x{}", width, height);
        Self {
            width,
            height,
            data,
        }
    }

    fn sample_clamped(&self, x: isize, y: isize) -> f32 {
        let xi = x.clamp(0, self.width as isize - 1) as usize;
        let yi = y.clamp(0, self.height as isize - 1) as usize;
        self.get(xi, yi)
    }
}

impl Index<(usize, usize)> for ImageBuffer {
    type Output = f32;

    fn index(&self, index: (usize, usize)) -> &Self::Output {
        let (x, y) = index;
        &self.data[y * self.width + x]
    }
}

impl IndexMut<(usize, usize)> for ImageBuffer {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let (x, y) = index;
        &mut self.data[y * self.width + x]
    }
}

#[derive(Debug, Clone)]
pub struct PyramidLevel {
    pub octave: usize,
    pub scale: f32,
    pub image: ImageBuffer,
    pub grad_x: ImageBuffer,
    pub grad_y: ImageBuffer,
}

#[derive(Debug, Clone)]
pub struct Pyramid {
    levels: Vec<PyramidLevel>,
}

impl Pyramid {
    pub fn levels(&self) -> &[PyramidLevel] {
        &self.levels
    }
}

pub fn build_pyramid(base: &ImageBuffer, octaves: usize) -> Pyramid {
    assert!(octaves >= 1, "Pyramid must contain at least one octave.");

    debug!(
        "Building pyramid with {} octaves from {}x{} image",
        octaves,
        base.width(),
        base.height()
    );

    let mut levels = Vec::with_capacity(octaves);
    let mut current = base.clone();

    for octave in 0..octaves {
        trace!("Processing octave {}", octave);
        let blurred = gaussian_blur(&current);
        let (grad_x, grad_y) = sobel_gradients(&blurred);
        levels.push(PyramidLevel {
            octave,
            scale: 1.0 / 2f32.powi(octave as i32),
            image: blurred.clone(),
            grad_x,
            grad_y,
        });

        if octave + 1 != octaves {
            current = downsample(&blurred);
            if current.width() < 2 || current.height() < 2 {
                break;
            }
        }
    }

    Pyramid { levels }
}

fn gaussian_blur(src: &ImageBuffer) -> ImageBuffer {
    let mut tmp = ImageBuffer::zeros(src.width(), src.height());
    let mut dst = ImageBuffer::zeros(src.width(), src.height());

    // Horizontal pass
    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut accum = 0.0;
            for (k, weight) in GAUSS_KERNEL.iter().enumerate() {
                let sample_x = x as isize + k as isize - 2;
                accum += weight * src.sample_clamped(sample_x, y as isize);
            }
            tmp.set(x, y, accum / GAUSS_KERNEL_SUM);
        }
    }

    // Vertical pass
    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut accum = 0.0;
            for (k, weight) in GAUSS_KERNEL.iter().enumerate() {
                let sample_y = y as isize + k as isize - 2;
                accum += weight * tmp.sample_clamped(x as isize, sample_y);
            }
            dst.set(x, y, accum / GAUSS_KERNEL_SUM);
        }
    }

    dst
}

fn downsample(src: &ImageBuffer) -> ImageBuffer {
    let new_width = (src.width() / 2).max(1);
    let new_height = (src.height() / 2).max(1);
    let mut dst = ImageBuffer::zeros(new_width, new_height);

    for y in 0..new_height {
        for x in 0..new_width {
            let sx = (x * 2) as isize;
            let sy = (y * 2) as isize;
            let mut accum = 0.0;
            for dy in 0..2 {
                for dx in 0..2 {
                    accum += src.sample_clamped(sx + dx as isize, sy + dy as isize);
                }
            }
            dst.set(x, y, accum * 0.25);
        }
    }

    dst
}

fn sobel_gradients(src: &ImageBuffer) -> (ImageBuffer, ImageBuffer) {
    let mut gx = ImageBuffer::zeros(src.width(), src.height());
    let mut gy = ImageBuffer::zeros(src.width(), src.height());

    let kernel_x = [[-1.0f32, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    let kernel_y = [[-1.0f32, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];

    for y in 0..src.height() {
        for x in 0..src.width() {
            let mut acc_x = 0.0;
            let mut acc_y = 0.0;
            for ky in 0..3 {
                for kx in 0..3 {
                    let sample = src
                        .sample_clamped(x as isize + kx as isize - 1, y as isize + ky as isize - 1);
                    acc_x += kernel_x[ky][kx] * sample;
                    acc_y += kernel_y[ky][kx] * sample;
                }
            }
            gx.set(x, y, acc_x * 0.25); // Normalise sobel magnitude to keep scale manageable
            gy.set(x, y, acc_y * 0.25);
        }
    }

    (gx, gy)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn bgra_conversion_yields_expected_luma() {
        let width = 2;
        let height = 1;
        let bytes_per_row = width * 4;
        let pixels = [
            255u8, 0, 0, 0, // Blue pixel
            0, 255, 0, 0, // Green pixel
        ];
        let image = ImageBuffer::from_bgra8(&pixels, width, height, bytes_per_row);
        assert_eq!(image.width(), width);
        assert_eq!(image.height(), height);
        assert_relative_eq!(image.get(0, 0), 0.0722 * 255.0, epsilon = 1e-4);
        assert_relative_eq!(image.get(1, 0), 0.7152 * 255.0, epsilon = 1e-4);
    }

    #[test]
    fn gaussian_blur_preserves_constant_image() {
        let width = 8;
        let height = 8;
        let mut image = ImageBuffer::zeros(width, height);
        for y in 0..height {
            for x in 0..width {
                image.set(x, y, 5.0);
            }
        }
        let blurred = gaussian_blur(&image);
        for y in 0..height {
            for x in 0..width {
                assert_relative_eq!(blurred.get(x, y), 5.0, epsilon = 1e-6);
            }
        }
    }

    #[test]
    fn pyramid_produces_expected_dimensions() {
        let width = 16;
        let height = 16;
        let data: Vec<f32> = (0..width * height).map(|v| v as f32).collect();
        let base = ImageBuffer::new(width, height, data);
        let pyramid = build_pyramid(&base, 3);
        let levels = pyramid.levels();
        assert_eq!(levels.len(), 3);
        assert_eq!(levels[0].image.width(), 16);
        assert_eq!(levels[0].image.height(), 16);
        assert_eq!(levels[1].image.width(), 8);
        assert_eq!(levels[1].image.height(), 8);
        assert_eq!(levels[2].image.width(), 4);
        assert_eq!(levels[2].image.height(), 4);
    }

    #[test]
    fn sobel_detects_horizontal_edge() {
        let width = 5;
        let height = 5;
        let mut image = ImageBuffer::zeros(width, height);
        for y in 0..height {
            for x in 0..width {
                if y >= height / 2 {
                    image.set(x, y, 10.0);
                } else {
                    image.set(x, y, 0.0);
                }
            }
        }
        let (gx, gy) = sobel_gradients(&image);
        for y in 0..height {
            for x in 0..width {
                assert!(gx.get(x, y).abs() < 1e-3);
            }
        }
        // Expected vertical gradient near edge
        assert!(gy.get(2, 2).abs() > 0.1);
    }
}
