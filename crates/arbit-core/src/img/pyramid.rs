use image::{
    Luma,
    imageops::{FilterType, resize},
};
use imageproc::{
    filter::gaussian_blur_f32,
    gradients::{horizontal_sobel, vertical_sobel},
    image::{GrayImage, ImageBuffer},
};

use log::{debug, trace};

#[derive(Debug, Clone)]
pub struct PyramidLevel {
    pub octave: usize,
    pub scale: f32,
    pub image: GrayImage,
    pub grad_x: ImageBuffer<Luma<i16>, Vec<i16>>,
    pub grad_y: ImageBuffer<Luma<i16>, Vec<i16>>,
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

pub fn build_pyramid(base: &GrayImage, octaves: usize) -> Pyramid {
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
        trace!("Processing octave {octave}");
        let blurred = gaussian_blur_f32(&current, 1.0);
        let grad_y = scale_gradient(vertical_sobel(&blurred), 0.25);
        let grad_x = scale_gradient(horizontal_sobel(&blurred), 0.25);
        let next = if octave + 1 != octaves {
            resize(
                &blurred,
                blurred.width() / 2,
                blurred.height() / 2,
                FilterType::Triangle,
            )
        } else {
            // Not used, dummy
            GrayImage::new(0, 0)
        };

        levels.push(PyramidLevel {
            octave,
            scale: 1.0 / 2f32.powi(octave as i32),
            image: blurred,
            grad_x,
            grad_y,
        });

        if octave + 1 != octaves {
            if next.width() < 2 || next.height() < 2 {
                break;
            }
            current = next;
        }
    }

    Pyramid { levels }
}

fn scale_gradient(
    grad: ImageBuffer<Luma<i16>, Vec<i16>>,
    scale: f32,
) -> ImageBuffer<Luma<i16>, Vec<i16>> {
    let (width, height) = grad.dimensions();
    let scaled_data: Vec<i16> = grad
        .as_raw()
        .iter()
        .map(|&v| ((v as f32) * scale) as i16)
        .collect();
    ImageBuffer::from_raw(width, height, scaled_data)
        .expect("Failed to create scaled gradient buffer")
}
