use image::{GenericImageView, Luma};

pub fn bilinear_sample_luma<T, I>(img: &I, x: f32, y: f32) -> f32
where
    I: GenericImageView<Pixel = Luma<T>>,
    T: Copy + Into<f32>,
{
    if x < 0.0 || y < 0.0 {
        return 0.0;
    }
    let w = img.width() as f32;
    let h = img.height() as f32;
    if x > w - 1.0 || y > h - 1.0 {
        return 0.0;
    }

    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(img.width() - 1);
    let y1 = (y0 + 1).min(img.height() - 1);

    let dx = x - x0 as f32;
    let dy = y - y0 as f32;

    // Extract scalar luminance from the pixel
    let p00 = img.get_pixel(x0, y0).0[0].into();
    let p10 = img.get_pixel(x1, y0).0[0].into();
    let p01 = img.get_pixel(x0, y1).0[0].into();
    let p11 = img.get_pixel(x1, y1).0[0].into();

    let top = p00 + dx * (p10 - p00);
    let bot = p01 + dx * (p11 - p01);
    top + dy * (bot - top)
}
