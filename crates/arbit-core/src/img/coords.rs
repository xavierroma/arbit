use nalgebra::Vector2;

/// Converts pixel coordinates (u, v) into Normalized Device Coordinates in [-1, 1].
/// The coordinate system assumes the image origin is at the top-left corner with the
/// positive direction to the right (u) and down (v).
pub fn pixel_to_ndc(pixel: Vector2<f64>, width: u32, height: u32) -> Vector2<f64> {
    assert!(
        width > 0 && height > 0,
        "Image dimensions must be positive."
    );

    let u = pixel.x;
    let v = pixel.y;
    let w = width as f64;
    let h = height as f64;

    Vector2::new((2.0 * u / (w - 1.0)) - 1.0, 1.0 - (2.0 * v / (h - 1.0)))
}

/// Converts Normalized Device Coordinates (in [-1, 1]) back to pixel coordinates using
/// the same axis conventions as [`pixel_to_ndc`].
pub fn ndc_to_pixel(ndc: Vector2<f64>, width: u32, height: u32) -> Vector2<f64> {
    assert!(
        width > 0 && height > 0,
        "Image dimensions must be positive."
    );

    let w = width as f64;
    let h = height as f64;
    let u = (ndc.x + 1.0) * 0.5 * (w - 1.0);
    let v = (1.0 - ndc.y) * 0.5 * (h - 1.0);

    Vector2::new(u, v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn origin_maps_to_lower_left_ndc() {
        let ndc = pixel_to_ndc(Vector2::new(0.0, 0.0), 640, 480);
        assert_relative_eq!(ndc.x, -1.0, epsilon = 1e-12);
        assert_relative_eq!(ndc.y, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn center_round_trip() {
        let width = 640u32;
        let height = 480u32;
        let center = Vector2::new((width as f64 - 1.0) * 0.5, (height as f64 - 1.0) * 0.5);
        let ndc = pixel_to_ndc(center, width, height);
        assert_relative_eq!(ndc.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(ndc.y, 0.0, epsilon = 1e-12);

        let pixel = ndc_to_pixel(ndc, width, height);
        assert_relative_eq!(pixel.x, center.x, epsilon = 1e-12);
        assert_relative_eq!(pixel.y, center.y, epsilon = 1e-12);
    }

    #[test]
    fn ndc_corners_map_to_pixels() {
        let width = 1280u32;
        let height = 720u32;

        let top_left = ndc_to_pixel(Vector2::new(-1.0, 1.0), width, height);
        assert_relative_eq!(top_left.x, 0.0, epsilon = 1e-12);
        assert_relative_eq!(top_left.y, 0.0, epsilon = 1e-12);

        let bottom_right = ndc_to_pixel(Vector2::new(1.0, -1.0), width, height);
        assert_relative_eq!(bottom_right.x, width as f64 - 1.0, epsilon = 1e-12);
        assert_relative_eq!(bottom_right.y, height as f64 - 1.0, epsilon = 1e-12);
    }
}
