use nalgebra::{Matrix2x3, Point3};

/// Basic radial/tangential distortion container.
#[derive(Debug, Clone, PartialEq)]
pub enum DistortionModel {
    /// No distortion. Useful as the default on iPhone providers where we trust the native intrinsics.
    None,
    /// Brown-Conrady (k1, k2, p1, p2, k3) parameterization.
    BrownConrady {
        k1: f64,
        k2: f64,
        p1: f64,
        p2: f64,
        k3: f64,
    },
    /// Custom coefficients provided by the platform without interpretation in this milestone.
    Custom(Vec<f64>),
}

impl Default for DistortionModel {
    fn default() -> Self {
        Self::None
    }
}

/// A 2D pixel coordinate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImagePoint {
    pub u: f64,
    pub v: f64,
}

/// Normalized image coordinates (projected onto the z = 1 plane).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizedImagePoint {
    pub x: f64,
    pub y: f64,
}

/// Standard pinhole camera intrinsics used throughout the pipeline.
#[derive(Debug, Clone, PartialEq)]
pub struct CameraIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub skew: f64,
    pub width: u32,
    pub height: u32,
    pub distortion: DistortionModel,
}

impl CameraIntrinsics {
    pub fn new(
        fx: f64,
        fy: f64,
        cx: f64,
        cy: f64,
        skew: f64,
        width: u32,
        height: u32,
        distortion: DistortionModel,
    ) -> Self {
        Self {
            fx,
            fy,
            cx,
            cy,
            skew,
            width,
            height,
            distortion,
        }
    }

    /// Projects a 3D point in camera coordinates onto the image plane.
    /// Returns `None` if the point lies on or behind the camera plane (z <= 0).
    pub fn project_point(&self, point_cam: &Point3<f64>) -> Option<ImagePoint> {
        if point_cam.z <= 0.0 {
            return None;
        }

        let normalized = self.normalize(point_cam);
        let distorted = self.apply_distortion(normalized);

        let u = self.fx * distorted.x + self.skew * distorted.y + self.cx;
        let v = self.fy * distorted.y + self.cy;

        Some(ImagePoint { u, v })
    }

    /// Returns the normalized coordinates for a camera-space 3D point.
    pub fn normalize(&self, point_cam: &Point3<f64>) -> NormalizedImagePoint {
        NormalizedImagePoint {
            x: point_cam.x / point_cam.z,
            y: point_cam.y / point_cam.z,
        }
    }

    /// Back-projects a pixel (with depth in meters) into camera coordinates.
    pub fn back_project(&self, pixel: &ImagePoint, depth: f64) -> Point3<f64> {
        let x_n = (pixel.u - self.cx - self.skew * (pixel.v - self.cy) / self.fy) / self.fx;
        let y_n = (pixel.v - self.cy) / self.fy;
        Point3::new(x_n * depth, y_n * depth, depth)
    }

    /// Computes the Jacobian of the projection function with respect to camera-space coordinates.
    /// The Jacobian is evaluated assuming no additional distortion is applied (pass-through model).
    pub fn projection_jacobian(&self, point_cam: &Point3<f64>) -> Option<Matrix2x3<f64>> {
        if point_cam.z <= 0.0 {
            return None;
        }
        let z = point_cam.z;
        let x = point_cam.x;
        let y = point_cam.y;

        let mut jacobian = Matrix2x3::zeros();
        jacobian[(0, 0)] = self.fx / z;
        jacobian[(0, 1)] = self.skew / z;
        jacobian[(0, 2)] = -(self.fx * x + self.skew * y) / (z * z);

        jacobian[(1, 0)] = 0.0;
        jacobian[(1, 1)] = self.fy / z;
        jacobian[(1, 2)] = -(self.fy * y) / (z * z);

        Some(jacobian)
    }

    fn apply_distortion(&self, normalized: NormalizedImagePoint) -> NormalizedImagePoint {
        match &self.distortion {
            DistortionModel::None => normalized,
            _ => normalized, // Pass-through for milestone one; platform-provided coefficients stored only.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn project_and_back_project_round_trip() {
        let intrinsics = CameraIntrinsics::new(
            800.0,
            800.0,
            320.0,
            240.0,
            0.0,
            640,
            480,
            DistortionModel::None,
        );
        let point = Point3::new(0.1, -0.2, 2.5);
        let pixel = intrinsics
            .project_point(&point)
            .expect("point in front of camera");
        let reconstructed = intrinsics.back_project(&pixel, point.z);

        assert_relative_eq!(point, reconstructed, epsilon = 1e-9);
    }

    #[test]
    fn projection_jacobian_matches_finite_difference() {
        let intrinsics = CameraIntrinsics::new(
            600.0,
            620.0,
            300.0,
            200.0,
            5.0,
            640,
            480,
            DistortionModel::None,
        );
        let point = Point3::new(0.3, -0.1, 1.5);
        let jacobian = intrinsics
            .projection_jacobian(&point)
            .expect("Jacobian defined for points in front of camera");

        let epsilon = 1e-6;
        let mut numeric = Matrix2x3::zeros();
        for i in 0..3 {
            let mut forward = point;
            let mut backward = point;

            match i {
                0 => {
                    forward.x += epsilon;
                    backward.x -= epsilon;
                }
                1 => {
                    forward.y += epsilon;
                    backward.y -= epsilon;
                }
                2 => {
                    forward.z += epsilon;
                    backward.z -= epsilon;
                }
                _ => unreachable!(),
            }

            let proj_forward = intrinsics.project_point(&forward).unwrap();
            let proj_backward = intrinsics.project_point(&backward).unwrap();

            numeric[(0, i)] = (proj_forward.u - proj_backward.u) / (2.0 * epsilon);
            numeric[(1, i)] = (proj_forward.v - proj_backward.v) / (2.0 * epsilon);
        }

        assert_relative_eq!(jacobian, numeric, epsilon = 1e-6);
    }
}
