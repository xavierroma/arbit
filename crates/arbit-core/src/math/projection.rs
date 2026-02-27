use nalgebra::{Matrix2x3, Matrix3, Point3, Vector2};

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
    pub fn project_point(&self, cam_xyz: &Point3<f64>) -> Option<Vector2<f32>> {
        if cam_xyz.z <= 0.0 {
            return None;
        }

        let norm_xy = self.normalize(cam_xyz);
        let distorted_xy = self.apply_distortion(norm_xy);

        let u = self.fx * distorted_xy.x + self.skew * distorted_xy.y + self.cx;
        let v = self.fy * distorted_xy.y + self.cy;
        let px_uv = Vector2::new(u as f32, v as f32);
        Some(px_uv)
    }

    /// Returns the normalized coordinates for a camera-space 3D point.
    pub fn normalize(&self, cam_xyz: &Point3<f64>) -> Vector2<f64> {
        let norm_xy = Vector2::new(cam_xyz.x / cam_xyz.z, cam_xyz.y / cam_xyz.z);
        norm_xy
    }

    /// Back-projects a pixel (with depth in meters) into camera coordinates.
    pub fn back_project(&self, px_uv: &Vector2<f32>, depth: f64) -> Point3<f64> {
        let x_n =
            (px_uv.x as f64 - self.cx - self.skew * (px_uv.y as f64 - self.cy) / self.fy) / self.fx;
        let y_n = (px_uv.y as f64 - self.cy) / self.fy;
        let cam_xyz = Point3::new(x_n * depth, y_n * depth, depth);
        cam_xyz
    }

    /// Computes the Jacobian of the projection function with respect to camera-space coordinates.
    /// The Jacobian is evaluated assuming no additional distortion is applied (pass-through model).
    pub fn projection_jacobian(&self, cam_xyz: &Point3<f64>) -> Option<Matrix2x3<f64>> {
        if cam_xyz.z <= 0.0 {
            return None;
        }
        let z = cam_xyz.z;
        let x = cam_xyz.x;
        let y = cam_xyz.y;

        let mut jacobian = Matrix2x3::zeros();
        jacobian[(0, 0)] = self.fx / z;
        jacobian[(0, 1)] = self.skew / z;
        jacobian[(0, 2)] = -(self.fx * x + self.skew * y) / (z * z);

        jacobian[(1, 0)] = 0.0;
        jacobian[(1, 1)] = self.fy / z;
        jacobian[(1, 2)] = -(self.fy * y) / (z * z);

        Some(jacobian)
    }

    fn apply_distortion(&self, norm_xy: Vector2<f64>) -> Vector2<f64> {
        let distorted_xy = match &self.distortion {
            DistortionModel::None => norm_xy,
            _ => norm_xy, // Pass-through for milestone one; platform-provided coefficients stored only.
        };
        distorted_xy
    }

    /// Returns true when the intrinsics carry a non-trivial distortion model.
    pub fn has_distortion(&self) -> bool {
        !matches!(self.distortion, DistortionModel::None)
    }

    /// Builds the column-major 3x3 camera matrix often referred to as `K`.
    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, self.skew, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0,
        )
    }
}

/// Converts pixel coordinates to normalized camera coordinates.
pub fn px_uv_to_norm_xy(px: f32, py: f32, intrinsics: &CameraIntrinsics) -> Vector2<f64> {
    Vector2::new(
        (px as f64 - intrinsics.cx) / intrinsics.fx,
        (py as f64 - intrinsics.cy) / intrinsics.fy,
    )
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

        // `project_point` outputs f32 pixels, so finite differences must use a larger
        // step/tolerance than pure f64 Jacobian checks.
        let epsilon = 1e-3;
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

            numeric[(0, i)] = (proj_forward.x as f64 - proj_backward.x as f64) / (2.0 * epsilon);
            numeric[(1, i)] = (proj_forward.y as f64 - proj_backward.y as f64) / (2.0 * epsilon);
        }

        assert_relative_eq!(jacobian, numeric, epsilon = 6.0);
    }

    #[test]
    fn intrinsic_matrix_layout_matches_definition() {
        let intrinsics = CameraIntrinsics::new(
            800.0,
            820.0,
            400.0,
            300.0,
            5.0,
            1280,
            720,
            DistortionModel::BrownConrady {
                k1: 0.1,
                k2: 0.01,
                p1: 0.0,
                p2: 0.0,
                k3: 0.0,
            },
        );

        let matrix = intrinsics.matrix();
        assert_relative_eq!(matrix[(0, 0)], 800.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(0, 1)], 5.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(0, 2)], 400.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(1, 0)], 0.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(1, 1)], 820.0, epsilon = 1e-12);
        assert_relative_eq!(matrix[(1, 2)], 300.0, epsilon = 1e-12);
        assert!(intrinsics.has_distortion());
    }
}
