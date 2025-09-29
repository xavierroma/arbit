use nalgebra::{Matrix3, Vector3};

/// Returns the 3x3 skew-symmetric matrix for the provided 3D vector.
///
/// The resulting matrix satisfies `skew(v) * w == v.cross(&w)` for any vector `w`.
pub fn skew_symmetric(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(0.0, -v.z, v.y, v.z, 0.0, -v.x, -v.y, v.x, 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn skew_matches_cross_product() {
        let v = Vector3::new(1.0, -2.0, 0.5);
        let w = Vector3::new(0.25, 0.5, -1.5);

        let lhs = skew_symmetric(&v) * w;
        let rhs = v.cross(&w);

        assert_relative_eq!(lhs, rhs, epsilon = 1e-12);
    }
}
