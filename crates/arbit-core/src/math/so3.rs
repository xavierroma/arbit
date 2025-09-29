use nalgebra::{Matrix3, Quaternion, Unit, UnitQuaternion, Vector3};
use std::f64;

/// Group helper for SO(3) operations backed by unit quaternions.
#[derive(Debug, Clone, PartialEq)]
pub struct SO3 {
    rotation: UnitQuaternion<f64>,
}

impl SO3 {
    /// Returns the identity rotation.
    pub fn identity() -> Self {
        Self::from_unit_quaternion(UnitQuaternion::identity())
    }

    /// Builds an SO(3) rotation from a unit quaternion.
    pub fn from_unit_quaternion(rotation: UnitQuaternion<f64>) -> Self {
        Self { rotation }
    }

    /// Builds an SO(3) rotation from an axis-angle pair.
    pub fn from_axis_angle(axis: &Vector3<f64>, angle: f64) -> Self {
        if axis.norm_squared() < f64::EPSILON {
            return Self::identity();
        }

        let unit_axis = Unit::new_normalize(*axis);
        let rotation = UnitQuaternion::from_axis_angle(&unit_axis, angle);
        Self::from_unit_quaternion(rotation)
    }

    /// Returns the unit quaternion produced by the exponential map of an so(3) vector.
    pub fn exp(omega: &Vector3<f64>) -> Self {
        Self::from_unit_quaternion(UnitQuaternion::from_scaled_axis(*omega))
    }

    /// Returns the so(3) vector (axis-angle) associated with this rotation.
    pub fn log(&self) -> Vector3<f64> {
        self.rotation.scaled_axis()
    }

    /// Returns a reference to the underlying unit quaternion.
    pub fn unit_quaternion(&self) -> &UnitQuaternion<f64> {
        &self.rotation
    }

    /// Consumes the rotation and returns the underlying unit quaternion.
    pub fn into_unit_quaternion(self) -> UnitQuaternion<f64> {
        self.rotation
    }
}

impl From<UnitQuaternion<f64>> for SO3 {
    fn from(rotation: UnitQuaternion<f64>) -> Self {
        SO3::from_unit_quaternion(rotation)
    }
}

impl From<SO3> for UnitQuaternion<f64> {
    fn from(rotation: SO3) -> Self {
        rotation.into_unit_quaternion()
    }
}
/// Normalizes an arbitrary quaternion and returns the associated unit quaternion.
///
/// If the quaternion has zero norm, the identity rotation is returned.
pub fn normalize_quaternion(quaternion: &Quaternion<f64>) -> UnitQuaternion<f64> {
    UnitQuaternion::try_new(quaternion.clone(), f64::EPSILON)
        .unwrap_or_else(|| UnitQuaternion::identity())
}

/// Composes two rotations represented as unit quaternions.
pub fn compose_rotations(a: &UnitQuaternion<f64>, b: &UnitQuaternion<f64>) -> UnitQuaternion<f64> {
    a * b
}

/// Converts a unit quaternion into the corresponding 3x3 rotation matrix.
pub fn rotation_matrix_from_quaternion(rotation: &UnitQuaternion<f64>) -> Matrix3<f64> {
    rotation.to_rotation_matrix().into_inner()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn so3_roundtrip_log_exp() {
        let omega = Vector3::new(0.2, -0.1, 0.05);
        let rotation = SO3::exp(&omega);
        let recovered = rotation.log();

        assert_relative_eq!(omega, recovered, epsilon = 1e-9);
    }

    #[test]
    fn normalization_handles_zero_quaternion() {
        let zero = Quaternion::new(0.0, 0.0, 0.0, 0.0);
        let unit = normalize_quaternion(&zero);
        assert_relative_eq!(unit.quaternion(), &Quaternion::identity());
    }

    #[test]
    fn compose_and_inverse_match_identity() {
        let first = UnitQuaternion::from_euler_angles(0.3, -0.2, 0.5);
        let second = UnitQuaternion::from_euler_angles(-0.4, 0.1, 0.2);
        let composed = compose_rotations(&first, &second);
        let inv = composed.inverse();
        let identity = compose_rotations(&composed, &inv);

        assert_relative_eq!(
            identity.quaternion(),
            &Quaternion::identity(),
            epsilon = 1e-9
        );
    }

    #[test]
    fn rotation_matrix_conversion() {
        let rotation = UnitQuaternion::from_euler_angles(0.1, 0.2, 0.3);
        let matrix = rotation_matrix_from_quaternion(&rotation);
        let expected = rotation.to_rotation_matrix().into_inner();
        assert_relative_eq!(matrix, expected, epsilon = 1e-12);
    }
}
