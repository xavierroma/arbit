use nalgebra::{Isometry3, Matrix3, Translation3, UnitQuaternion, Vector3};

use super::so3::SO3;
use super::utils::skew_symmetric;

pub type TransformSE3 = Isometry3<f64>;

/// Group helper for SE(3) operations backed by an SO(3) rotation and translation.
#[derive(Debug, Clone, PartialEq)]
pub struct SE3 {
    rotation: SO3,
    translation: Vector3<f64>,
}

impl SE3 {
    /// Returns the identity transform.
    pub fn identity() -> Self {
        Self::from_parts(SO3::identity(), Vector3::zeros())
    }

    /// Constructs an SE(3) element from a rotation and translation.
    pub fn new(rotation: UnitQuaternion<f64>, translation: Vector3<f64>) -> Self {
        Self::from_parts(SO3::from_unit_quaternion(rotation), translation)
    }

    /// Constructs an SE(3) element from parts.
    pub fn from_parts(rotation: SO3, translation: Vector3<f64>) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// Returns the stored rotation component.
    pub fn rotation(&self) -> SO3 {
        self.rotation.clone()
    }

    /// Returns the stored translation component.
    pub fn translation(&self) -> Vector3<f64> {
        self.translation
    }

    /// Converts the transform into an isometry.
    pub fn to_isometry(&self) -> TransformSE3 {
        TransformSE3::from_parts(
            Translation3::from(self.translation),
            self.rotation.unit_quaternion().clone(),
        )
    }

    /// Builds an SE(3) element from an isometry.
    pub fn from_isometry(transform: &TransformSE3) -> Self {
        let rotation = SO3::from_unit_quaternion(transform.rotation.clone());
        let translation = transform.translation.vector;
        Self::from_parts(rotation, translation)
    }

    /// Computes the SE(3) exponential map of the supplied twist.
    pub fn exp(twist: &Twist) -> Self {
        let omega = twist.omega;
        let v = twist.v;
        let theta = omega.norm();

        log::trace!(target: "arbit_core::math", "SE3 exp: theta={:.6}, omega={:.4?}, v={:.4?}", theta, omega, v);

        let rotation = SO3::exp(&omega);
        let omega_hat = skew_symmetric(&omega);
        let identity = Matrix3::identity();

        let translation = if theta < 1e-9 {
            let omega_hat_sq = omega_hat * omega_hat;
            let v_transform = identity + 0.5 * omega_hat + (1.0 / 6.0) * omega_hat_sq;
            v_transform * v
        } else {
            let theta_sq = theta * theta;
            let omega_hat_sq = omega_hat * omega_hat;
            let s = (1.0 - theta.cos()) / theta_sq;
            let c = (theta - theta.sin()) / (theta_sq * theta);
            let v_transform = identity + s * omega_hat + c * omega_hat_sq;
            v_transform * v
        };

        Self::from_parts(rotation, translation)
    }

    /// Computes the logarithm map of this SE(3) transform.
    pub fn log(&self) -> Twist {
        let omega = self.rotation.log();
        let theta = omega.norm();
        let omega_hat = skew_symmetric(&omega);
        let identity = Matrix3::identity();
        let translation = self.translation;

        log::trace!(target: "arbit_core::math", "SE3 log: theta={:.6}, translation={:.4?}", theta, translation);

        let v = if theta < 1e-9 {
            let omega_hat_sq = omega_hat * omega_hat;
            let v_inverse = identity - 0.5 * omega_hat + (1.0 / 12.0) * omega_hat_sq;
            v_inverse * translation
        } else {
            let theta_sq = theta * theta;
            let omega_hat_sq = omega_hat * omega_hat;
            let s = (1.0 - theta.cos()) / theta_sq;
            let c = (theta - theta.sin()) / (theta_sq * theta);
            let v_transform = identity + s * omega_hat + c * omega_hat_sq;
            let v_inverse = v_transform
                .try_inverse()
                .expect("SE(3) V matrix should be invertible for finite theta");
            v_inverse * translation
        };

        Twist::new(omega, v)
    }

    /// Computes the logarithm map for an isometry.
    pub fn log_map(transform: &TransformSE3) -> Twist {
        Self::from_isometry(transform).log()
    }
}

impl From<SE3> for TransformSE3 {
    fn from(transform: SE3) -> Self {
        transform.to_isometry()
    }
}

impl From<&SE3> for TransformSE3 {
    fn from(transform: &SE3) -> Self {
        transform.to_isometry()
    }
}

impl From<TransformSE3> for SE3 {
    fn from(transform: TransformSE3) -> Self {
        Self::from_isometry(&transform)
    }
}

/// Represents a twist in se(3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Twist {
    pub omega: Vector3<f64>,
    pub v: Vector3<f64>,
}

impl Twist {
    pub fn new(omega: Vector3<f64>, v: Vector3<f64>) -> Self {
        Self { omega, v }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn se3_round_trip_log_exp() {
        let twist = Twist::new(Vector3::new(0.1, -0.2, 0.3), Vector3::new(0.5, -0.1, 0.2));
        let transform = SE3::exp(&twist);
        let recovered = transform.log();

        assert_relative_eq!(twist.omega, recovered.omega, epsilon = 1e-9);
        assert_relative_eq!(twist.v, recovered.v, epsilon = 1e-9);
    }

    #[test]
    fn se3_identity_round_trip() {
        let identity = SE3::identity();
        let twist = identity.log();
        assert_relative_eq!(twist.omega, Vector3::zeros(), epsilon = 1e-12);
        assert_relative_eq!(twist.v, Vector3::zeros(), epsilon = 1e-12);

        let composed = SE3::exp(&twist);
        assert_relative_eq!(
            TransformSE3::from(&composed).to_homogeneous(),
            TransformSE3::from(&identity).to_homogeneous(),
            epsilon = 1e-12
        );
    }
}
