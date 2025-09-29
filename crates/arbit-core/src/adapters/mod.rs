use crate::math::se3::TransformSE3;

/// SceneKit and RealityKit already use a right-handed, Y-up, Z-forward convention that matches the core.
/// The conversion is therefore the identity.
pub fn world_from_scenekit(transform: &TransformSE3) -> TransformSE3 {
    transform.clone()
}

pub fn scenekit_from_world(transform: &TransformSE3) -> TransformSE3 {
    transform.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{Isometry3, Vector3};

    #[test]
    fn scenekit_identity_conversion() {
        let pose = Isometry3::new(Vector3::new(0.0, 1.0, 2.0), Vector3::new(0.2, 0.1, -0.1));
        let converted = world_from_scenekit(&pose);
        assert_relative_eq!(
            pose.translation.vector,
            converted.translation.vector,
            epsilon = 1e-12
        );
        assert_relative_eq!(
            pose.rotation.to_rotation_matrix().into_inner(),
            converted.rotation.to_rotation_matrix().into_inner(),
            epsilon = 1e-12
        );
    }
}
