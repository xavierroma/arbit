use nalgebra::Vector2;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ForwardBackwardMetrics {
    pub forward_error: f32,
    pub backward_error: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackHealth {
    pub valid: bool,
    pub metrics: ForwardBackwardMetrics,
    pub age: u32,
}

impl TrackHealth {
    pub fn evaluate(
        predicted: Vector2<f32>,
        observed: Vector2<f32>,
        backward: Option<Vector2<f32>>,
        age: u32,
        max_forward: f32,
        max_backward: f32,
    ) -> Self {
        let forward_error = (observed - predicted).norm();
        let backward_error = backward.map(|b| (b - predicted).norm());
        let valid = forward_error <= max_forward
            && backward_error
                .map(|err| err <= max_backward)
                .unwrap_or(true);
        Self {
            valid,
            metrics: ForwardBackwardMetrics {
                forward_error,
                backward_error,
            },
            age,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn health_accepts_small_errors() {
        let predicted = Vector2::new(10.0, 10.0);
        let observed = Vector2::new(10.3, 10.2);
        let backward = Some(Vector2::new(9.9, 9.8));
        let health = TrackHealth::evaluate(predicted, observed, backward, 5, 1.0, 1.2);
        assert!(health.valid);
        assert_relative_eq!(health.metrics.forward_error, 0.36055514, epsilon = 1e-5);
        assert!(health.metrics.backward_error.unwrap() <= 1.2);
        assert_eq!(health.age, 5);
    }

    #[test]
    fn health_rejects_large_forward_error() {
        let predicted = Vector2::new(0.0, 0.0);
        let observed = Vector2::new(3.0, 4.0);
        let health = TrackHealth::evaluate(predicted, observed, None, 1, 4.0, 1.0);
        assert!(!health.valid);
        assert_relative_eq!(health.metrics.forward_error, 5.0, epsilon = 1e-6);
    }
}
