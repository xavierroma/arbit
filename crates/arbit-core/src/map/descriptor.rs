use super::{CELL_COUNT, cell_for_normalized};
use crate::db::KeyframeDescriptor;
use crate::track::{TrackObservation, TrackOutcome};
use nalgebra::Vector2;

#[derive(Debug, Clone, Copy)]
pub enum DescriptorSample {
    Initial,
    Refined,
}

#[derive(Debug, Clone)]
pub struct DescriptorInfo {
    pub descriptor: KeyframeDescriptor,
    pub normalized_points: Vec<(usize, Vector2<f32>)>,
    pub total_tracks: usize,
    pub converged_tracks: usize,
    pub average_residual: f32,
}

impl DescriptorInfo {
    pub fn inlier_ratio(&self) -> f32 {
        if self.total_tracks == 0 {
            0.0
        } else {
            self.converged_tracks as f32 / self.total_tracks as f32
        }
    }
}

pub fn build_descriptor(
    tracks: &[TrackObservation],
    width: u32,
    height: u32,
    sample: DescriptorSample,
) -> DescriptorInfo {
    let mut histogram = vec![0.0f32; CELL_COUNT];
    let mut normalized_points = Vec::new();
    let mut residual_sum = 0.0f32;

    let width = width.max(1) as f32;
    let height = height.max(1) as f32;

    for (index, track) in tracks.iter().enumerate() {
        if !matches!(track.outcome, TrackOutcome::Converged) {
            continue;
        }

        let point = match sample {
            DescriptorSample::Initial => track.initial,
            DescriptorSample::Refined => track.refined,
        };

        let normalized = Vector2::new(point.x / width, point.y / height);
        let normalized = Vector2::new(
            normalized.x.clamp(0.0, 0.999_9),
            normalized.y.clamp(0.0, 0.999_9),
        );
        let cell = cell_for_normalized(&normalized);
        histogram[cell] += 1.0;
        normalized_points.push((index, normalized));
        residual_sum += track.residual;
    }

    let converged_tracks = normalized_points.len();
    let total_tracks = tracks.len();
    let average_residual = if converged_tracks > 0 {
        residual_sum / converged_tracks as f32
    } else {
        0.0
    };

    let norm = histogram.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-6 {
        for value in histogram.iter_mut() {
            *value /= norm;
        }
    }

    let inlier_ratio = if total_tracks == 0 {
        0.0
    } else {
        converged_tracks as f32 / total_tracks as f32
    };

    histogram.push(inlier_ratio);
    histogram.push(average_residual);
    histogram.push(total_tracks as f32);

    DescriptorInfo {
        descriptor: KeyframeDescriptor::from_vec(histogram),
        normalized_points,
        total_tracks,
        converged_tracks,
        average_residual,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::{TrackObservation, TrackOutcome};
    use nalgebra::Vector2;

    fn track(initial: Vector2<f32>, refined: Vector2<f32>, residual: f32) -> TrackObservation {
        TrackObservation {
            initial,
            refined,
            iterations: 5,
            residual,
            outcome: TrackOutcome::Converged,
        }
    }

    #[test]
    fn descriptor_has_expected_length() {
        let tracks = vec![
            track(Vector2::new(10.0, 20.0), Vector2::new(11.0, 21.0), 0.5),
            track(Vector2::new(30.0, 40.0), Vector2::new(31.0, 41.0), 0.25),
        ];
        let info = build_descriptor(&tracks, 64, 64, DescriptorSample::Initial);
        assert_eq!(info.descriptor.len(), CELL_COUNT + 3);
        assert_eq!(info.converged_tracks, 2);
        assert!(info.inlier_ratio() > 0.0);
        assert!(info.average_residual > 0.0);
    }
}
