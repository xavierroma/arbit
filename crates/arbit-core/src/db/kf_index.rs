use crate::math::se3::TransformSE3;
use log::{debug, warn};
use nalgebra::DVector;

#[derive(Debug, Clone)]
pub struct KeyframeDescriptor {
    data: DVector<f32>,
}

impl KeyframeDescriptor {
    pub fn from_vec(data: Vec<f32>) -> Self {
        Self {
            data: DVector::from_row_slice(&data),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }
}

#[derive(Debug, Clone)]
pub struct KeyframeEntry {
    pub id: u64,
    pub pose: TransformSE3,
    pub descriptor: KeyframeDescriptor,
}

#[derive(Debug, Default, Clone)]
pub struct KeyframeIndex {
    keyframes: Vec<KeyframeEntry>,
}

impl KeyframeIndex {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
        }
    }

    pub fn insert(&mut self, entry: KeyframeEntry) {
        debug!(target: "arbit_core::db", "Inserting keyframe {} with descriptor length {}", entry.id, entry.descriptor.len());
        self.keyframes.push(entry);
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    pub fn len(&self) -> usize {
        self.keyframes.len()
    }

    pub fn query(&self, descriptor: &KeyframeDescriptor, k: usize) -> Vec<&KeyframeEntry> {
        assert!(descriptor.len() > 0, "Query descriptor must not be empty");
        debug!(target: "arbit_core::db", "Querying keyframe database: {} keyframes, descriptor length {}, requesting {} results",
               self.keyframes.len(), descriptor.len(), k);

        let mut scored: Vec<_> = self
            .keyframes
            .iter()
            .filter(|kf| {
                if kf.descriptor.len() != descriptor.len() {
                    warn!(target: "arbit_core::db", "Keyframe {} has descriptor length {} but query has {}",
                          kf.id, kf.descriptor.len(), descriptor.len());
                    false
                } else {
                    true
                }
            })
            .map(|kf| {
                let score = cosine_similarity(kf.descriptor.as_slice(), descriptor.as_slice());
                (score, kf)
            })
            .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let results: Vec<&KeyframeEntry> = scored.into_iter().take(k).map(|(_, kf)| kf).collect();

        debug!(target: "arbit_core::db", "Query returned {} results (best score: {:.4})",
               results.len(), results.first().map(|kf| cosine_similarity(kf.descriptor.as_slice(), descriptor.as_slice())).unwrap_or(0.0));

        results
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        dot += va * vb;
        norm_a += va * va;
        norm_b += vb * vb;
    }
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::se3::TransformSE3;
    use nalgebra::{Translation3, UnitQuaternion};

    fn pose() -> TransformSE3 {
        TransformSE3::from_parts(Translation3::new(0.0, 0.0, 0.0), UnitQuaternion::identity())
    }

    #[test]
    fn index_returns_most_similar() {
        let mut index = KeyframeIndex::new();
        for id in 0..10u64 {
            let descriptor = KeyframeDescriptor::from_vec(vec![id as f32, 1.0, 0.5]);
            index.insert(KeyframeEntry {
                id,
                pose: pose(),
                descriptor,
            });
        }

        let query = KeyframeDescriptor::from_vec(vec![9.0, 1.0, 0.5]);
        let result = index.query(&query, 3);
        assert_eq!(result.first().unwrap().id, 9);
        assert_eq!(result.len(), 3);
    }
}
