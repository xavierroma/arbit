use crate::track::feat_descriptor::{DescriptorBuffer, FeatDescriptor};
use std::collections::HashMap;

/// A feature match between a query descriptor and a train descriptor.
pub struct Match {
    /// Index of the query descriptor
    pub query_idx: usize,
    /// Index of the train descriptor
    pub train_idx: usize,
    /// Hamming distance between the descriptors
    pub distance: u32,
}

pub struct HammingFeatMatcher {
    /// Enable cross-check matching.
    ///
    /// When true, a match is accepted only if the best match from query→train
    /// and the best match from train→query are mutually consistent (i.e., both directions agree).
    /// This generally makes matches more robust but can reduce the total number of putative matches.
    pub cross_check: bool,

    /// Maximum allowed Hamming distance for a match to be accepted.
    /// If None, no distance filtering is applied.
    /// Typical values: 50-80 for ORB descriptors (256 bits).
    pub max_distance: Option<u32>,

    /// Ratio test threshold (Lowe's ratio test).
    /// If Some(ratio), only accept matches where best_distance < ratio * second_best_distance.
    /// Typical value: 0.7-0.8 for more distinctive matches.
    /// If None, no ratio test is applied.
    pub ratio_threshold: Option<f32>,
}

impl Default for HammingFeatMatcher {
    fn default() -> Self {
        HammingFeatMatcher {
            cross_check: true,
            max_distance: None,
            ratio_threshold: None,
        }
    }
}

impl HammingFeatMatcher {
    /// Match query descriptors to train descriptors.
    pub fn match_feats<D: DescriptorBuffer>(
        &self,
        query: &[FeatDescriptor<D>],
        train: &[FeatDescriptor<D>],
    ) -> Vec<Match> {
        if self.cross_check {
            self.match_with_cross_check(query, train)
        } else {
            self.match_forward_only(query, train)
        }
    }

    /// Match query descriptors to train descriptors (forward direction only).
    fn match_forward_only<D: DescriptorBuffer>(
        &self,
        query: &[FeatDescriptor<D>],
        train: &[FeatDescriptor<D>],
    ) -> Vec<Match> {
        let mut matches = Vec::new();

        for (query_idx, query_desc) in query.iter().enumerate() {
            let mut best_distance = u32::MAX;
            let mut second_best_distance = u32::MAX;
            let mut best_train_idx = None;

            // Find best and second-best matches for this query descriptor
            for (train_idx, train_desc) in train.iter().enumerate() {
                let distance =
                    hamming_distance(query_desc.data.as_bytes(), train_desc.data.as_bytes());

                if distance < best_distance {
                    second_best_distance = best_distance;
                    best_distance = distance;
                    best_train_idx = Some(train_idx);
                } else if distance < second_best_distance {
                    second_best_distance = distance;
                }
            }

            // Apply filters and add match if it passes
            if let Some(train_idx) = best_train_idx
                && self.passes_filters(best_distance, second_best_distance)
            {
                matches.push(Match {
                    query_idx,
                    train_idx,
                    distance: best_distance,
                });
            }
        }

        matches
    }

    /// Match with cross-check: only accept mutually consistent matches.
    fn match_with_cross_check<D: DescriptorBuffer>(
        &self,
        query: &[FeatDescriptor<D>],
        train: &[FeatDescriptor<D>],
    ) -> Vec<Match> {
        // Build forward matches: query -> train
        let mut forward_matches: HashMap<usize, (usize, u32, u32)> = HashMap::new();
        for (query_idx, query_desc) in query.iter().enumerate() {
            let mut best_distance = u32::MAX;
            let mut second_best_distance = u32::MAX;
            let mut best_train_idx = None;

            for (train_idx, train_desc) in train.iter().enumerate() {
                let distance =
                    hamming_distance(query_desc.data.as_bytes(), train_desc.data.as_bytes());

                if distance < best_distance {
                    second_best_distance = best_distance;
                    best_distance = distance;
                    best_train_idx = Some(train_idx);
                } else if distance < second_best_distance {
                    second_best_distance = distance;
                }
            }

            if let Some(train_idx) = best_train_idx {
                forward_matches.insert(query_idx, (train_idx, best_distance, second_best_distance));
            }
        }

        // Build backward matches: train -> query
        let mut backward_matches: HashMap<usize, (usize, u32, u32)> = HashMap::new();
        for (train_idx, train_desc) in train.iter().enumerate() {
            let mut best_distance = u32::MAX;
            let mut second_best_distance = u32::MAX;
            let mut best_query_idx = None;

            for (query_idx, query_desc) in query.iter().enumerate() {
                let distance =
                    hamming_distance(query_desc.data.as_bytes(), train_desc.data.as_bytes());

                if distance < best_distance {
                    second_best_distance = best_distance;
                    best_distance = distance;
                    best_query_idx = Some(query_idx);
                } else if distance < second_best_distance {
                    second_best_distance = distance;
                }
            }

            if let Some(query_idx) = best_query_idx {
                backward_matches
                    .insert(train_idx, (query_idx, best_distance, second_best_distance));
            }
        }

        // Cross-check: keep only mutually consistent matches that pass filters
        let mut matches = Vec::new();
        for (query_idx, (train_idx, distance, second_best)) in forward_matches.iter() {
            if let Some((back_query_idx, _, _)) = backward_matches.get(train_idx)
                && *back_query_idx == *query_idx
                && self.passes_filters(*distance, *second_best)
            {
                matches.push(Match {
                    query_idx: *query_idx,
                    train_idx: *train_idx,
                    distance: *distance,
                });
            }
        }

        matches
    }

    /// Check if a match passes the configured filters.
    fn passes_filters(&self, best_distance: u32, second_best_distance: u32) -> bool {
        // Check max distance threshold
        if let Some(max_dist) = self.max_distance
            && best_distance > max_dist
        {
            return false;
        }

        // Check ratio test (Lowe's test)
        if let Some(ratio) = self.ratio_threshold
            && second_best_distance != u32::MAX
        {
            let actual_ratio = best_distance as f32 / second_best_distance as f32;
            if actual_ratio >= ratio {
                return false;
            }
        }

        true
    }
}

fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    let mut distance = 0;
    for i in 0..a.len() {
        distance += (a[i] ^ b[i]).count_ones();
    }
    distance
}
