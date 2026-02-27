use crate::db::{KeyframeDescriptor, KeyframeEntry, KeyframeIndex};
use crate::math::se3::TransformSE3;
use image::Rgba;
use log::{info, warn};
use nalgebra::{Matrix4, Point2, Point3, Translation3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use tracing::trace;

const GRID_SIZE: usize = 4;
const CELL_COUNT: usize = GRID_SIZE * GRID_SIZE;
const MIN_TRANSLATION_DELTA: f64 = 0.02;
const MIN_ROTATION_DELTA_RAD: f64 = 5_f64.to_radians();

#[derive(Debug)]
pub enum MapIoError {
    InvalidHeader,
    UnsupportedVersion(u16),
    ChecksumMismatch { expected: u32, actual: u32 },
    Serialization(bincode::Error),
    Deserialization(bincode::Error),
}

impl fmt::Display for MapIoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MapIoError::InvalidHeader => write!(f, "map header is invalid"),
            MapIoError::UnsupportedVersion(v) => write!(f, "unsupported map version {v}"),
            MapIoError::ChecksumMismatch { expected, actual } => write!(
                f,
                "map checksum mismatch (expected {expected:#010x}, got {actual:#010x})"
            ),
            MapIoError::Serialization(err) => write!(f, "failed to serialize map: {err}"),
            MapIoError::Deserialization(err) => write!(f, "failed to deserialize map: {err}"),
        }
    }
}

impl std::error::Error for MapIoError {}

#[derive(Debug, Clone)]
pub struct MapLandmark {
    pub id: u64,
    pub world_xyz: Point3<f64>,
    pub color: Option<Rgba<u8>>,
}

#[derive(Debug, Clone)]
pub struct Anchor {
    pub id: u64,
    pub pose_wc: TransformSE3,
    pub created_from_keyframe: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct BinaryFeatureDescriptor {
    pub data: [u8; 32],
}

impl BinaryFeatureDescriptor {
    pub fn new(data: [u8; 32]) -> Self {
        Self { data }
    }
}

#[derive(Debug, Clone)]
pub struct KeyframeFeature {
    pub norm_xy: Point2<f64>,
    pub world_xyz: Point3<f64>,
    pub landmark_id: u64,
    pub cell: usize,
    pub color: Option<Rgba<u8>>,
    pub descriptor: BinaryFeatureDescriptor,
}

impl KeyframeFeature {
    pub fn cell(&self) -> usize {
        self.cell
    }
}

#[derive(Debug, Clone)]
pub struct KeyframeData {
    pub id: u64,
    pub pose: TransformSE3,
    features: Vec<KeyframeFeature>,
    cell_lookup: Vec<Vec<usize>>,
}

impl KeyframeData {
    pub fn feature_count(&self) -> usize {
        self.features.len()
    }

    pub fn features(&self) -> &[KeyframeFeature] {
        &self.features
    }

    pub fn features_in_cell(&self, cell: usize) -> impl Iterator<Item = &KeyframeFeature> {
        self.cell_lookup
            .get(cell)
            .into_iter()
            .flat_map(|indices| indices.iter())
            .map(|idx| &self.features[*idx])
    }
}

#[derive(Debug, Default, Clone)]
pub struct WorldMap {
    keyframes: HashMap<u64, KeyframeData>,
    keyframe_index: KeyframeIndex,
    landmarks: HashMap<u64, MapLandmark>,
    anchors: HashMap<u64, Anchor>,
    next_keyframe_id: u64,
    next_landmark_id: u64,
    next_anchor_id: u64,
    last_keyframe_pose: Option<TransformSE3>,
}

impl WorldMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty()
    }

    pub fn keyframe_count(&self) -> usize {
        self.keyframes.len()
    }

    pub fn should_insert_keyframe(&self, pose: &TransformSE3) -> bool {
        match &self.last_keyframe_pose {
            None => true,
            Some(last_pose) => {
                let delta = last_pose.inverse() * pose;
                let translation_delta = delta.translation.vector.norm();
                let rotation_delta = delta.rotation.angle();
                translation_delta > MIN_TRANSLATION_DELTA || rotation_delta > MIN_ROTATION_DELTA_RAD
            }
        }
    }

    pub fn insert_keyframe(
        &mut self,
        pose: TransformSE3,
        descriptor: KeyframeDescriptor,
        // (norm_xy, world_xyz)
        features: Vec<(
            Point2<f64>,
            Point3<f64>,
            Option<Rgba<u8>>,
            BinaryFeatureDescriptor,
        )>,
    ) -> Option<u64> {
        trace!(target: "arbit_core::map", "Inserting keyframe with pose: {:?}", pose);
        if features.is_empty() {
            warn!(target: "arbit_core::map", "Skipping keyframe insert due to empty feature set");
            return None;
        }

        let id = self.next_keyframe_id;
        self.next_keyframe_id = self.next_keyframe_id.saturating_add(1);

        let feature_count = features.len();
        let mut keyframe_features = Vec::with_capacity(feature_count);
        let mut cell_lookup: Vec<Vec<usize>> = vec![Vec::new(); CELL_COUNT];

        for (norm_xy, world_xyz, color, descriptor) in features {
            let landmark_id = self.next_landmark_id;
            self.next_landmark_id = self.next_landmark_id.saturating_add(1);

            let cell = cell_for_normalized(&norm_xy);
            self.landmarks.insert(
                landmark_id,
                MapLandmark {
                    id: landmark_id,
                    world_xyz,
                    color,
                },
            );

            let feature_index = keyframe_features.len();
            cell_lookup[cell].push(feature_index);

            keyframe_features.push(KeyframeFeature {
                norm_xy,
                world_xyz,
                landmark_id,
                cell,
                color,
                descriptor,
            });
        }

        let keyframe = KeyframeData {
            id,
            pose: pose.clone(),
            features: keyframe_features,
            cell_lookup,
        };

        let entry = KeyframeEntry {
            id,
            pose_wc: pose.clone(),
            descriptor: descriptor.clone(),
        };
        self.keyframe_index.insert(entry);
        self.keyframes.insert(id, keyframe);
        self.last_keyframe_pose = Some(pose);

        info!(
            target: "arbit_core::map",
            "Inserted keyframe {} ({} landmarks)",
            id,
            feature_count
        );
        Some(id)
    }

    pub fn insert_keyframe_with_id(
        &mut self,
        id: u64,
        pose_wc: TransformSE3,
        descriptor: KeyframeDescriptor,
        features: Vec<(
            Point2<f64>,
            Point3<f64>,
            u64,
            Option<Rgba<u8>>,
            BinaryFeatureDescriptor,
        )>,
    ) {
        if features.is_empty() {
            warn!(
                target: "arbit_core::map",
                "Skipping keyframe insert due to empty feature set"
            );
            return;
        }

        let mut keyframe_features = Vec::with_capacity(features.len());
        let mut cell_lookup: Vec<Vec<usize>> = vec![Vec::new(); CELL_COUNT];
        for (norm_xy, world_xyz, landmark_id, color, descriptor) in features {
            let cell = cell_for_normalized(&norm_xy);
            let feature_index = keyframe_features.len();
            cell_lookup[cell].push(feature_index);

            self.next_landmark_id = self.next_landmark_id.max(landmark_id.saturating_add(1));

            self.landmarks.insert(
                landmark_id,
                MapLandmark {
                    id: landmark_id,
                    world_xyz,
                    color,
                },
            );

            keyframe_features.push(KeyframeFeature {
                norm_xy,
                world_xyz,
                landmark_id,
                cell,
                color,
                descriptor,
            });
        }

        let landmark_count = keyframe_features.len();
        let keyframe = KeyframeData {
            id,
            pose: pose_wc.clone(),
            features: keyframe_features,
            cell_lookup,
        };

        let entry = KeyframeEntry {
            id,
            pose_wc: pose_wc.clone(),
            descriptor: descriptor.clone(),
        };

        self.keyframe_index.insert(entry);
        self.keyframes.insert(id, keyframe);
        self.next_keyframe_id = self.next_keyframe_id.max(id.saturating_add(1));
        self.last_keyframe_pose = Some(pose_wc);

        info!(
            target: "arbit_core::map",
            "Loaded keyframe {} ({} landmarks)",
            id,
            landmark_count
        );
    }

    pub fn landmark_count(&self) -> usize {
        self.landmarks.len()
    }

    pub fn landmark(&self, id: u64) -> Option<&MapLandmark> {
        self.landmarks.get(&id)
    }

    pub fn landmarks_iter(&self) -> impl Iterator<Item = &MapLandmark> {
        self.landmarks.values()
    }

    pub fn add_landmark(&mut self, position: Point3<f64>, color: Option<Rgba<u8>>) -> u64 {
        let id = self.next_landmark_id;
        self.next_landmark_id = self.next_landmark_id.saturating_add(1);
        self.landmarks.insert(
            id,
            MapLandmark {
                id,
                world_xyz: position,
                color,
            },
        );
        id
    }

    pub fn keyframe(&self, id: u64) -> Option<&KeyframeData> {
        self.keyframes.get(&id)
    }

    pub fn query<'a>(&'a self, descriptor: &KeyframeDescriptor, k: usize) -> Vec<&'a KeyframeData> {
        let mut results = Vec::new();
        for entry in self.keyframe_index.query(descriptor, k) {
            if let Some(kf) = self.keyframes.get(&entry.id) {
                results.push(kf);
            } else {
                warn!(
                    target: "arbit_core::map",
                    "Keyframe {} present in index but missing from storage",
                    entry.id
                );
            }
        }
        results
    }

    pub fn keyframes(&self) -> impl Iterator<Item = &KeyframeData> {
        self.keyframes.values()
    }

    pub fn last_keyframe_pose(&self) -> Option<&TransformSE3> {
        self.last_keyframe_pose.as_ref()
    }

    pub fn max_keyframe_id(&self) -> Option<u64> {
        self.keyframes.keys().max().copied()
    }

    pub fn create_anchor(&mut self, pose: TransformSE3, keyframe_hint: Option<u64>) -> u64 {
        let id = self.next_anchor_id;
        self.next_anchor_id = self.next_anchor_id.saturating_add(1);
        let anchor = Anchor {
            id,
            pose_wc: pose,
            created_from_keyframe: keyframe_hint,
        };
        self.anchors.insert(id, anchor);
        info!(target: "arbit_core::map", "Created anchor {}", id);
        id
    }

    pub fn resolve_anchor(&self, id: u64) -> Option<&Anchor> {
        self.anchors.get(&id)
    }

    pub fn update_anchor_pose(&mut self, id: u64, pose: TransformSE3) -> bool {
        if let Some(anchor) = self.anchors.get_mut(&id) {
            anchor.pose_wc = pose;
            true
        } else {
            false
        }
    }

    pub fn anchor_count(&self) -> usize {
        self.anchors.len()
    }

    pub fn anchors(&self) -> impl Iterator<Item = &Anchor> {
        self.anchors.values()
    }

    pub fn remove_anchor(&mut self, id: u64) -> bool {
        self.anchors.remove(&id).is_some()
    }

    // pub fn to_bytes(&self) -> Result<Vec<u8>, MapIoError> {
    //     let mut keyframes: Vec<SerializableKeyframe> = self
    //         .keyframes
    //         .values()
    //         .map(|kf| SerializableKeyframe {
    //             id: kf.id,
    //             pose_wc: pose_to_array(&kf.pose),
    //             descriptor: kf.descriptor.as_slice().to_vec(),
    //             features: kf
    //                 .features
    //                 .iter()
    //                 .map(|feature| SerializableFeature {
    //                     landmark_id: feature.landmark_id,
    //                     norm_xy: [feature.norm_xy.x, feature.norm_xy.y],
    //                     world_xyz: [
    //                         feature.world_xyz.x,
    //                         feature.world_xyz.y,
    //                         feature.world_xyz.z,
    //                     ],
    //                     color: feature.color.map(|c| c.0).unwrap_or([255, 255, 255, 255]),
    //                 })
    //                 .collect(),
    //         })
    //         .collect();

    //     keyframes.sort_by_key(|kf| kf.id);

    //     let mut anchors: Vec<SerializableAnchor> = self
    //         .anchors
    //         .values()
    //         .map(|anchor| SerializableAnchor {
    //             id: anchor.id,
    //             pose_wc: pose_to_array(&anchor.pose_wc),
    //             created_from_keyframe: anchor.created_from_keyframe,
    //         })
    //         .collect();
    //     anchors.sort_by_key(|anchor| anchor.id);

    //     let payload = SerializableMap {
    //         version: MAP_VERSION,
    //         provider_caps: 0,
    //         world_basis: WORLD_BASIS_Y_UP,
    //         keyframes,
    //         anchors,
    //     };

    //     let payload_bytes = bincode::serialize(&payload).map_err(MapIoError::Serialization)?;
    //     let mut hasher = Hasher::new();
    //     hasher.update(&payload_bytes);
    //     let checksum = hasher.finalize();

    //     let mut buffer = Vec::with_capacity(
    //         MAP_MAGIC.len() + 2 + std::mem::size_of::<u32>() * 2 + payload_bytes.len(),
    //     );
    //     buffer.extend_from_slice(MAP_MAGIC);
    //     buffer.extend_from_slice(&MAP_VERSION.to_le_bytes());
    //     buffer.extend_from_slice(&(payload_bytes.len() as u32).to_le_bytes());
    //     buffer.extend_from_slice(&checksum.to_le_bytes());
    //     buffer.extend_from_slice(&payload_bytes);
    //     Ok(buffer)
    // }

    // pub fn from_bytes(bytes: &[u8]) -> Result<Self, MapIoError> {
    //     if bytes.len() < MAP_MAGIC.len() + 2 + 4 + 4 {
    //         return Err(MapIoError::InvalidHeader);
    //     }

    //     let (magic, remainder) = bytes.split_at(MAP_MAGIC.len());
    //     if magic != MAP_MAGIC {
    //         return Err(MapIoError::InvalidHeader);
    //     }

    //     let (version_bytes, remainder) = remainder.split_at(2);
    //     let version = u16::from_le_bytes([version_bytes[0], version_bytes[1]]);
    //     if version != MAP_VERSION {
    //         return Err(MapIoError::UnsupportedVersion(version));
    //     }

    //     let (length_bytes, remainder) = remainder.split_at(4);
    //     let payload_len = u32::from_le_bytes(length_bytes.try_into().unwrap()) as usize;

    //     let (checksum_bytes, remainder) = remainder.split_at(4);
    //     let expected_checksum = u32::from_le_bytes(checksum_bytes.try_into().unwrap());

    //     if remainder.len() < payload_len {
    //         return Err(MapIoError::InvalidHeader);
    //     }

    //     let (payload_bytes, tail) = remainder.split_at(payload_len);
    //     if !tail.is_empty() {
    //         return Err(MapIoError::InvalidHeader);
    //     }

    //     let mut hasher = Hasher::new();
    //     hasher.update(payload_bytes);
    //     let actual_checksum = hasher.finalize();
    //     if actual_checksum != expected_checksum {
    //         return Err(MapIoError::ChecksumMismatch {
    //             expected: expected_checksum,
    //             actual: actual_checksum,
    //         });
    //     }

    //     let payload: SerializableMap =
    //         bincode::deserialize(payload_bytes).map_err(MapIoError::Deserialization)?;
    //     if payload.version != MAP_VERSION {
    //         return Err(MapIoError::UnsupportedVersion(payload.version));
    //     }
    //     if payload.world_basis != WORLD_BASIS_Y_UP {
    //         return Err(MapIoError::InvalidHeader);
    //     }

    //     let mut map = WorldMap::new();

    //     for keyframe in payload.keyframes {
    //         let pose = pose_from_array(keyframe.pose_wc);
    //         let descriptor = KeyframeDescriptor::from_slice(&keyframe.descriptor);
    //         let features = keyframe
    //             .features
    //             .into_iter()
    //             .map(|feature| {
    //                 (
    //                     Point2::new(feature.norm_xy[0], feature.norm_xy[1]),
    //                     Point3::new(
    //                         feature.world_xyz[0],
    //                         feature.world_xyz[1],
    //                         feature.world_xyz[2],
    //                     ),
    //                     feature.landmark_id,
    //                     Some(Rgba::from(feature.color)),
    //                     feature.descriptor,
    //                 )
    //             })
    //             .collect();
    //         map.insert_keyframe_with_id(keyframe.id, pose, descriptor, features);
    //     }

    //     for anchor in payload.anchors {
    //         let pose = pose_from_array(anchor.pose_wc);
    //         map.anchors.insert(
    //             anchor.id,
    //             Anchor {
    //                 id: anchor.id,
    //                 pose_wc: pose.clone(),
    //                 created_from_keyframe: anchor.created_from_keyframe,
    //             },
    //         );
    //         map.next_anchor_id = map.next_anchor_id.max(anchor.id.saturating_add(1));
    //     }

    //     if let Some(last) = map.keyframes.values().max_by_key(|kf| kf.id) {
    //         map.last_keyframe_pose = Some(last.pose.clone());
    //     }

    //     Ok(map)
    // }

    // pub fn load_from_bytes(&mut self, bytes: &[u8]) -> Result<(), MapIoError> {
    //     let loaded = WorldMap::from_bytes(bytes)?;
    //     *self = loaded;
    //     Ok(())
    // }
}

#[derive(Serialize, Deserialize)]
struct SerializableMap {
    version: u16,
    provider_caps: u32,
    world_basis: u8,
    keyframes: Vec<SerializableKeyframe>,
    anchors: Vec<SerializableAnchor>,
}

#[derive(Serialize, Deserialize)]
struct SerializableKeyframe {
    id: u64,
    pose_wc: [f64; 16],
    descriptor: Vec<f32>,
    features: Vec<SerializableFeature>,
}

#[derive(Serialize, Deserialize)]
struct SerializableFeature {
    landmark_id: u64,
    norm_xy: [f64; 2],
    world_xyz: [f64; 3],
    color: [u8; 4],
}

#[derive(Serialize, Deserialize)]
struct SerializableAnchor {
    id: u64,
    pose_wc: [f64; 16],
    created_from_keyframe: Option<u64>,
}

fn pose_to_array(pose_wc: &TransformSE3) -> [f64; 16] {
    let matrix = pose_wc.to_homogeneous();
    let mut out = [0.0; 16];
    for row in 0..4 {
        for col in 0..4 {
            out[row * 4 + col] = matrix[(row, col)];
        }
    }
    out
}

fn pose_from_array(elements: [f64; 16]) -> TransformSE3 {
    let matrix = Matrix4::from_row_slice(&elements);
    let rotation = matrix.fixed_view::<3, 3>(0, 0).into_owned();
    let translation = Vector3::new(matrix[(0, 3)], matrix[(1, 3)], matrix[(2, 3)]);
    TransformSE3::from_parts(
        Translation3::from(translation),
        UnitQuaternion::from_matrix(&rotation),
    )
}

pub fn cell_for_normalized(point: &Point2<f64>) -> usize {
    let x = (point.x.clamp(0.0, 0.999_9) * GRID_SIZE as f64) as usize;
    let y = (point.y.clamp(0.0, 0.999_9) * GRID_SIZE as f64) as usize;
    y * GRID_SIZE + x
}

pub fn grid_size() -> usize {
    GRID_SIZE
}
