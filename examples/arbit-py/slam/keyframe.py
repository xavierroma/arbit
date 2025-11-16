"""KeyFrame - Camera Pose + Observations."""
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import numpy as np
import cv2

if TYPE_CHECKING:
    from .map_point import MapPoint
else:
    from .map_point import MapPoint  # type: ignore


class KeyFrame:
    """Camera pose with image and feature observations.
    
    Core Responsibilities:
    - Store camera pose (SE(3) transformation)
    - Store image and extracted features (keypoints, descriptors)
    - Maintain bidirectional links to observed MapPoints
    - Manage covisibility graph (which keyframes share points)
    - Participate in spanning tree for pose graph optimization
    """
    
    _next_id = 0
    
    def __init__(self, image: np.ndarray, camera_matrix: np.ndarray,
                 pose_c_to_w: np.ndarray, keypoints: List[cv2.KeyPoint],
                 descriptors: np.ndarray):
        """Initialize a KeyFrame.
        
        Args:
            image: Grayscale image
            camera_matrix: 3x3 intrinsic matrix K
            pose_c_to_w: 4x4 SE(3) transformation (camera to world)
            keypoints: Detected keypoints
            descriptors: NxM descriptor matrix
        """
        self.id = KeyFrame._next_id
        KeyFrame._next_id += 1
        
        self.image = image.copy()
        self.camera_matrix = camera_matrix.copy()
        self.pose_c_to_w = pose_c_to_w.copy()  # 4x4 SE(3) transformation
        
        # Features
        self.keypoints = keypoints.copy() if isinstance(keypoints, list) else list(keypoints)
        self.descriptors = descriptors.copy()  # NxM descriptor matrix
        
        # Map point associations (bidirectional)
        # map_points[i] ↔ keypoints[i]
        self.map_points: List[Optional[MapPoint]] = [None] * len(keypoints)
        
        # Covisibility graph
        self.connected_keyframes: Dict[int, int] = {}  # {keyframe_id: num_shared_points}
        
        # Spanning tree
        self.parent: Optional['KeyFrame'] = None
        self.children: List['KeyFrame'] = []
        
        # State
        self.is_bad: bool = False
    
    # === Map Point Association ===
    
    def add_map_point(self, map_point: MapPoint, keypoint_idx: int) -> None:
        """Associate a map point with a keypoint (creates bidirectional link).
        
        Args:
            map_point: MapPoint to associate
            keypoint_idx: Index of the keypoint in this keyframe
        """
        if keypoint_idx < 0 or keypoint_idx >= len(self.map_points):
            raise ValueError(f"Invalid keypoint index: {keypoint_idx}")
        
        # Remove old association if exists
        old_mp = self.map_points[keypoint_idx]
        if old_mp is not None and old_mp.id != map_point.id:
            old_mp.remove_observation(self.id)
        
        # Set new association
        self.map_points[keypoint_idx] = map_point
        
        # Add observation to map point (bidirectional link)
        descriptor = None
        if keypoint_idx < len(self.descriptors):
            descriptor = self.descriptors[keypoint_idx]
        map_point.add_observation(self.id, keypoint_idx, descriptor)
    
    def remove_map_point(self, keypoint_idx: int) -> None:
        """Remove map point association (cleans bidirectional link).
        
        Args:
            keypoint_idx: Index of the keypoint
        """
        if keypoint_idx < 0 or keypoint_idx >= len(self.map_points):
            return
        
        mp = self.map_points[keypoint_idx]
        if mp is not None:
            mp.remove_observation(self.id)
            self.map_points[keypoint_idx] = None
    
    def get_map_points(self) -> List[MapPoint]:
        """Get all valid (non-None, non-bad) map points.
        
        Returns:
            List of valid MapPoints
        """
        return [mp for mp in self.map_points if mp is not None and not mp.is_bad]
    
    def get_descriptor_map_point_pairs(self) -> List[Tuple[np.ndarray, MapPoint]]:
        """Return list of (descriptor, map_point) pairs for valid associations."""
        pairs: List[Tuple[np.ndarray, MapPoint]] = []
        for idx, mp in enumerate(self.map_points):
            if mp is None or mp.is_bad:
                continue
            if idx >= len(self.descriptors):
                continue
            pairs.append((self.descriptors[idx].copy(), mp))
        return pairs
    
    def get_n_matches(self) -> int:
        """Count how many keypoints have associated map points.
        
        Returns:
            Number of matched keypoints
        """
        return sum(1 for mp in self.map_points if mp is not None and not mp.is_bad)
    
    # === Covisibility Graph ===
    
    def update_connections(self, keyframes: Dict[int, 'KeyFrame']) -> None:
        """Recompute covisibility based on shared map points.
        
        Args:
            keyframes: Dictionary of all keyframes indexed by ID
        """
        self.connected_keyframes.clear()
        
        # Count shared map points with each keyframe
        for mp in self.get_map_points():
            for kf_id in mp.observations:
                if kf_id == self.id:
                    continue
                if kf_id not in keyframes:
                    continue
                
                if kf_id not in self.connected_keyframes:
                    self.connected_keyframes[kf_id] = 0
                self.connected_keyframes[kf_id] += 1
    
    def get_best_covisible_keyframes(self, n: int) -> List[int]:
        """Get N most covisible keyframes (sorted by shared points).
        
        Args:
            n: Number of keyframes to return
            
        Returns:
            List of keyframe IDs sorted by covisibility (descending)
        """
        # Sort by weight (number of shared points) descending
        sorted_kfs = sorted(
            self.connected_keyframes.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [kf_id for kf_id, _ in sorted_kfs[:n]]
    
    def get_covisible_by_weight(self, min_weight: int) -> List[int]:
        """Get all keyframes sharing >= min_weight points.
        
        Args:
            min_weight: Minimum number of shared points
            
        Returns:
            List of keyframe IDs with sufficient covisibility
        """
        return [
            kf_id for kf_id, weight in self.connected_keyframes.items()
            if weight >= min_weight
        ]
    
    # === Spanning Tree ===
    
    def set_parent(self, parent_kf: 'KeyFrame') -> None:
        """Set parent in spanning tree (updates both directions).
        
        Args:
            parent_kf: Parent keyframe
        """
        # Remove from old parent's children
        if self.parent is not None:
            if self in self.parent.children:
                self.parent.children.remove(self)
        
        # Set new parent
        self.parent = parent_kf
        
        # Add to new parent's children
        if parent_kf is not None and self not in parent_kf.children:
            parent_kf.children.append(self)
    
    def add_child(self, child_kf: 'KeyFrame') -> None:
        """Add child to spanning tree.
        
        Args:
            child_kf: Child keyframe to add
        """
        if child_kf not in self.children:
            self.children.append(child_kf)
        child_kf.parent = self
    
    def remove_child(self, child_kf: 'KeyFrame') -> None:
        """Remove child from spanning tree.
        
        Args:
            child_kf: Child keyframe to remove
        """
        if child_kf in self.children:
            self.children.remove(child_kf)
        if child_kf.parent == self:
            child_kf.parent = None
    
    # === Pose Utilities ===
    
    def get_camera_center(self) -> np.ndarray:
        """Get camera center in world coordinates (3D point).
        
        Returns:
            3D point representing camera center in world frame
        """
        # Camera center in camera frame is [0, 0, 0]
        # Transform to world: C_w = R * [0,0,0] + t = t
        translation = self.get_translation()
        return translation
    
    def get_pose_w_to_c(self) -> np.ndarray:
        """Get world-to-camera pose (inverse of pose_c_to_w).
        
        Returns:
            4x4 SE(3) transformation (world to camera)
        """
        pose_w_to_c = np.linalg.inv(self.pose_c_to_w)
        return pose_w_to_c
    
    def get_rotation(self) -> np.ndarray:
        """Get rotation matrix (3x3) from pose_c_to_w.
        
        Returns:
            3x3 rotation matrix
        """
        return self.pose_c_to_w[:3, :3].copy()
    
    def get_translation(self) -> np.ndarray:
        """Get translation vector (3D) from pose_c_to_w.
        
        Returns:
            3D translation vector
        """
        return self.pose_c_to_w[:3, 3].copy()
    
    # === Projection ===
    
    def project_point(self, point_3d: np.ndarray) -> np.ndarray:
        """Project 3D world point to 2D image coordinates.
        
        Args:
            point_3d: 3D point in world coordinates [x, y, z]
            
        Returns:
            2D point in image coordinates [u, v]
        """
        # Transform to camera frame
        point_3d_h = np.append(point_3d, 1.0)  # Homogeneous
        point_cam = self.get_pose_w_to_c() @ point_3d_h
        
        # Project using camera matrix
        point_cam_3d = point_cam[:3]
        if point_cam_3d[2] <= 0:
            # Behind camera
            return np.array([np.nan, np.nan])
        
        projected = self.camera_matrix @ point_cam_3d
        projected_2d = projected[:2] / projected[2]
        
        return projected_2d
    
    def is_in_frustum(self, point_3d: np.ndarray, margin: float = 0.0) -> bool:
        """Check if 3D point projects inside image bounds.
        
        Args:
            point_3d: 3D point in world coordinates [x, y, z]
            margin: Margin in pixels (default 0.0)
            
        Returns:
            True if point projects within image bounds
        """
        projected = self.project_point(point_3d)
        
        if np.any(np.isnan(projected)):
            return False
        
        h, w = self.image.shape[:2]
        
        return (projected[0] >= -margin and projected[0] < w + margin and
                projected[1] >= -margin and projected[1] < h + margin)
    
    # === Feature Matching ===
    
    def get_features_in_area(self, x: float, y: float, radius: float) -> List[int]:
        """Get keypoint indices within pixel radius of (x,y).
        
        Args:
            x: X coordinate in image
            y: Y coordinate in image
            radius: Search radius in pixels
            
        Returns:
            List of keypoint indices within radius
        """
        indices = []
        for i, kp in enumerate(self.keypoints):
            dx = kp.pt[0] - x
            dy = kp.pt[1] - y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist <= radius:
                indices.append(i)
        return indices
    
    def __repr__(self) -> str:
        """String representation of KeyFrame."""
        n_matches = self.get_n_matches()
        n_connections = len(self.connected_keyframes)
        parent_id = self.parent.id if self.parent else None
        return f"KeyFrame(id={self.id}, matches={n_matches}, connections={n_connections}, parent={parent_id})"

