"""MapPoint - 3D Point in World."""
from typing import Dict, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .keyframe import KeyFrame


class MapPoint:
    """3D point in world coordinates with observation tracking.
    
    Core Responsibilities:
    - Store 3D position in world coordinates
    - Track all keyframes that observe it (bidirectional relationship)
    - Maintain multiple descriptors and select the best representative
    - Compute viewing statistics (direction, depth range)
    - Track quality metrics for outlier detection
    """
    
    _next_id = 0
    
    def __init__(self, 
    position: np.ndarray, 
    descriptor: Optional[np.ndarray] = None,
    ):
        """Initialize a MapPoint.
        
        Args:
            position: 3D [x, y, z] in world frame
            descriptor: Optional initial descriptor
        """
        self.id = MapPoint._next_id
        MapPoint._next_id += 1
        
        self.position: np.ndarray = position.copy()  # 3D [x, y, z] in world frame
        
        # Observations (bidirectional with KeyFrame)
        self.observations: Dict[int, int] = {}  # {keyframe_id: keypoint_index}
        self.descriptors: List[np.ndarray] = []  # All descriptors from observations
        
        if descriptor is not None:
            self.descriptors.append(descriptor.copy())
        
        self.descriptor: Optional[np.ndarray] = None  # Best representative descriptor
        
        # Quality tracking
        self.n_visible: int = 0  # How many times should be visible
        self.n_found: int = 0  # How many times actually matched
        self.is_bad: bool = False  # Outlier flag
        
        # Viewing statistics
        self.normal: Optional[np.ndarray] = None  # Mean viewing direction (unit vector)
        self.min_distance: float = 0.0  # Min distance from cameras
        self.max_distance: float = 0.0  # Max distance from cameras
    
    # === Observation Management ===
    
    def add_observation(self, keyframe_id: int, keypoint_idx: int, 
                       descriptor: Optional[np.ndarray] = None) -> None:
        """Add observation from a keyframe, optionally with descriptor.
        
        Args:
            keyframe_id: ID of the observing keyframe
            keypoint_idx: Index of the keypoint in that keyframe
            descriptor: Optional descriptor for this observation
        """
        already_observing = keyframe_id in self.observations
        self.observations[keyframe_id] = keypoint_idx
        
        if descriptor is not None and not already_observing:
            self.descriptors.append(descriptor.copy())
        
        # Recompute distinctive descriptor if we have descriptors
        if len(self.descriptors) > 0:
            self.compute_distinctive_descriptor()
    
    def remove_observation(self, keyframe_id: int) -> None:
        """Remove observation when keyframe is removed or point is culled.
        
        Args:
            keyframe_id: ID of the keyframe to remove observation from
        """
        if keyframe_id in self.observations:
            del self.observations[keyframe_id]
            # Note: We don't remove descriptors as we don't track which descriptor
            # came from which observation. This is acceptable as descriptors are
            # used for matching and having extra ones doesn't hurt.
    
    def get_observation_count(self) -> int:
        """Number of keyframes observing this point.
        
        Returns:
            Number of observations
        """
        return len(self.observations)
    
    # === Descriptor Management ===
    
    def compute_distinctive_descriptor(self) -> None:
        """Select best descriptor using median Hamming distance criterion.
        
        For each descriptor, compute median Hamming distance to all others.
        The descriptor with the smallest median distance is most representative.
        """
        if len(self.descriptors) == 0:
            self.descriptor = None
            return
        
        if len(self.descriptors) == 1:
            self.descriptor = self.descriptors[0].copy()
            return
        
        # Convert descriptors to uint8 if needed
        descs = [desc.astype(np.uint8) for desc in self.descriptors]
        
        # Compute pairwise Hamming distances
        n = len(descs)
        median_distances = []
        
        for i in range(n):
            distances = []
            for j in range(n):
                if i != j:
                    # Hamming distance: XOR and count bits
                    dist = np.sum(descs[i] != descs[j])
                    distances.append(dist)
            
            # Median distance from this descriptor to all others
            median_distances.append(np.median(distances))
        
        # Select descriptor with minimum median distance
        best_idx = np.argmin(median_distances)
        self.descriptor = self.descriptors[best_idx].copy()
    
    # === Viewing Statistics ===
    
    def update_normal_and_depth(self, keyframes: Dict[int, 'KeyFrame']) -> None:
        """Compute mean viewing direction and depth range from observations.
        
        Args:
            keyframes: Dictionary of all keyframes indexed by ID
        """
        if len(self.observations) == 0:
            self.normal = None
            self.min_distance = 0.0
            self.max_distance = 0.0
            return
        
        # Compute viewing directions from each keyframe
        viewing_directions = []
        distances = []
        
        for kf_id in self.observations:
            if kf_id not in keyframes:
                continue
            
            kf = keyframes[kf_id]
            camera_center = kf.get_camera_center()
            
            # Viewing direction: from camera to point (normalized)
            direction = self.position - camera_center
            distance = np.linalg.norm(direction)
            
            if distance > 1e-6:  # Avoid division by zero
                direction = direction / distance
                viewing_directions.append(direction)
                distances.append(distance)
        
        if len(viewing_directions) == 0:
            self.normal = None
            self.min_distance = 0.0
            self.max_distance = 0.0
            return
        
        # Mean viewing direction (normalized)
        mean_direction = np.mean(viewing_directions, axis=0)
        norm = np.linalg.norm(mean_direction)
        if norm > 1e-6:
            self.normal = mean_direction / norm
        else:
            self.normal = None
        
        # Depth range
        distances_array = np.array(distances)
        self.min_distance = float(np.min(distances_array))
        self.max_distance = float(np.max(distances_array))
    
    # === Quality Tracking ===
    
    def get_found_ratio(self) -> float:
        """Return n_found / n_visible for quality assessment.
        
        Returns:
            Found ratio (0.0 to 1.0), or 0.0 if n_visible is 0
        """
        if self.n_visible == 0:
            return 0.0
        return float(self.n_found) / float(self.n_visible)
    
    def increase_visible(self, n: int = 1) -> None:
        """Increment visibility counter.
        
        Args:
            n: Amount to increment by (default 1)
        """
        self.n_visible += n
    
    def increase_found(self, n: int = 1) -> None:
        """Increment found counter.
        
        Args:
            n: Amount to increment by (default 1)
        """
        self.n_found += n
    
    # === Additional Utility Methods ===
    
    def is_in_keyframe(self, kf_id: int) -> bool:
        """Check if observed by keyframe.
        
        Args:
            kf_id: Keyframe ID to check
            
        Returns:
            True if this point is observed by the keyframe
        """
        return kf_id in self.observations
    
    def get_reference_descriptor(self) -> Optional[np.ndarray]:
        """Get reference descriptor (alias for descriptor).
        
        Returns:
            Best representative descriptor, or None if not computed
        """
        return self.descriptor
    
    def set_bad(self) -> None:
        """Mark as bad and cleanup."""
        self.is_bad = True
        self.observations.clear()
        # Keep descriptors and position for potential recovery
    
    def __repr__(self) -> str:
        """String representation of MapPoint."""
        n_obs = len(self.observations)
        n_descs = len(self.descriptors)
        pos_str = f"[{self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f}]"
        return f"MapPoint(id={self.id}, pos={pos_str}, obs={n_obs}, descs={n_descs}, bad={self.is_bad})"


