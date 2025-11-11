"""Map - Central Database for SLAM."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from .keyframe import KeyFrame
from .map_point import MapPoint


class Map:
    """Central database for managing KeyFrames and MapPoints.
    
    Core Responsibilities:
    - Store and manage all KeyFrames
    - Store and manage all MapPoints
    - Provide queries for local mapping (get nearby keyframes/points)
    - Maintain covisibility graph integrity
    - Manage spanning tree structure
    - Handle keyframe and map point removal
    - Track reference keyframe for tracking
    """
    
    def __init__(self):
        """Initialize an empty Map."""
        # Core storage
        self.keyframes: Dict[int, KeyFrame] = {}  # All keyframes indexed by ID
        self.map_points: List[MapPoint] = []  # All map points
        
        # Tracking reference
        self.reference_keyframe: Optional[KeyFrame] = None  # Current reference for tracking
    
    # === KeyFrame Management ===
    
    def add_keyframe(self, keyframe: KeyFrame) -> None:
        """Add keyframe to map.
        
        Process:
        1. Add to keyframes dict
        2. Update covisibility of new keyframe
        3. Update covisibility of connected keyframes
        4. Set parent in spanning tree (best covisibility)
        5. Update map point statistics
        
        Args:
            keyframe: KeyFrame to add
        """
        # 1. Add to dict
        self.keyframes[keyframe.id] = keyframe
        
        # 2. Update covisibility of new keyframe
        keyframe.update_connections(self.keyframes)
        
        # 3. Update covisibility of connected keyframes
        for kf_id in keyframe.connected_keyframes:
            if kf_id in self.keyframes:
                self.keyframes[kf_id].update_connections(self.keyframes)
        
        # 4. Set parent in spanning tree (best covisibility)
        if len(keyframe.connected_keyframes) > 0:
            best_covis = keyframe.get_best_covisible_keyframes(1)
            if len(best_covis) > 0:
                parent_id = best_covis[0]
                if parent_id in self.keyframes:
                    keyframe.set_parent(self.keyframes[parent_id])
        # If no connections, this is the first keyframe (root of tree)
        
        # 5. Update map point statistics
        for mp in keyframe.get_map_points():
            mp.update_normal_and_depth(self.keyframes)
    
    def remove_keyframe(self, keyframe: KeyFrame) -> None:
        """Remove keyframe from map.
        
        Process:
        1. Mark as bad
        2. Reconnect children to parent in spanning tree
        3. Update covisibility of connected keyframes
        4. Remove from dict
        5. Update reference if needed
        
        Args:
            keyframe: KeyFrame to remove
        """
        if keyframe.id not in self.keyframes:
            return
        
        # 1. Mark as bad
        keyframe.is_bad = True
        
        # 2. Reconnect children to parent in spanning tree
        # When removing root, make first child the new root
        parent = keyframe.parent
        children = keyframe.children
        if parent is not None:
            for child in children:
                child.set_parent(parent)
        else:
            # Removing root: make first child new root
            if len(children) > 0:
                new_root = children[0]
                new_root.parent = None
                for child in children[1:]:
                    child.set_parent(new_root)
        
        # 3. Update covisibility of connected keyframes
        connected_ids = list(keyframe.connected_keyframes.keys())
        for kf_id in connected_ids:
            if kf_id in self.keyframes:
                self.keyframes[kf_id].update_connections(self.keyframes)
        
        # 4. Remove map point observations
        for mp in keyframe.get_map_points():
            mp.remove_observation(keyframe.id)
        
        # 5. Remove from dict
        del self.keyframes[keyframe.id]
        
        # 6. Update reference if needed
        if self.reference_keyframe is not None and self.reference_keyframe.id == keyframe.id:
            # Set reference to most recent keyframe
            if len(self.keyframes) > 0:
                recent_kfs = self.get_recent_keyframes(1)
                if len(recent_kfs) > 0:
                    self.reference_keyframe = recent_kfs[0]
                else:
                    self.reference_keyframe = None
            else:
                self.reference_keyframe = None
    
    def get_keyframe(self, keyframe_id: int) -> Optional[KeyFrame]:
        """Get keyframe by ID.
        
        Args:
            keyframe_id: ID of the keyframe
            
        Returns:
            KeyFrame if found, None otherwise
        """
        return self.keyframes.get(keyframe_id)
    
    def get_all_keyframes(self) -> List[KeyFrame]:
        """Get all valid keyframes.
        
        Returns:
            List of all non-bad keyframes
        """
        return [kf for kf in self.keyframes.values() if not kf.is_bad]
    
    def get_recent_keyframes(self, n: int) -> List[KeyFrame]:
        """Get N most recent keyframes (by ID).
        
        Args:
            n: Number of keyframes to return
            
        Returns:
            List of most recent keyframes (sorted by ID descending)
        """
        valid_kfs = self.get_all_keyframes()
        sorted_kfs = sorted(valid_kfs, key=lambda kf: kf.id, reverse=True)
        return sorted_kfs[:n]
    
    # === MapPoint Management ===
    
    def add_map_point(self, map_point: MapPoint) -> None:
        """Add map point to map.
        
        Process:
        1. Add to map_points list
        2. Update viewing statistics
        
        Args:
            map_point: MapPoint to add
        """
        self.map_points.append(map_point)
        map_point.update_normal_and_depth(self.keyframes)
    
    def remove_map_point(self, map_point: MapPoint) -> None:
        """Remove map point.
        
        Process:
        1. Mark as bad
        2. Remove observations from all keyframes
        
        Args:
            map_point: MapPoint to remove
        """
        map_point.is_bad = True
        
        # Remove observations from all keyframes
        for kf_id in list(map_point.observations.keys()):
            kp_idx = map_point.observations[kf_id]
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                if kp_idx < len(kf.map_points):
                    if kf.map_points[kp_idx] is map_point:
                        kf.map_points[kp_idx] = None
        
        map_point.observations.clear()
    
    def get_all_map_points(self) -> List[MapPoint]:
        """Get all valid (non-bad) map points.
        
        Returns:
            List of all non-bad map points
        """
        return [mp for mp in self.map_points if not mp.is_bad]
    
    def cull_map_points(self) -> None:
        """Remove bad map points based on criteria.
        
        Criteria:
        - Too few observations (< 3 keyframes)
        - Bad found ratio (< 25%)
        - Should be visible but not found
        """
        to_remove = []
        
        for mp in self.map_points:
            if mp.is_bad:
                continue
            
            # Too few observations
            if mp.get_observation_count() < 3:
                to_remove.append(mp)
                continue
            
            # Bad found ratio
            found_ratio = mp.get_found_ratio()
            if found_ratio < 0.25:
                to_remove.append(mp)
                continue
            
            # Should be visible but not found
            if mp.n_visible > 0 and mp.n_found == 0:
                to_remove.append(mp)
                continue
        
        # Remove bad points
        for mp in to_remove:
            self.remove_map_point(mp)
    
    # === Local Mapping Queries ===
    
    def get_local_map(self, reference_kf: KeyFrame, n_neighbors: int) -> Tuple[
        List[KeyFrame], List[MapPoint], List[KeyFrame]
    ]:
        """Get local map for tracking/local BA.
        
        Returns:
            Tuple of:
            - K1: Local keyframes (reference + N best covisible)
            - Local map points: All points seen by K1
            - K2: Fixed keyframes (neighbors of K1 not in K1)
        
        Args:
            reference_kf: Reference keyframe
            n_neighbors: Number of best covisible neighbors to include
            
        Returns:
            Tuple (K1 keyframes, local map points, K2 keyframes)
        """
        # K1: Local keyframes (reference + N best covisible)
        K1 = [reference_kf]
        
        # Get best covisible keyframes
        best_covis = reference_kf.get_best_covisible_keyframes(n_neighbors)
        for kf_id in best_covis:
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                if not kf.is_bad:
                    K1.append(kf)
        
        # Local map points: All points seen by K1
        local_map_points = []
        seen_point_ids = set()
        
        for kf in K1:
            for mp in kf.get_map_points():
                if mp.id not in seen_point_ids:
                    local_map_points.append(mp)
                    seen_point_ids.add(mp.id)
        
        # K2: Fixed keyframes (neighbors of K1 not in K1)
        K1_ids = {kf.id for kf in K1}
        K2 = []
        seen_k2_ids = set()
        
        for kf in K1:
            for kf_id in kf.connected_keyframes:
                if kf_id not in K1_ids and kf_id not in seen_k2_ids:
                    if kf_id in self.keyframes:
                        neighbor_kf = self.keyframes[kf_id]
                        if not neighbor_kf.is_bad:
                            K2.append(neighbor_kf)
                            seen_k2_ids.add(kf_id)
        
        return K1, local_map_points, K2
    
    def get_local_keyframes(self, current_kf: KeyFrame, n: int) -> List[KeyFrame]:
        """Get local keyframes for local mapping.
        
        Args:
            current_kf: Current keyframe
            n: Number of best covisible keyframes to include
            
        Returns:
            List of local keyframes (current + best covisible)
        """
        local_kfs = [current_kf]
        best_covis = current_kf.get_best_covisible_keyframes(n)
        
        for kf_id in best_covis:
            if kf_id in self.keyframes:
                kf = self.keyframes[kf_id]
                if not kf.is_bad:
                    local_kfs.append(kf)
        
        return local_kfs
    
    # === Reference Tracking ===
    
    def set_reference_keyframe(self, kf: KeyFrame) -> None:
        """Set current reference keyframe for tracking.
        
        Args:
            kf: KeyFrame to set as reference
        """
        if kf.id in self.keyframes:
            self.reference_keyframe = kf
    
    def get_reference_keyframe(self) -> Optional[KeyFrame]:
        """Get current reference keyframe.
        
        Returns:
            Reference keyframe if set, None otherwise
        """
        if self.reference_keyframe is not None and self.reference_keyframe.is_bad:
            self.reference_keyframe = None
        return self.reference_keyframe
    
    # === Utilities ===
    
    def clear(self) -> None:
        """Clear all data and reset ID counters."""
        self.keyframes.clear()
        self.map_points.clear()
        self.reference_keyframe = None
        
        # Reset ID counters
        KeyFrame._next_id = 0
        MapPoint._next_id = 0
    
    def get_map_size(self) -> Tuple[int, int]:
        """Return (num_keyframes, num_map_points).
        
        Returns:
            Tuple of (number of keyframes, number of map points)
        """
        num_keyframes = len(self.get_all_keyframes())
        num_map_points = len(self.get_all_map_points())
        return (num_keyframes, num_map_points)

    def __repr__(self) -> str:
        return f"Map(kfs={len(self.keyframes)}, points={len(self.get_all_map_points())})"