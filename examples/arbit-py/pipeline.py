from collections import namedtuple
from typing import List, Optional
import cv2
import numpy as np
from eth3_utils import load_case
from pathlib import Path
from enum import Enum, auto
from visualizer import visualize_frontend

from slam import CameraMatrix, Map, KeyFrame, MapPoint, to_matrix
from video_utils import IphoneCameraInstrinsics

class PipelineState(Enum):
    PRE_INIT = auto()
    INIT = auto()
    INITIALISED = auto()


class Front_End:
  def __init__(self, camera_matrix: CameraMatrix):
    self.orb = cv2.ORB.create(2000)
    self.state: PipelineState = PipelineState.PRE_INIT
    
    self.camera_matrix = to_matrix(camera_matrix)
    self.map = Map()
    self.tracking_pose: np.ndarray | None = None
    self.motion_velocity: np.ndarray | None = None
    self.frames_since_last_keyframe: int = 0
    self.last_keyframe_pose: np.ndarray | None = None
    self.min_frames_between_keyframes = 12
    self.max_frames_between_keyframes = 60
    self.translation_threshold = 0.1
    self.rotation_threshold = np.deg2rad(10.0)
    self.guided_search_radius = 8.0
    self.min_tracked_points_for_keyframe = 40
    
    self.prev_frame: np.ndarray | None = None
    self.prev_color_frame: np.ndarray | None = None
    self.prev_keypoints: list | None = None
    self.prev_descriptors: np.ndarray | None = None
    
    # Tracking info
    self.init_method: str = ""  # Track which method was used (H or F)
    self.current_keyframe: KeyFrame | None = None

  def ingest_image(self, image: np.ndarray, color_image: np.ndarray | None = None):
    if color_image is None:
      color_image = image
    match self.state:
      case PipelineState.PRE_INIT:
        self.pre_init(image, color_image)
      case PipelineState.INIT:
        self.init(image, color_image)
      case PipelineState.INITIALISED:
        self.initialised(image, color_image)

  def pre_init(self, image: np.ndarray, color_image: np.ndarray):
    """Store first frame and extract features."""
    self.prev_frame = image
    self.prev_color_frame = color_image
    self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(image, None)  # type: ignore
    n_keypoints = len(self.prev_keypoints) if self.prev_keypoints is not None else 0
    print(f"PRE_INIT: Detected {n_keypoints} keypoints in first frame")
    self.state = PipelineState.INIT

  def compute_symmetric_transfer_error(self, points0: np.ndarray, points1: np.ndarray, 
                                        model: np.ndarray, is_homography: bool) -> tuple[float, np.ndarray]:
    """
    Compute symmetric transfer error for model scoring.
    Returns score (higher is better) and inlier mask.

    Args:
      points0: 2D points in image 0
      points1: 2D points in image 1
      model: 3x3 homography or 3x4 fundamental matrix
      is_homography: True if model is a homography, False if model is a fundamental matrix

    Returns:
      score: float score of the model
      inliers: boolean mask of inliers
    """
    threshold = 5.991  # Chi-square 2 DOF 95% (ORB-SLAM uses this)
    n_points = len(points0)
    
    if is_homography:
      # Homography: compute reprojection errors both ways
      points0_h = np.hstack([points0, np.ones((n_points, 1))])
      points1_h = np.hstack([points1, np.ones((n_points, 1))])
      
      # Forward: H * p0 -> p1
      projected1 = (model @ points0_h.T).T
      projected1 = projected1[:, :2] / projected1[:, 2:3]
      error_forward = np.sum((points1 - projected1) ** 2, axis=1)
      
      # Backward: H_inv * p1 -> p0
      H_inv = np.linalg.inv(model)
      projected0 = (H_inv @ points1_h.T).T
      projected0 = projected0[:, :2] / projected0[:, 2:3]
      error_backward = np.sum((points0 - projected0) ** 2, axis=1)
      
      # Symmetric error
      errors = error_forward + error_backward
    else:
      # Fundamental matrix: compute symmetric epipolar error
      points0_h = np.hstack([points0, np.ones((n_points, 1))])
      points1_h = np.hstack([points1, np.ones((n_points, 1))])
      
      # Symmetric epipolar distance
      # d(p1, F*p0) + d(p0, F^T*p1)
      epilines1 = (model @ points0_h.T).T  # Lines in image 1
      epilines0 = (model.T @ points1_h.T).T  # Lines in image 0
      
      error_forward = np.abs(np.sum(points1_h * epilines1, axis=1)) / np.sqrt(epilines1[:, 0]**2 + epilines1[:, 1]**2)
      error_backward = np.abs(np.sum(points0_h * epilines0, axis=1)) / np.sqrt(epilines0[:, 0]**2 + epilines0[:, 1]**2)
      
      errors = error_forward + error_backward
    
    # Score: count inliers and penalize outliers
    inliers = errors < threshold
    score = np.sum(inliers)
    
    return float(score), inliers

  def _predict_pose(self) -> np.ndarray | None:
    if self.tracking_pose is None:
      return None
    if self.motion_velocity is None:
      return self.tracking_pose.copy()
    return (self.motion_velocity @ self.tracking_pose).copy()

  def _update_motion_model(self, new_pose: np.ndarray) -> None:
    if self.tracking_pose is not None:
      prev_pose_inv = np.linalg.inv(self.tracking_pose)
      self.motion_velocity = new_pose @ prev_pose_inv
    else:
      self.motion_velocity = None
    self.tracking_pose = new_pose.copy()

  def _project_point_from_pose(self, pose_w_to_c: np.ndarray, point_3d: np.ndarray) -> np.ndarray | None:
    point_h = np.append(point_3d, 1.0)
    point_cam = pose_w_to_c @ point_h
    if point_cam[2] <= 0:
      return None
    projected = self.camera_matrix @ point_cam[:3]
    if np.abs(projected[2]) < 1e-6:
      return None
    return projected[:2] / projected[2]

  def _within_distance_range(self, map_point: MapPoint, camera_center: np.ndarray) -> bool:
    if map_point.min_distance == 0.0 and map_point.max_distance == 0.0:
      return True
    distance = np.linalg.norm(map_point.position - camera_center)
    min_dist = map_point.min_distance * 0.8 if map_point.min_distance > 0 else 0.0
    max_dist = map_point.max_distance * 1.2 if map_point.max_distance > 0 else float('inf')
    lower_ok = bool(distance >= min_dist)
    upper_ok = bool(distance <= max_dist)
    return lower_ok and upper_ok

  def _collect_matchable_map_points(
    self,
    image_shape: tuple[int, int],
    predicted_pose: np.ndarray | None
  ) -> tuple[list[MapPoint], np.ndarray, list[np.ndarray | None]]:
    height, width = image_shape
    map_points = []
    descriptors: list[np.ndarray] = []
    projections: list[np.ndarray | None] = []
    if predicted_pose is not None:
      pose_w_to_c = np.linalg.inv(predicted_pose)
      camera_center = predicted_pose[:3, 3]
    else:
      pose_w_to_c = None
      camera_center = None
    for mp in self.map.get_all_map_points():
      descriptor = mp.get_reference_descriptor()
      if descriptor is None:
        continue
      proj = None
      if pose_w_to_c is not None:
        proj = self._project_point_from_pose(pose_w_to_c, mp.position)
        if proj is None:
          continue
        if proj[0] < 0 or proj[0] >= width or proj[1] < 0 or proj[1] >= height:
          continue
        if camera_center is not None and not self._within_distance_range(mp, camera_center):
          continue
      map_points.append(mp)
      descriptors.append(descriptor)
      projections.append(proj)
    descriptor_array = np.asarray(descriptors, dtype=np.uint8) if descriptors else np.empty((0, 32), dtype=np.uint8)
    return map_points, descriptor_array, projections

  def _sample_keypoint_color(self, keypoint: cv2.KeyPoint, color_image: Optional[np.ndarray], fallback_image: np.ndarray) -> np.ndarray:
    """Sample RGB color from the image at the keypoint location."""
    source = color_image if color_image is not None else fallback_image
    if source is None or keypoint is None:
      return np.array([0, 255, 0], dtype=np.uint8)
    h, w = source.shape[:2]
    x = int(np.clip(round(keypoint.pt[0]), 0, w - 1))
    y = int(np.clip(round(keypoint.pt[1]), 0, h - 1))
    pixel = source[y, x]
    if source.ndim == 2:
      value = int(pixel)
      return np.array([value, value, value], dtype=np.uint8)
    if pixel.ndim == 0:
      value = int(pixel)
      return np.array([value, value, value], dtype=np.uint8)
    # Assume BGR from OpenCV, convert to RGB
    bgr = pixel.astype(np.uint8)
    return np.array([bgr[2], bgr[1], bgr[0]], dtype=np.uint8)

  def _build_keyframe_metrics(self, n_tracked: int, current_pose: np.ndarray) -> dict:
    translation = 0.0
    rotation = 0.0
    if self.last_keyframe_pose is not None:
      delta = current_pose @ np.linalg.inv(self.last_keyframe_pose)
      translation = float(np.linalg.norm(delta[:3, 3]))
      trace_value = np.clip((np.trace(delta[:3, :3]) - 1.0) * 0.5, -1.0, 1.0)
      rotation = float(np.arccos(trace_value))
    return {
      "frames_since_last": self.frames_since_last_keyframe,
      "min_interval": self.min_frames_between_keyframes,
      "max_interval": self.max_frames_between_keyframes,
      "n_matches": n_tracked,
      "min_tracked": self.min_tracked_points_for_keyframe,
      "translation": translation,
      "rotation": rotation,
      "translation_thresh": self.translation_threshold,
      "rotation_thresh": self.rotation_threshold
    }

  def _should_insert_keyframe(self, metrics: dict) -> bool:
    if self.last_keyframe_pose is None:
      return True
    frames_since_last = metrics.get("frames_since_last", 0)
    min_interval = metrics.get("min_interval", 0)
    max_interval = metrics.get("max_interval", float('inf'))
    if frames_since_last < min_interval:
      return False
    if frames_since_last >= max_interval:
      return True
    translation = metrics.get("translation", 0.0)
    rotation = metrics.get("rotation", 0.0)
    translation_thresh = metrics.get("translation_thresh", 0.0)
    rotation_thresh = metrics.get("rotation_thresh", 0.0)
    if translation > translation_thresh or rotation > rotation_thresh:
      n_matches = metrics.get("n_matches", 0)
      min_tracked = metrics.get("min_tracked", 0)
      return n_matches <= min_tracked
    return False

  def triangulate_points(
    self,
    points0: np.ndarray,
    points1: np.ndarray,
    P0: np.ndarray,
    P1: np.ndarray,
  ) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from two projection matrices.
    Returns 3D points in world frame and mask of valid points.
    """
    if len(points0) == 0 or len(points1) == 0:
      return np.empty((0, 3)), np.empty((0,), dtype=bool)

    points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T

    K_inv = np.linalg.inv(self.camera_matrix)
    Rt0 = K_inv @ P0
    Rt1 = K_inv @ P1
    R0, t0 = Rt0[:, :3], Rt0[:, 3:]
    R1, t1 = Rt1[:, :3], Rt1[:, 3:]

    points_cam0 = (R0 @ points_3d.T + t0).T
    points_cam1 = (R1 @ points_3d.T + t1).T

    depth0 = points_cam0[:, 2]
    depth1 = points_cam1[:, 2]
    max_depth = 100.0
    valid_depth = (depth0 > 0) & (depth1 > 0) & (depth0 < max_depth) & (depth1 < max_depth)

    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    proj0 = (P0 @ points_3d_h.T).T
    proj0 = proj0[:, :2] / proj0[:, 2:3]
    proj1 = (P1 @ points_3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]

    reproj_error0 = np.sum((points0 - proj0) ** 2, axis=1)
    reproj_error1 = np.sum((points1 - proj1) ** 2, axis=1)
    reproj_threshold = 5.991
    valid_reproj = (reproj_error0 < reproj_threshold) & (reproj_error1 < reproj_threshold)

    valid_mask = valid_depth & valid_reproj & np.isfinite(points_3d).all(axis=1)
    return points_3d, valid_mask

  def init(self, image: np.ndarray, color_image: np.ndarray):
    assert self.prev_frame is not None, "prev_frame must be set before init"
    assert self.prev_keypoints is not None, "prev_keypoints must be set before init"
    assert self.prev_descriptors is not None, "prev_descriptors must be set before init"
    
    # Use pre-extracted features for frame 0
    keypoints0 = self.prev_keypoints
    descriptors0 = self.prev_descriptors
    
    # Extract features for frame 1
    keypoints1, descriptors1 = self.orb.detectAndCompute(image, None)  # type: ignore
    print(f"INIT: Frame 0: {len(keypoints0)} keypoints, Frame 1: {len(keypoints1)} keypoints")
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors0, descriptors1)
    matches = sorted(matches, key=lambda x: x.distance)
    points0 = np.array([keypoints0[m.queryIdx].pt for m in matches])
    points1 = np.array([keypoints1[m.trainIdx].pt for m in matches])
    print(f"Found {len(matches)} matches")
    
    # ========================================
    # H/F Model Selection (ORB-SLAM approach)
    # ========================================
    
    # 1. Compute Homography
    H, mask_H = cv2.findHomography(points0, points1, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None:
      print("ERROR: findHomography returned None")
      return
    
    # 2. Compute Fundamental Matrix
    F, mask_F = cv2.findFundamentalMat(points0, points1, cv2.RANSAC, ransacReprojThreshold=3.0)
    if F is None:
      print("ERROR: findFundamentalMat returned None")
      return
    
    # 3. Score both models using symmetric transfer error
    mask_H_bool = mask_H.ravel().astype(bool)
    mask_F_bool = mask_F.ravel().astype(bool)
    
    score_H, inliers_H = self.compute_symmetric_transfer_error(
      points0[mask_H_bool], points1[mask_H_bool], H, is_homography=True
    )
    score_F, inliers_F = self.compute_symmetric_transfer_error(
      points0[mask_F_bool], points1[mask_F_bool], F, is_homography=False
    )
    
    print(f"Homography score: {score_H} ({np.sum(mask_H_bool)} RANSAC inliers)")
    print(f"Fundamental score: {score_F} ({np.sum(mask_F_bool)} RANSAC inliers)")
    
    # 4. Model selection: R_H = S_H / (S_H + S_F)
    # If R_H > 0.45, scene is planar (use H), otherwise non-planar (use F)
    R_H = score_H / (score_H + score_F) if (score_H + score_F) > 0 else 0
    print(f"R_H ratio: {R_H:.3f} (threshold: 0.45)")
    
    # ========================================
    # Pose Recovery
    # ========================================
    
    if R_H > 0.45:
      # Planar scene: decompose homography
      self.init_method = "Homography"
      print("Selected: HOMOGRAPHY (planar scene)")
      
      # Get inliers
      H_inlier_indices = np.where(mask_H_bool)[0][inliers_H]
      points0_inliers = points0[H_inlier_indices]
      points1_inliers = points1[H_inlier_indices]
      
      # Decompose homography to get multiple solutions
      num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.camera_matrix)
      
      # Test all solutions and pick the one with most points in front
      best_solution = None
      best_num_infront = 0
      
      for i in range(num_solutions):
        R_test = Rs[i]
        t_test = ts[i].reshape(3, 1)
        
        # Triangulate and count points with positive depth

        # Projection matrices: P0 = K[I|0], P1 = K[R|t]
        P0 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P1 = self.camera_matrix @ np.hstack([R_test, t_test])
        points_3d, valid_mask = self.triangulate_points(points0_inliers, points1_inliers, P0, P1)
        num_infront = np.sum(valid_mask)
        
        if num_infront > best_num_infront:
          best_num_infront = num_infront
          best_solution = (R_test, t_test, points_3d, valid_mask)
      
      if best_solution is None:
        print("ERROR: No valid homography decomposition found")
        return
      
      R, t, points_3d, valid_mask = best_solution
      print(f"Best H solution: {best_num_infront} points in front")
      
    else:
      # Non-planar scene: use Fundamental/Essential matrix
      self.init_method = "Fundamental"
      print("Selected: FUNDAMENTAL (non-planar scene)")
      
      # Get inliers
      F_inlier_indices = np.where(mask_F_bool)[0][inliers_F]
      points0_inliers = points0[F_inlier_indices]
      points1_inliers = points1[F_inlier_indices]
      
      # Convert F to Essential matrix: E = K^T * F * K
      E = self.camera_matrix.T @ F @ self.camera_matrix
      
      # Recover pose from essential matrix
      _, R, t, mask_pose = cv2.recoverPose(E, points0_inliers, points1_inliers, self.camera_matrix)
      
      # Filter by pose mask
      mask_pose_bool = mask_pose.ravel().astype(bool)
      points0_inliers = points0_inliers[mask_pose_bool]
      points1_inliers = points1_inliers[mask_pose_bool]
      print(f"recoverPose: {np.sum(mask_pose_bool)} points passed cheirality check")
      
      # Triangulate all inliers
      P0 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
      P1 = self.camera_matrix @ np.hstack([R, t])
      points_3d, valid_mask = self.triangulate_points(points0_inliers, points1_inliers, P0, P1)
    
    # ========================================
    # Create Map Structure (KeyFrames + MapPoints)
    # ========================================
    
    # Get original match indices to track which keypoints were used
    if self.init_method == "Homography":
      # For homography, we have H_inlier_indices which maps to original matches
      match_indices = H_inlier_indices
    else:
      # For fundamental, we have F_inlier_indices
      match_indices = F_inlier_indices
    
    # Build KeyFrame 0 (first frame, origin pose)
    pose_kf0 = np.eye(4)  # World origin
    kf0 = KeyFrame(
      image=self.prev_frame,
      camera_matrix=self.camera_matrix,
      pose_c_to_w=pose_kf0,
      keypoints=keypoints0,
      descriptors=descriptors0,
      color_image=self.prev_color_frame
    )
    
    # Build KeyFrame 1 (second frame, recovered pose)
    # R and t represent world-to-camera, need to invert for camera-to-world
    pose_kf1 = np.eye(4)
    pose_kf1[:3, :3] = R.T
    pose_kf1[:3, 3] = -R.T @ t.ravel()
    kf1 = KeyFrame(
      image=image,
      camera_matrix=self.camera_matrix,
      pose_c_to_w=pose_kf1,
      keypoints=keypoints1,
      descriptors=descriptors1,
      color_image=color_image
    )
    print(f"pose_kf1: {pose_kf1}")
    
    # Create MapPoints and link to KeyFrames
    n_valid = np.sum(valid_mask)
    map_point_ids = []

    self.map.add_keyframe(kf0)
    self.map.add_keyframe(kf1)
    self.current_keyframe = kf1
    self._update_motion_model(pose_kf1)
    self.last_keyframe_pose = pose_kf1.copy()
    self.frames_since_last_keyframe = 0

    for i, (is_valid, point_3d) in enumerate(zip(valid_mask, points_3d)):
      if not is_valid:
        continue
      match_idx = match_indices[i]
      match = matches[match_idx]
      kp0_idx, kp1_idx = match.queryIdx, match.trainIdx
      
      kp_color = self._sample_keypoint_color(kf1.keypoints[kp1_idx], kf1.get_color_image(), kf1.image)
      mp = MapPoint(position=point_3d, color=kp_color)
      kf0.add_map_point(mp, kp0_idx)
      kf1.add_map_point(mp, kp1_idx)
      self.map.add_map_point(mp)
      map_point_ids.append(mp.id)

    # Print statistics
    print(f"\n{'='*60}")
    print(f"Initialization successful!")
    print(f"Method: {self.init_method}")
    print(f"Triangulated {n_valid} / {len(points_3d)} 3D map points")
    print(f"Map: {self.map}")
    print(f"KeyFrame 0: {kf0}")
    print(f"KeyFrame 1: {kf1}")
    
    valid_points = points_3d[valid_mask]
    if len(valid_points) > 0:
      print(f"Average depth: {np.mean(valid_points[:, 2]):.2f}")
      print(f"Depth range: [{np.min(valid_points[:, 2]):.2f}, {np.max(valid_points[:, 2]):.2f}]")
    print(f"{'='*60}\n")
    
    self.state = PipelineState.INITIALISED

  def initialised(self, image: np.ndarray, color_image: np.ndarray):
    """Process frames after initialization (tracking state)."""
    # TODO: Implement tracking against local map
    # 1. Extract features from current frame
    # 2. Match against local map points
    # 3. Estimate pose via PnP
    # 4. Decide if new keyframe needed
    # 5. Local mapping / BA
    mask = np.ones((image.shape[0], image.shape[1]), dtype=bool)
    keypoints, descriptors = self.orb.detectAndCompute(image, None)  # type: ignore  # pyright: ignore[reportGeneralTypeIssues]
    if keypoints is None or descriptors is None:
      print("ERROR: detectAndCompute returned None")
      return
    self.frames_since_last_keyframe += 1

    predicted_pose = self._predict_pose()
    matchable_map_pts, map_descriptors, projections = self._collect_matchable_map_points(image.shape[:2], predicted_pose)
    if len(matchable_map_pts) == 0 and predicted_pose is not None:
      matchable_map_pts, map_descriptors, projections = self._collect_matchable_map_points(image.shape[:2], None)
    if len(matchable_map_pts) == 0 or map_descriptors.size == 0:
      print("Tracking skipped: no map points available for matching")
      self.motion_velocity = None
      return

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(map_descriptors, descriptors)
    matches = sorted(matches, key=lambda x: x.distance)

    filtered_matches: list[cv2.DMatch] = []
    filtered_map_pts: list[MapPoint] = []
    for match in matches:
      mp = matchable_map_pts[match.queryIdx]
      proj = projections[match.queryIdx]
      if proj is not None:
        kp = keypoints[match.trainIdx]
        if np.hypot(kp.pt[0] - proj[0], kp.pt[1] - proj[1]) > self.guided_search_radius:
          continue
      filtered_matches.append(match)
      filtered_map_pts.append(mp)
      if len(filtered_matches) >= 100:
        break

    if len(filtered_matches) < 6:
      print("Tracking skipped: insufficient guided matches")
      self.motion_velocity = None
      return

    matches = filtered_matches
    matches_map_pts = filtered_map_pts
    world_points = np.array([mp.position for mp in matches_map_pts], dtype=np.float32)
    image_points = np.array([keypoints[m.trainIdx].pt for m in matches], dtype=np.float32)
    
    success, rvec, tvec = cv2.solvePnP(world_points, image_points, self.camera_matrix, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
      self.motion_velocity = None
      return

    rvec, tvec = cv2.solvePnPRefineLM(world_points, image_points, self.camera_matrix, np.zeros((4, 1)), rvec, tvec)

    R_mat, _ = cv2.Rodrigues(rvec)
    T_w_to_c = np.hstack((R_mat, tvec))
    R_c_to_w = R_mat.T
    t_c_to_w = -R_c_to_w @ tvec.ravel()
    T_c_to_w = np.eye(4)
    T_c_to_w[:3, :3] = R_c_to_w
    T_c_to_w[:3, 3:] = t_c_to_w.reshape(3, 1)
    print(f"pose_kf: {T_c_to_w}")
    self._update_motion_model(T_c_to_w)
    kf = KeyFrame(
      image=image,
      camera_matrix=self.camera_matrix,
      pose_c_to_w=T_c_to_w,
      keypoints=keypoints,
      descriptors=descriptors,
      color_image=color_image
    )

    for match, mp in zip(matches, matches_map_pts):
      if mp is None:
        continue
      kp_idx = match.trainIdx
      try:
        kf.add_map_point(mp, kp_idx)
      except ValueError:
        continue
      if mp.get_color() is None:
        sampled_color = self._sample_keypoint_color(keypoints[kp_idx], color_image, image)
        mp.set_color(sampled_color)
    keyframe_metrics = self._build_keyframe_metrics(len(matches), T_c_to_w)
    if self._should_insert_keyframe(keyframe_metrics):
      self.map.add_keyframe(kf)
      self.current_keyframe = kf
      self.last_keyframe_pose = T_c_to_w.copy()
      self.frames_since_last_keyframe = 0
      # Grow the map
      self.grow_map(kf)

      # Bundle Adjustment
      self.bundle_adjustment()
    else:
      print("Keyframe not added to map")
    
  def grow_map(self, kf: KeyFrame):
    """Grow the map by fusing observations and triangulating new points."""
    unmatched_indices = {idx for idx, mp in enumerate(kf.map_points) if mp is None}
    if not unmatched_indices:
      return

    local_keyframes = self.map.get_local_keyframes(kf, 10)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    touched_neighbors: set[int] = set()
    kf_modified = False

    for neighbor in local_keyframes:
      if neighbor.id == kf.id:
        continue

      neighbor_descriptors = neighbor.descriptors
      if neighbor_descriptors is None or len(neighbor_descriptors) == 0:
        continue

      candidate_indices = sorted(unmatched_indices)
      if not candidate_indices:
        break

      candidate_descriptors = kf.descriptors[candidate_indices]
      if candidate_descriptors is None or len(candidate_descriptors) == 0:
        break

      knn_matches = matcher.knnMatch(neighbor_descriptors, candidate_descriptors, k=2)
      triangulation_pairs: list[tuple[int, int]] = []

      for match_pair in knn_matches:
        if len(match_pair) < 2:
          continue
        m, n = match_pair
        if m.distance >= 0.75 * n.distance:
          continue

        neighbor_idx = m.queryIdx
        candidate_offset = m.trainIdx
        if candidate_offset >= len(candidate_indices):
          continue
        kf_idx = candidate_indices[candidate_offset]
        if kf_idx not in unmatched_indices:
          continue

        if neighbor_idx >= len(neighbor.map_points):
          continue

        neighbor_mp = neighbor.map_points[neighbor_idx]
        if neighbor_mp is not None and not neighbor_mp.is_bad:
          if not neighbor_mp.is_in_keyframe(kf.id):
            kf.add_map_point(neighbor_mp, kf_idx)
            if neighbor_mp.get_color() is None:
              sampled_color = self._sample_keypoint_color(kf.keypoints[kf_idx], kf.get_color_image(), kf.image)
              neighbor_mp.set_color(sampled_color)
            kf_modified = True
          unmatched_indices.discard(kf_idx)
          touched_neighbors.add(neighbor.id)
          continue

        triangulation_pairs.append((neighbor_idx, kf_idx))

      if not triangulation_pairs:
        continue

      new_points = self._triangulate_candidate_pairs(neighbor, kf, triangulation_pairs)
      for point_3d, neighbor_idx, kf_idx in new_points:
        if kf_idx not in unmatched_indices:
          continue

        descriptor = kf.descriptors[kf_idx]
        sampled_color = self._sample_keypoint_color(kf.keypoints[kf_idx], kf.get_color_image(), kf.image)
        mp = MapPoint(position=point_3d, descriptor=descriptor, color=sampled_color)
        neighbor.add_map_point(mp, neighbor_idx)
        kf.add_map_point(mp, kf_idx)
        self.map.add_map_point(mp)
        unmatched_indices.discard(kf_idx)
        touched_neighbors.add(neighbor.id)
        kf_modified = True

      if not unmatched_indices:
        break

    if kf_modified:
      kf.update_connections(self.map.keyframes)
    for neighbor_id in touched_neighbors:
      neighbor_kf: KeyFrame | None = self.map.get_keyframe(neighbor_id)
      if neighbor_kf is not None and not neighbor_kf.is_bad:
        neighbor_kf.update_connections(self.map.keyframes)

  def _triangulate_candidate_pairs(
    self,
    neighbor: KeyFrame,
    current_kf: KeyFrame,
    index_pairs: list[tuple[int, int]]
  ) -> list[tuple[np.ndarray, int, int]]:
    """Triangulate candidate feature correspondences between two keyframes."""
    if len(index_pairs) == 0:
      return []

    unique_pairs = []
    seen_pairs = set()
    for neighbor_idx, current_idx in index_pairs:
      key = (neighbor_idx, current_idx)
      if key in seen_pairs:
        continue
      seen_pairs.add(key)
      unique_pairs.append((neighbor_idx, current_idx))

    if len(unique_pairs) == 0:
      return []

    pts_neighbor = np.array([neighbor.keypoints[i].pt for i, _ in unique_pairs], dtype=np.float32)
    pts_current = np.array([current_kf.keypoints[j].pt for _, j in unique_pairs], dtype=np.float32)

    pose_neighbor_w_to_c = neighbor.get_pose_w_to_c()
    pose_current_w_to_c = current_kf.get_pose_w_to_c()

    P0 = self.camera_matrix @ pose_neighbor_w_to_c[:3, :]
    P1 = self.camera_matrix @ pose_current_w_to_c[:3, :]

    points_4d = cv2.triangulatePoints(P0, P1, pts_neighbor.T, pts_current.T)
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T

    valid_points: list[tuple[np.ndarray, int, int]] = []

    max_reproj_error = 3.0
    min_parallax_rad = np.deg2rad(1.0)
    cos_min_parallax = np.cos(min_parallax_rad)

    for (neighbor_idx, current_idx), point_3d, pt_neighbor, pt_current in zip(
      unique_pairs, points_3d, pts_neighbor, pts_current
    ):
      point_h = np.hstack([point_3d, 1.0])

      cam_neighbor = pose_neighbor_w_to_c @ point_h
      cam_current = pose_current_w_to_c @ point_h
      depth_neighbor = cam_neighbor[2]
      depth_current = cam_current[2]

      if depth_neighbor <= 0 or depth_current <= 0:
        continue

      proj_neighbor = P0 @ point_h
      proj_neighbor = proj_neighbor[:2] / proj_neighbor[2]
      proj_current = P1 @ point_h
      proj_current = proj_current[:2] / proj_current[2]

      err_neighbor = np.linalg.norm(pt_neighbor - proj_neighbor)
      err_current = np.linalg.norm(pt_current - proj_current)
      if err_neighbor > max_reproj_error or err_current > max_reproj_error:
        continue

      dir_neighbor = cam_neighbor[:3]
      dir_current = cam_current[:3]
      norm_neighbor = np.linalg.norm(dir_neighbor)
      norm_current = np.linalg.norm(dir_current)
      if norm_neighbor < 1e-6 or norm_current < 1e-6:
        continue

      cos_parallax = np.dot(dir_neighbor / norm_neighbor, dir_current / norm_current)
      if cos_parallax > cos_min_parallax:
        continue

      if not np.all(np.isfinite(point_3d)):
        continue

      valid_points.append((point_3d, neighbor_idx, current_idx))

    return valid_points


  def bundle_adjustment(self):
    """Bundle Adjustment"""
    pass
  
  def get_map_statistics(self):
    """Get statistics about the current map for visualization/debugging."""
    stats = {
      'n_keyframes': len(self.map.get_all_keyframes()),
      'n_map_points': len(self.map.get_all_map_points()),
      'init_method': self.init_method
    }
    
    # Get all 3D positions for visualization
    map_points = self.map.get_all_map_points()
    if len(map_points) > 0:
      positions = np.array([mp.position for mp in map_points])
      stats['map_points_3d'] = positions
      stats['mean_depth'] = np.mean(positions[:, 2])
      stats['std_depth'] = np.std(positions[:, 2])
      stats['min_depth'] = np.min(positions[:, 2])
      stats['max_depth'] = np.max(positions[:, 2])
    else:
      stats['map_points_3d'] = np.array([])
    
    return stats

if __name__ == "__main__":
  import argparse
  
  parser = argparse.ArgumentParser(description="SLAM Pipeline with visualization options")
  parser.add_argument("--video", type=str, default="data/office.MOV", help="Path to input video")
  parser.add_argument("--visualizer", type=str, choices=["matplotlib", "rerun"], default="matplotlib",
                      help="Visualization backend: 'matplotlib' (static) or 'rerun' (real-time)")
  parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process")
  args = parser.parse_args()
  
  if args.visualizer == "rerun":
    # Use Rerun for real-time visualization
    from rerun_visualizer import visualize_pipeline_with_rerun
    visualize_pipeline_with_rerun(args.video, IphoneCameraInstrinsics, max_frames=args.max_frames)
  else:
    # Use matplotlib for static visualization (original behavior)
    video_path = args.video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Unable to open video: {video_path}"

    front_end = Front_End(IphoneCameraInstrinsics)

    while True:
      ret, frame = cap.read()
      if not ret:
        break
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      front_end.ingest_image(gray, frame)
    cap.release()
    
    # Get and print map statistics
    stats = front_end.get_map_statistics()
    
    print("\n--- Map Statistics ---")
    print(f"KeyFrames: {stats['n_keyframes']}")
    print(f"MapPoints: {stats['n_map_points']}")
    print(f"Initialization method: {stats['init_method']}")
    
    if stats['n_map_points'] > 0:
      print(f"Mean depth: {stats['mean_depth']:.3f}")
      print(f"Std depth: {stats['std_depth']:.3f}")
      print(f"Min/Max depth: [{stats['min_depth']:.3f}, {stats['max_depth']:.3f}]")
    else:
      print("No map points triangulated")
    
    # Print covisibility information
    print("\n--- Covisibility Graph ---")
    for kf in front_end.map.get_all_keyframes():
      print(f"{kf}")
      if kf.parent:
        print(f"  Parent: KeyFrame {kf.parent.id}")
    
    print("\n" + "="*60)
    
    # Visualize the 3D map and camera poses
    print("\nGenerating 3D visualization...")
    visualize_frontend(
      front_end, 
      title="ORB-SLAM Initialization Result",
      save_path="slam_visualization.png"
    )

