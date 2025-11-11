from collections import namedtuple
import cv2
import numpy as np
from eth3_utils import load_case
from pathlib import Path
from enum import Enum, auto
from visualizer import visualize_frontend

from slam import CameraMatrix, Map, KeyFrame, MapPoint, to_matrix

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
    
    self.prev_frame: np.ndarray | None = None
    self.prev_keypoints: list | None = None
    self.prev_descriptors: np.ndarray | None = None
    
    # Tracking info
    self.init_method: str = ""  # Track which method was used (H or F)
    self.current_keyframe: KeyFrame | None = None

  def ingest_image(self, image: np.ndarray):
    match self.state:
      case PipelineState.PRE_INIT:
        self.pre_init(image)
      case PipelineState.INIT:
        self.init(image)
      case PipelineState.INITIALISED:
        self.initialised(image)

  def pre_init(self, image: np.ndarray):
    """Store first frame and extract features."""
    self.prev_frame = image
    self.prev_keypoints, self.prev_descriptors = self.orb.detectAndCompute(image, None)
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

  def triangulate_points(self, points0: np.ndarray, points1: np.ndarray, 
                         R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Triangulate 3D points from two views.
    Returns 3D points in world frame and mask of valid points.
    """
    # Projection matrices: P0 = K[I|0], P1 = K[R|t]
    P0 = self.camera_matrix @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = self.camera_matrix @ np.hstack([R, t])
    
    # Triangulate
    points_4d = cv2.triangulatePoints(P0, P1, points0.T, points1.T)
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T
    
    # Filter by positive depth in both cameras
    # Camera 0 (identity): just check Z > 0
    depth0 = points_3d[:, 2]
    
    # Camera 1: transform to camera 1 frame
    points_3d_cam1 = (R @ points_3d.T + t).T
    depth1 = points_3d_cam1[:, 2]
    
    # Keep only points with positive depth in both views and reasonable depth
    max_depth = 100.0  # Reasonable for most scenes
    valid_depth = (depth0 > 0) & (depth1 > 0) & (depth0 < max_depth) & (depth1 < max_depth)
    
    # Also filter by reprojection error
    points_3d_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])
    
    # Reproject to both cameras
    proj0 = (P0 @ points_3d_h.T).T
    proj0 = proj0[:, :2] / proj0[:, 2:3]
    reproj_error0 = np.sum((points0 - proj0) ** 2, axis=1)
    
    proj1 = (P1 @ points_3d_h.T).T
    proj1 = proj1[:, :2] / proj1[:, 2:3]
    reproj_error1 = np.sum((points1 - proj1) ** 2, axis=1)
    
    reproj_threshold = 5.991  # Chi-square threshold
    valid_reproj = (reproj_error0 < reproj_threshold) & (reproj_error1 < reproj_threshold)
    
    valid_mask = valid_depth & valid_reproj
    
    return points_3d, valid_mask

  def init(self, image: np.ndarray):
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
        points_3d, valid_mask = self.triangulate_points(points0_inliers, points1_inliers, R_test, t_test)
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
      points_3d, valid_mask = self.triangulate_points(points0_inliers, points1_inliers, R, t)
    
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
      descriptors=descriptors0
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
      descriptors=descriptors1
    )
    
    # Create MapPoints and link to KeyFrames
    n_valid = np.sum(valid_mask)
    map_point_ids = []
    
    for i, (is_valid, point_3d) in enumerate(zip(valid_mask, points_3d)):
      if not is_valid:
        continue
      
      # Get the match index for this inlier
      match_idx = match_indices[i]
      match = matches[match_idx]
      kp0_idx = match.queryIdx  # Index in keypoints0
      kp1_idx = match.trainIdx  # Index in keypoints1
      
      # Create MapPoint WITHOUT initial descriptor to avoid duplication bug
      mp = MapPoint(position=point_3d)
      
      # Add to Map first
      self.map.add_map_point(mp)
      map_point_ids.append(mp.id)
      
      # Link to KeyFrames (this will add descriptors)
      # TODO: simplify interface between KeyFrame and MapPoint; MapPoint is always created in the same way
      kf0.add_map_point(mp, kp0_idx)
      kf1.add_map_point(mp, kp1_idx)
    
    # Add KeyFrames to Map
    self.map.add_keyframe(kf0)
    self.map.add_keyframe(kf1)
    
    # Set current keyframe for tracking
    self.current_keyframe = kf1
    
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


  def initialised(self, image: np.ndarray):
    """Process frames after initialization (tracking state)."""
    # TODO: Implement tracking against local map
    # 1. Extract features from current frame
    # 2. Match against local map points
    # 3. Estimate pose via PnP
    # 4. Decide if new keyframe needed
    # 5. Local mapping / BA
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
  case = load_case(Path('data/two-view/exa'))
  front_end = Front_End(CameraMatrix(cx=case.calib.cam0[0][2], cy=case.calib.cam0[1][2], fx=case.calib.cam0[0][0], fy=case.calib.cam0[1][1]))
  
  # Process images
  for im in [case.im0, case.im1]:
    front_end.ingest_image(cv2.imread(im, cv2.IMREAD_GRAYSCALE))
  
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

