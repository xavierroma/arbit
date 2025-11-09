from collections import namedtuple
import cv2
import numpy as np
from eth3_utils import load_case
from pathlib import Path
from enum import Enum, auto

class PipelineState(Enum):
    PRE_INIT = auto()
    INIT = auto()
    INITIALISED = auto()


CameraMatrix = namedtuple('CameraMatrix', ['cx', 'cy', 'fx', 'fy'])


class Front_End:
  def __init__(self, camera_matrix: CameraMatrix):
    self.orb = cv2.ORB_create()
    self.state: PipelineState = PipelineState.PRE_INIT
    self.prev_frame: np.ndarray | None = None
    self.camera_matrix = np.array([[camera_matrix.fx, 0, camera_matrix.cx], [0, camera_matrix.fy, camera_matrix.cy], [0, 0, 1]])
    self.pose_w_to_c = np.eye(4)

  def ingest_image(self, image: np.ndarray):
    match self.state:
      case PipelineState.PRE_INIT:
        self.pre_init(image)
      case PipelineState.INIT:
        self.init(image)
      case PipelineState.INITIALISED:
        self.initialised(image)

  def pre_init(self, image: np.ndarray):
    self.prev_frame = image
    self.state = PipelineState.INIT

  def init(self, image: np.ndarray):
    keypoints0, descriptors0 = self.orb.detectAndCompute(self.prev_frame, None)
    keypoints1, descriptors1 = self.orb.detectAndCompute(image, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors0, descriptors1)
    matches = sorted(matches, key=lambda x: x.distance)
    points0 = np.array([keypoints0[m.queryIdx].pt for m in matches])
    points1 = np.array([keypoints1[m.trainIdx].pt for m in matches])
    [es, mask] = cv2.findEssentialMat(
      points1=points0,
      points2=points1, 
      cameraMatrix=self.camera_matrix,
      method=cv2.LMEDS,
      # method=cv2.RANSAC,
    )
    mask_essential = mask.ravel().astype(bool)
    points0_inliers = points0[mask_essential]
    points1_inliers = points1[mask_essential]
    _, R, t, _ = cv2.recoverPose(es, points0_inliers, points1_inliers, cameraMatrix=self.camera_matrix)
    self.pose_w_to_c = np.hstack((R, t))
    self.state = PipelineState.INITIALISED


  def initialised(self, image: np.ndarray):
    pass

if __name__ == "__main__":
  case = load_case(Path('data/two-view/delivery_area_2s'))
  front_end = Front_End(CameraMatrix(cx=case.calib.cam0[0][2], cy=case.calib.cam0[1][2], fx=case.calib.cam0[0][0], fy=case.calib.cam0[1][1]))
  for im in [case.im0, case.im1]:
    front_end.ingest_image(cv2.imread(im, cv2.IMREAD_GRAYSCALE))

  gt_pose = case.gt_poses[0]
  print("Ground truth pose:")
  print(gt_pose)
  print("Recovered pose:")
  print(front_end.pose_w_to_c)
  print("Error:")
  R_gt = gt_pose['R']
  t_gt = gt_pose['t']
  R_rec = front_end.pose_w_to_c[:3, :3]
  t_rec = front_end.pose_w_to_c[:3, 3]
  # Rotation error using Frobenius norm
  R_error = np.linalg.norm(R_gt - R_rec, 'fro')
  print(f"Rotation error (Frobenius norm): {R_error:.6f}")
  # Rotation error in degrees using trace
  trace_val = np.trace(R_gt.T @ R_rec)
  # Clamp to valid range [-1, 3] to avoid numerical issues with arccos
  trace_val = np.clip((trace_val - 1) / 2, -1, 1)
  angle_error = np.arccos(trace_val) * 180 / np.pi
  print(f"Rotation error (angle): {angle_error:.4f} degrees")

  # Translation error (direction only, since scale is ambiguous)
  t_gt_normalized = t_gt / np.linalg.norm(t_gt)
  t_rec_normalized = t_rec / np.linalg.norm(t_rec)
  t_angle_error = np.arccos(np.clip(np.dot(t_gt_normalized.T, t_rec_normalized)[0], -1, 1))
  t_angle_error_deg = t_angle_error * 180 / np.pi
  print(f"Translation direction error: {t_angle_error_deg:.4f} degrees")

