from collections import namedtuple
from pathlib import Path
import numpy as np


def load_calibration(file):
    """
    Parse calibration file containing camera intrinsics and stereo parameters.
    
    Args:
        file: Path to calibration file
        
    Returns:
        dict: Dictionary containing:
            - cam0: 3x3 numpy array (left camera intrinsic matrix)
            - cam1: 3x3 numpy array (right camera intrinsic matrix)
            - doffs: float (disparity offset)
            - baseline: float (stereo baseline in mm)
            - width: int (image width)
            - height: int (image height)
            - ndisp: int (number of disparities)
    """
    calib = {}
    
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            key, value = line.split('=')
            
            if key in ['cam0', 'cam1']:
                matrix_str = value.strip('[]')
                rows = matrix_str.split(';')
                matrix = []
                for row in rows:
                    matrix.append([float(x) for x in row.split()])
                calib[key] = np.array(matrix)
            elif key in ['width', 'height', 'ndisp']:
                calib[key] = int(value)
            else:  # doffs, baseline
                calib[key] = float(value)
    
    return calib
def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to 3x3 rotation matrix.
    Uses the standard quaternion to rotation matrix formula.
    
    Args:
        qw, qx, qy, qz: Quaternion components (scalar-first convention)
    
    Returns:
        3x3 numpy rotation matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Compute rotation matrix
    R = np.array([
        [1 - 2*(qy**2 + qz**2), 2*(qx*qy - qw*qz), 2*(qx*qz + qw*qy)],
        [2*(qx*qy + qw*qz), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qw*qx)],
        [2*(qx*qz - qw*qy), 2*(qy*qz + qw*qx), 1 - 2*(qx**2 + qy**2)]
    ])
    return R


def load_ground_truth_poses(file):
    """
    Parse ETH3D images.txt file to extract ground truth poses.
    
    Args:
        file: Path to images.txt file
    
    Returns:
        dict: Dictionary mapping image_id to pose dict containing:
            - R: 3x3 rotation matrix (world to camera)
            - t: 3x1 translation vector (world to camera)
            - name: image filename
    """
    poses = {}
    
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            if len(parts) >= 10:
                image_id = int(parts[0])
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                camera_id = int(parts[8])
                name = parts[9]
                
                R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
                t = np.array([[tx], [ty], [tz]])
                
                poses[image_id] = {
                    'R': R,
                    't': t,
                    'camera_id': camera_id,
                    'name': name
                }
    
    return poses


Eth3Calib = namedtuple('Eth3Calib', ['cam0', 'cam1', 'doffs', 'baseline', 'width', 'height', 'ndisp'])
class Eth3Case:
  def __init__(self, im0: Path, im1: Path, calib: Eth3Calib, gt_poses: dict):
    self.im0 = im0
    self.im1 = im1
    self.calib = calib
    self.gt_poses = gt_poses

def load_case(directory: Path):
    im0path = directory / 'im0.png'
    im1path = directory / 'im1.png'
    assert im0path.exists() and im1path.exists(), f"Images not found in {directory}"
    assert (directory / 'calib.txt').exists(), f"Calibration file not found in {directory}"
    assert (directory / 'images.txt').exists(), f"Ground truth poses file not found in {directory}"
    calib = load_calibration(directory / 'calib.txt')
    gt = load_ground_truth_poses(directory / 'images.txt')
    return Eth3Case(im0path, im1path, Eth3Calib(**calib), gt_poses=gt)