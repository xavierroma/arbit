"""Camera intrinsic parameters container."""
from collections import namedtuple
import numpy as np

CameraMatrix = namedtuple('CameraMatrix', ['cx', 'cy', 'fx', 'fy'])


def to_matrix(camera_matrix: CameraMatrix) -> np.ndarray:
    """Convert CameraMatrix to 3x3 K matrix.
    
    Args:
        camera_matrix: CameraMatrix namedtuple
        
    Returns:
        3x3 numpy array representing the camera intrinsic matrix K
    """
    return np.array([
        [camera_matrix.fx, 0, camera_matrix.cx],
        [0, camera_matrix.fy, camera_matrix.cy],
        [0, 0, 1]
    ])

