import numpy as np
from slam import from_intrinsics

IphoneCameraInstrinsics = from_intrinsics(np.array([
  [840.164, 0, 640.0],
  [0, 840.164, 360.0],
  [0, 0, 1]
]))

