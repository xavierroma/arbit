"""
Arbit Python bindings - Visual-Inertial SLAM engine
"""

from .engine import ArbitEngine, CameraFrame, ImuSample
from .types import PixelFormat, MotionState, TrackStatus, Transform, FrameState, ImuState

__all__ = [
    "ArbitEngine",
    "CameraFrame", 
    "ImuSample",
    "PixelFormat",
    "MotionState",
    "TrackStatus",
    "Transform",
    "FrameState",
    "ImuState",
    "init_logging",
]


def init_logging(verbose: bool = True):
    """
    Initialize Arbit logging for debugging.
    
    Call this before creating any ArbitEngine instances to enable detailed
    logging from the Rust engine. Set RUST_LOG environment variable to
    control log levels (e.g., RUST_LOG=debug).
    
    Args:
        verbose: Whether to print library location
        
    Example:
        >>> import arbit
        >>> arbit.init_logging()
        >>> engine = arbit.ArbitEngine()
    """
    ArbitEngine.init_logging(verbose=verbose)

