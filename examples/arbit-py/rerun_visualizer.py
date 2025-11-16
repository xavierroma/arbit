"""
Rerun-based visualizer for the SLAM pipeline.

This visualizer provides real-time, interactive 3D visualization of:
- Camera poses and trajectories
- 3D map points
- Feature matches
- Current frame images
- Covisibility graph
"""

import numpy as np
import cv2
import rerun as rr
from typing import Optional
from pathlib import Path

from slam import Map, KeyFrame, MapPoint


class RerunSLAMVisualizer:
    """Real-time SLAM visualization using Rerun."""
    
    def __init__(self, recording_name: str = "SLAM Pipeline"):
        """
        Initialize the Rerun visualizer.
        
        Args:
            recording_name: Name for the Rerun recording
        """
        rr.init(recording_name, spawn=True)
        self.frame_count = 0
        
        # Set up coordinate system description
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
        # Log world origin
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
        
    def log_frame(self, image: np.ndarray, frame_id: int, is_keyframe: bool = False):
        """
        Log a video frame.
        
        Args:
            image: Grayscale or color image
            frame_id: Frame number
            is_keyframe: Whether this frame is a keyframe
        """
        rr.set_time_sequence("frame", frame_id)
        
        # Log the image
        if len(image.shape) == 2:
            # Grayscale
            rr.log("camera/image", rr.Image(image))
        else:
            # Color
            rr.log("camera/image", rr.Image(image))
        
        # Log keyframe indicator
        if is_keyframe:
            rr.log("events/keyframe", rr.TextLog(f"KeyFrame {frame_id}"))
    
    def log_features(self, keypoints: list, frame_id: int):
        """
        Log 2D feature keypoints on the current frame.
        
        Args:
            keypoints: List of cv2.KeyPoint objects
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        if keypoints is not None and len(keypoints) > 0:
            # Extract positions
            positions = np.array([kp.pt for kp in keypoints])
            
            # Log as 2D points
            rr.log(
                "camera/features",
                rr.Points2D(
                    positions,
                    radii=3.0,
                    colors=[0, 255, 0, 128],  # Green with transparency
                )
            )
    
    def log_matches(self, points0: np.ndarray, points1: np.ndarray, 
                   inlier_mask: Optional[np.ndarray] = None, frame_id: int = 0):
        """
        Log feature matches between two frames.
        
        Args:
            points0: Nx2 array of points in first frame
            points1: Nx2 array of points in second frame
            inlier_mask: Boolean mask indicating inliers
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        if inlier_mask is not None:
            inlier_points0 = points0[inlier_mask]
            inlier_points1 = points1[inlier_mask]
            outlier_points0 = points0[~inlier_mask]
            outlier_points1 = points1[~inlier_mask]
            
            # Log inliers in green
            if len(inlier_points1) > 0:
                rr.log(
                    "camera/matches/inliers",
                    rr.Points2D(
                        inlier_points1,
                        radii=4.0,
                        colors=[0, 255, 0, 255],
                    )
                )
            
            # Log outliers in red
            if len(outlier_points1) > 0:
                rr.log(
                    "camera/matches/outliers",
                    rr.Points2D(
                        outlier_points1,
                        radii=3.0,
                        colors=[255, 0, 0, 128],
                    )
                )
        else:
            # Log all matches
            rr.log(
                "camera/matches/all",
                rr.Points2D(
                    points1,
                    radii=3.0,
                    colors=[0, 255, 255, 255],
                )
            )
    
    def log_camera_pose(self, R: np.ndarray, t: np.ndarray, 
                       frame_id: int, label: str = "camera",
                       image: Optional[np.ndarray] = None,
                       camera_matrix: Optional[np.ndarray] = None):
        """
        Log a camera pose in 3D with optional image and pinhole camera.
        
        Args:
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 or (3,) translation vector (world to camera)
            frame_id: Frame number
            label: Label for this camera
            image: Optional image from this camera pose
            camera_matrix: Optional 3x3 camera intrinsic matrix for pinhole visualization
        """
        rr.set_time_sequence("frame", frame_id)
        
        # Camera center in world coordinates: C = -R^T * t
        t = t.flatten()
        C = -R.T @ t
        
        # Create transform (camera-to-world)
        # Log the camera pose using Transform3D
        rr.log(
            f"world/{label}",
            rr.Transform3D(
                translation=C,
                mat3x3=R.T,
            )
        )
        
        # If we have camera matrix and image, set up pinhole camera
        if camera_matrix is not None and image is not None:
            fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
            cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
            height, width = image.shape[:2]
            
            # Log pinhole camera
            rr.log(
                f"world/{label}",
                rr.Pinhole(
                    focal_length=[fx, fy],
                    principal_point=[cx, cy],
                    resolution=[width, height],
                )
            )
            
            # Log the image at this camera location
            rr.log(f"world/{label}", rr.Image(image))
        
        # Log a small coordinate frame at the camera position for visualization
        axis_length = 0.5
        axes_points = np.array([
            C,  # Origin
            C + R.T[:, 0] * axis_length,  # X-axis (right)
            C,
            C + R.T[:, 1] * axis_length,  # Y-axis (down)
            C,
            C + R.T[:, 2] * axis_length,  # Z-axis (forward)
        ])
        
        # Log axes as line strips
        rr.log(
            f"world/{label}/axes",
            rr.LineStrips3D(
                [axes_points[0:2], axes_points[2:4], axes_points[4:6]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
                radii=0.02,
            )
        )
    
    def log_keyframe(self, keyframe: KeyFrame, frame_id: int):
        """
        Log a keyframe with its pose, image, and features.
        
        Args:
            keyframe: KeyFrame object
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        # Get world-to-camera pose
        pose_w_to_c = keyframe.get_pose_w_to_c()
        R = pose_w_to_c[:3, :3]
        t = pose_w_to_c[:3, 3]
        
        # Log camera pose with image and pinhole camera
        label = f"keyframe_{keyframe.id}"
        self.log_camera_pose(
            R, t, frame_id, label,
            image=keyframe.image,
            camera_matrix=keyframe.camera_matrix
        )
        
        # Log the keyframe's map points
        if len(keyframe.map_points) > 0:
            map_points_3d = np.array([
                mp.position for mp in keyframe.map_points if mp is not None
            ])
            
            if len(map_points_3d) > 0:
                rr.log(
                    f"world/{label}/observed_points",
                    rr.Points3D(
                        map_points_3d,
                        radii=0.05,
                        colors=[0, 255, 0, 200],
                    )
                )
    
    def log_map_points(self, map_points: list[MapPoint], frame_id: int):
        """
        Log all 3D map points.
        
        Args:
            map_points: List of MapPoint objects
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        if len(map_points) == 0:
            return
        
        # Extract 3D positions
        positions = np.array([mp.position for mp in map_points])
        
        # Color by observation count (more observations = brighter)
        observation_counts = np.array([len(mp.observations) for mp in map_points])
        max_obs = max(observation_counts) if len(observation_counts) > 0 else 1
        
        # Normalize to [0, 1] and create color gradient (dark green to bright green)
        normalized_obs = observation_counts / max(max_obs, 1)
        colors = np.zeros((len(map_points), 4), dtype=np.uint8)
        colors[:, 1] = (50 + 205 * normalized_obs).astype(np.uint8)  # Green channel
        colors[:, 3] = 255  # Alpha
        
        rr.log(
            "world/map_points",
            rr.Points3D(
                positions,
                radii=0.03,
                colors=colors,
            )
        )
    
    def log_camera_trajectory(self, keyframes: list[KeyFrame], frame_id: int):
        """
        Log the camera trajectory as a line connecting keyframe positions.
        Also sets transform for trajectory visualization.
        
        Args:
            keyframes: List of KeyFrame objects
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        if len(keyframes) == 0:
            return
        
        # Extract camera centers
        positions_list = []
        for kf in keyframes:
            pose_w_to_c = kf.get_pose_w_to_c()
            R = pose_w_to_c[:3, :3]
            t = pose_w_to_c[:3, 3]
            C = -R.T @ t
            positions_list.append(C)
        
        positions = np.array(positions_list)
        
        # Log the latest camera position in the trajectory
        if len(positions) > 0:
            latest_pos = positions[-1]
            rr.log(
                "world/trajectory",
                rr.Transform3D(translation=latest_pos)
            )
        
        # Log full trajectory as line strip
        if len(positions) >= 2:
            rr.log(
                "world/trajectory/path",
                rr.LineStrips3D(
                    [positions],
                    colors=[255, 200, 0],  # Yellow
                    radii=0.02,
                )
            )
    
    def log_covisibility_graph(self, keyframes: list[KeyFrame], frame_id: int):
        """
        Log the covisibility graph between keyframes.
        
        Args:
            keyframes: List of KeyFrame objects
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        if len(keyframes) < 2:
            return
        
        # Build edges based on covisibility
        edges = []
        for kf in keyframes:
            if kf.parent is not None:
                # Get camera centers
                pose1 = kf.get_pose_w_to_c()
                R1, t1 = pose1[:3, :3], pose1[:3, 3]
                C1 = -R1.T @ t1
                
                pose2 = kf.parent.get_pose_w_to_c()
                R2, t2 = pose2[:3, :3], pose2[:3, 3]
                C2 = -R2.T @ t2
                
                edges.append([C1, C2])
        
        if len(edges) > 0:
            rr.log(
                "world/covisibility",
                rr.LineStrips3D(
                    edges,
                    colors=[255, 0, 255],  # Magenta
                    radii=0.01,
                )
            )
    
    def log_map(self, map_obj: Map, frame_id: int):
        """
        Log the entire map state.
        
        Args:
            map_obj: Map object containing keyframes and map points
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        # Log all map points
        map_points = map_obj.get_all_map_points()
        self.log_map_points(map_points, frame_id)
        
        # Log all keyframes
        keyframes = map_obj.get_all_keyframes()
        for kf in keyframes:
            self.log_keyframe(kf, frame_id)
        
        # Log trajectory
        self.log_camera_trajectory(keyframes, frame_id)
        
        # Log covisibility graph
        self.log_covisibility_graph(keyframes, frame_id)
        
        # Log statistics as text
        rr.log(
            "stats/map_info",
            rr.TextLog(
                f"KeyFrames: {len(keyframes)} | MapPoints: {len(map_points)}"
            )
        )
    
    def log_initialization_info(self, method: str, score_H: float, score_F: float, 
                               R_H: float, frame_id: int):
        """
        Log initialization method selection info.
        
        Args:
            method: "Homography" or "Fundamental"
            score_H: Homography score
            score_F: Fundamental matrix score
            R_H: Ratio R_H = score_H / (score_H + score_F)
            frame_id: Frame number
        """
        rr.set_time_sequence("frame", frame_id)
        
        info_text = (
            f"Init Method: {method}\n"
            f"H Score: {score_H:.1f}\n"
            f"F Score: {score_F:.1f}\n"
            f"R_H: {R_H:.3f} (threshold: 0.45)"
        )
        
        rr.log("stats/initialization", rr.TextLog(info_text))


def visualize_pipeline_with_rerun(video_path: str, camera_matrix, max_frames: Optional[int] = None):
    """
    Run the SLAM pipeline with real-time Rerun visualization.
    
    Args:
        video_path: Path to input video
        camera_matrix: Camera intrinsic matrix or CameraMatrix object
        max_frames: Optional maximum number of frames to process
    """
    from pipeline import Front_End, PipelineState
    from slam import to_matrix
    
    # Initialize visualizer
    viz = RerunSLAMVisualizer(recording_name=f"SLAM: {Path(video_path).name}")
    
    # Get camera matrix as numpy array
    camera_matrix_np = to_matrix(camera_matrix)
    
    # Initialize pipeline
    front_end = Front_End(camera_matrix)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), f"Unable to open video: {video_path}"
    
    frame_id = 0
    
    print("Starting SLAM pipeline with Rerun visualization...")
    print("Rerun viewer should open automatically.")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Log the current frame image (separate from 3D cameras)
        viz.log_frame(gray, frame_id, is_keyframe=False)
        
        # Process frame
        prev_state = front_end.state
        front_end.ingest_image(gray)
        
        # Log features if available
        if front_end.prev_keypoints is not None:
            viz.log_features(front_end.prev_keypoints, frame_id)
        
        # Log map state if initialized
        if front_end.state == PipelineState.INITIALISED:
            viz.log_map(front_end.map, frame_id)
            
            # Log initialization info on first initialization
            if prev_state != PipelineState.INITIALISED:
                print(f"\n✓ SLAM Initialized at frame {frame_id}")
                print(f"  Method: {front_end.init_method}")
                viz.log_initialization_info(
                    method=front_end.init_method,
                    score_H=0.0,  # Would need to pass these from init()
                    score_F=0.0,
                    R_H=0.0,
                    frame_id=frame_id
                )
            
            tracking_pose = getattr(front_end, "tracking_pose", None)
            pose_w_to_c = None
            if tracking_pose is not None:
                try:
                    pose_w_to_c = np.linalg.inv(tracking_pose)
                except np.linalg.LinAlgError:
                    pose_w_to_c = None
            elif front_end.current_keyframe is not None:
                pose_w_to_c = front_end.current_keyframe.get_pose_w_to_c()

            if pose_w_to_c is not None:
                R = pose_w_to_c[:3, :3]
                t = pose_w_to_c[:3, 3]
                
                viz.log_camera_pose(
                    R, t, frame_id, "tracking_camera",
                    image=gray,
                    camera_matrix=camera_matrix_np
                )
        
        frame_id += 1
        
        # Progress indicator
        if frame_id % 10 == 0:
            state_str = front_end.state.name
            n_kf = len(front_end.map.get_all_keyframes()) if front_end.state == PipelineState.INITIALISED else 0
            n_mp = len(front_end.map.get_all_map_points()) if front_end.state == PipelineState.INITIALISED else 0
            print(f"Frame {frame_id}: {state_str} | KF: {n_kf} | MP: {n_mp}", end="\r")
        
        # Limit processing if specified
        if max_frames and frame_id >= max_frames:
            print(f"\n\nReached max frames limit ({max_frames})")
            break
    
    cap.release()
    
    print(f"\n\n{'='*60}")
    print(f"Processed {frame_id} frames")
    
    if front_end.state == PipelineState.INITIALISED:
        stats = front_end.get_map_statistics()
        print(f"Final map: {stats['n_keyframes']} keyframes, {stats['n_map_points']} map points")
    
    print("Rerun visualization complete.")
    print("The Rerun viewer will remain open for exploration.")
    
    # Log completion
    rr.log("events/end", rr.TextLog("Pipeline complete"))


if __name__ == "__main__":
    import sys
    from video_utils import IphoneCameraInstrinsics
    
    # Example usage
    video_path = "data/office.MOV"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    visualize_pipeline_with_rerun(
        video_path=video_path,
        camera_matrix=IphoneCameraInstrinsics
    )

