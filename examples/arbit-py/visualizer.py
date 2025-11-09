import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D, proj3d  # type: ignore
from typing import Optional


class Arrow3D(FancyArrowPatch):
    """Helper class for drawing 3D arrows in matplotlib."""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


class MapVisualizer:
    """Visualizes 3D map points and camera poses from the SLAM frontend."""
    
    def __init__(self, figsize=(12, 9)):
        """
        Initialize the visualizer.
        
        Args:
            figsize: Figure size as (width, height) tuple
        """
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.camera_poses = []
        self.map_points = None
        
    def add_camera_pose(self, R: np.ndarray, t: np.ndarray, 
                       label: str = "", color: str = 'blue', 
                       scale: float = 1.0):
        """
        Add a camera pose to the visualization.
        
        Args:
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
            label: Label for this camera
            color: Color for the camera frustum
            scale: Scale factor for camera frustum size
        """
        self.camera_poses.append({
            'R': R,
            't': t,
            'label': label,
            'color': color,
            'scale': scale
        })
        
    def set_map_points(self, points_3d: np.ndarray):
        """
        Set the 3D map points to visualize.
        
        Args:
            points_3d: Nx3 array of 3D points in world coordinates
        """
        self.map_points = points_3d
        
    def _draw_camera_frustum(self, R: np.ndarray, t: np.ndarray, 
                            color: str = 'blue', scale: float = 1.0,
                            label: str = ""):
        """
        Draw a camera frustum to represent the camera pose.
        
        Args:
            R: 3x3 rotation matrix (world to camera)
            t: 3x1 translation vector (world to camera)
            color: Color for the frustum
            scale: Scale factor for frustum size
            label: Label for the camera
        """
        # Camera center in world coordinates: C = -R^T * t
        C = -R.T @ t.reshape(3, 1)
        C = C.flatten()
        
        # Camera coordinate system axes in world frame
        # The camera looks along +Z in camera frame
        x_axis = R.T[:, 0] * scale  # Right
        y_axis = R.T[:, 1] * scale  # Down
        z_axis = R.T[:, 2] * scale  # Forward (viewing direction)
        
        # Draw coordinate axes
        arrow_props = dict(mutation_scale=20, lw=2, arrowstyle='-|>', color=color)
        
        # X-axis (red component)
        x_color = (min(1.0, float(color == 'red') + 0.5), 0, 0)
        arrow_x = Arrow3D([C[0], C[0] + x_axis[0]], 
                         [C[1], C[1] + x_axis[1]], 
                         [C[2], C[2] + x_axis[2]], 
                         **{**arrow_props, 'color': 'red' if color == 'blue' else x_color})
        self.ax.add_artist(arrow_x)
        
        # Y-axis (green component)
        y_color = (0, min(1.0, float(color == 'green') + 0.5), 0)
        arrow_y = Arrow3D([C[0], C[0] + y_axis[0]], 
                         [C[1], C[1] + y_axis[1]], 
                         [C[2], C[2] + y_axis[2]], 
                         **{**arrow_props, 'color': 'green' if color == 'blue' else y_color})
        self.ax.add_artist(arrow_y)
        
        # Z-axis (blue/forward direction)
        arrow_z = Arrow3D([C[0], C[0] + z_axis[0]], 
                         [C[1], C[1] + z_axis[1]], 
                         [C[2], C[2] + z_axis[2]], 
                         **{**arrow_props, 'color': color})
        self.ax.add_artist(arrow_z)
        
        # Draw camera frustum pyramid
        frustum_scale = scale * 0.5
        frustum_depth = scale * 1.5
        
        # Define frustum corners in camera frame (image plane corners)
        corners_cam = np.array([
            [-frustum_scale, -frustum_scale, frustum_depth],
            [ frustum_scale, -frustum_scale, frustum_depth],
            [ frustum_scale,  frustum_scale, frustum_depth],
            [-frustum_scale,  frustum_scale, frustum_depth],
        ]).T
        
        # Transform to world frame
        corners_world = R.T @ corners_cam + C.reshape(3, 1)
        
        # Draw lines from camera center to frustum corners
        for i in range(4):
            self.ax.plot([C[0], corners_world[0, i]], 
                        [C[1], corners_world[1, i]], 
                        [C[2], corners_world[2, i]], 
                        color=color, alpha=0.6, linewidth=1)
        
        # Draw frustum edges
        for i in range(4):
            next_i = (i + 1) % 4
            self.ax.plot([corners_world[0, i], corners_world[0, next_i]], 
                        [corners_world[1, i], corners_world[1, next_i]], 
                        [corners_world[2, i], corners_world[2, next_i]], 
                        color=color, alpha=0.6, linewidth=1)
        
        # Add label
        if label:
            self.ax.text(C[0], C[1], C[2], f'  {label}', 
                        color=color, fontsize=10, fontweight='bold')
        
    def visualize(self, title: str = "3D Map and Camera Poses",
                 point_size: float = 20, point_color: str = 'black',
                 show_grid: bool = True, equal_aspect: bool = True):
        """
        Generate and display the visualization.
        
        Args:
            title: Title for the plot
            point_size: Size of the map point markers
            point_color: Color for the map points
            show_grid: Whether to show grid
            equal_aspect: Whether to use equal aspect ratio for all axes
        """
        # Draw map points
        if self.map_points is not None and len(self.map_points) > 0:
            self.ax.scatter(self.map_points[:, 0], 
                          self.map_points[:, 1], 
                          self.map_points[:, 2], 
                          c=point_color, marker='o', s=point_size, 
                          alpha=0.6, label='Map Points')
        
        # Draw all camera poses
        for i, cam in enumerate(self.camera_poses):
            label = cam['label'] if cam['label'] else f"Camera {i}"
            self._draw_camera_frustum(cam['R'], cam['t'], 
                                     color=cam['color'], 
                                     scale=cam['scale'],
                                     label=label)
        
        # Set labels and title
        self.ax.set_xlabel('X (World)', fontsize=10)
        self.ax.set_ylabel('Y (World)', fontsize=10)
        self.ax.set_zlabel('Z (World)', fontsize=10)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set equal aspect ratio if requested
        if equal_aspect:
            # Get the data limits
            all_points_list: list[np.ndarray] = []
            if self.map_points is not None and len(self.map_points) > 0:
                all_points_list.append(self.map_points)
            
            for cam in self.camera_poses:
                C = -cam['R'].T @ cam['t'].reshape(3, 1)
                all_points_list.append(C.T)
            
            if all_points_list:
                all_points_array = np.vstack(all_points_list)
                max_range = np.array([
                    all_points_array[:, 0].max() - all_points_array[:, 0].min(),
                    all_points_array[:, 1].max() - all_points_array[:, 1].min(),
                    all_points_array[:, 2].max() - all_points_array[:, 2].min()
                ]).max() / 2.0
                
                mid_x = (all_points_array[:, 0].max() + all_points_array[:, 0].min()) * 0.5
                mid_y = (all_points_array[:, 1].max() + all_points_array[:, 1].min()) * 0.5
                mid_z = (all_points_array[:, 2].max() + all_points_array[:, 2].min()) * 0.5
                
                self.ax.set_xlim(mid_x - max_range, mid_x + max_range)
                self.ax.set_ylim(mid_y - max_range, mid_y + max_range)
                self.ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Grid
        self.ax.grid(show_grid)
        
        # Legend
        if self.map_points is not None and len(self.map_points) > 0:
            self.ax.legend()
        
        # Better viewing angle
        self.ax.view_init(elev=20, azim=45)
        
        plt.tight_layout()
        
    def show(self):
        """Display the visualization."""
        plt.show()
        
    def save(self, filename: str, dpi: int = 300):
        """
        Save the visualization to a file.
        
        Args:
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        plt.savefig(filename, dpi=dpi, bbox_inches='tight')
        print(f"Saved visualization to {filename}")


def visualize_frontend(front_end, title: str = "SLAM Initialization",
                       save_path: Optional[str] = None):
    """
    Convenience function to visualize a Front_End instance.
    
    Args:
        front_end: Front_End instance with initialized map
        title: Title for the plot
        save_path: Optional path to save the figure
    """
    if front_end.state.name != 'INITIALISED':
        print(f"Warning: Frontend state is {front_end.state.name}, not INITIALISED")
        return
    
    if front_end.map_points_3d is None or len(front_end.map_points_3d) == 0:
        print("Error: No map points to visualize")
        return
    
    # Create visualizer
    viz = MapVisualizer(figsize=(14, 10))
    
    # Add camera poses
    # Camera 0 at origin (identity)
    viz.add_camera_pose(
        R=np.eye(3),
        t=np.zeros((3, 1)),
        label="Camera 0 (Reference)",
        color="blue",
        scale=2.0
    )
    
    # Camera 1 at recovered pose
    R_cam1 = front_end.pose_w_to_c[:3, :3]
    t_cam1 = front_end.pose_w_to_c[:3, 3:4]
    viz.add_camera_pose(
        R=R_cam1,
        t=t_cam1,
        label="Camera 1 (Current)",
        color="red",
        scale=2.0
    )
    
    # Add map points
    viz.set_map_points(front_end.map_points_3d)
    
    # Visualize
    full_title = f"{title}\nMethod: {front_end.init_method} | Points: {len(front_end.map_points_3d)}"
    viz.visualize(title=full_title, point_size=30, point_color='darkgreen')
    
    # Save if requested
    if save_path:
        viz.save(save_path)
    
    viz.show()

