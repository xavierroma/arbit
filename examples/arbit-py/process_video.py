from time import sleep
from math import ceil
from typing import List, Tuple

import numpy as np
import cv2
import arbit

from flow_visualizations import (
    draw_flow_arrows,
    draw_flow_dense_hsv,
    draw_flow_heatmap,
    draw_flow_circles_with_arrows,
    create_color_wheel_legend
)


def draw_keypoint_descriptors(image: np.ndarray, descriptors: List[arbit.FeatDescriptor]) -> np.ndarray:
    """Overlay descriptor positions and orientations on a copy of the frame."""
    overlay = image.copy()
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (255, 0, 255),
        (0, 255, 255),
        (255, 128, 0),
        (128, 0, 255),
    ]

    for descriptor in descriptors:
        pos = descriptor.position
        x, y = int(round(pos[0])), int(round(pos[1]))
        if x < 0 or y < 0 or x >= overlay.shape[1] or y >= overlay.shape[0]:
            continue

        color = colors[descriptor.level % len(colors)]
        cv2.circle(overlay, (x, y), 3, color, -1)

        angle = descriptor.angle
        length = 12
        end_point = (
            int(round(x + np.cos(angle) * length)),
            int(round(y + np.sin(angle) * length)),
        )
        cv2.line(overlay, (x, y), end_point, color, 1, lineType=cv2.LINE_AA)

    return overlay


def build_keyframe_grid(
    keyframes: List[Tuple[int, np.ndarray, List[arbit.FeatDescriptor]]],
    tile_size: Tuple[int, int] = (320, 240),
    cols: int = 4,
    show_matches: bool = True,
) -> np.ndarray:
    """Create a grid image showing keyframes with descriptor overlays and matches between adjacent keyframes."""
    if not keyframes:
        return np.zeros((tile_size[1], tile_size[0], 3), dtype=np.uint8)

    tile_w, tile_h = tile_size
    rows = ceil(len(keyframes) / cols)
    grid = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    # Place keyframe images in grid
    for idx, (kf_index, image, _) in enumerate(keyframes):
        r = idx // cols
        c = idx % cols
        
        # Get original image dimensions
        orig_h, orig_w = image.shape[:2]
        scale_x = tile_w / orig_w
        scale_y = tile_h / orig_h
        
        resized = cv2.resize(image, (tile_w, tile_h))
        cv2.putText(
            resized,
            f"KF {kf_index}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )
        grid[r * tile_h : (r + 1) * tile_h, c * tile_w : (c + 1) * tile_w] = resized

    # Draw matches between adjacent keyframes
    if show_matches and len(keyframes) >= 2:
        for idx in range(len(keyframes) - 1):
            _, _, query_descriptors = keyframes[idx]
            _, _, train_descriptors = keyframes[idx + 1]
            
            if not query_descriptors or not train_descriptors:
                continue
            
            # Compute matches between consecutive keyframes
            try:
                matches = arbit.ArbitEngine.match_descriptors(
                    query_descriptors,
                    train_descriptors,
                    cross_check=True,
                    max_distance=60,
                    max_matches=50,  # Limit for visualization clarity
                )
            except Exception as e:
                print(f"   Warning: Failed to match keyframes {idx} and {idx+1}: {e}")
                continue
            
            # Draw match lines
            query_idx = idx
            train_idx = idx + 1
            
            query_r = query_idx // cols
            query_c = query_idx % cols
            train_r = train_idx // cols
            train_c = train_idx % cols
            
            # Get original image dimensions for scaling
            _, query_img, _ = keyframes[query_idx]
            _, train_img, _ = keyframes[train_idx]
            query_orig_h, query_orig_w = query_img.shape[:2]
            train_orig_h, train_orig_w = train_img.shape[:2]
            query_scale_x = tile_w / query_orig_w
            query_scale_y = tile_h / query_orig_h
            train_scale_x = tile_w / train_orig_w
            train_scale_y = tile_h / train_orig_h
            
            for match in matches:
                # Get positions and scale to tile coordinates
                query_pos = match.query_position
                train_pos = match.train_position
                
                # Scale positions
                query_x = int(query_pos[0] * query_scale_x + query_c * tile_w)
                query_y = int(query_pos[1] * query_scale_y + query_r * tile_h)
                train_x = int(train_pos[0] * train_scale_x + train_c * tile_w)
                train_y = int(train_pos[1] * train_scale_y + train_r * tile_h)
                
                # Color by distance (green = good, red = bad)
                normalized_dist = min(match.distance / 80.0, 1.0)
                color = (
                    int(255 * normalized_dist),      # Blue
                    int(255 * (1 - normalized_dist)), # Green
                    0,                                 # Red
                )
                
                # Draw line connecting matched features
                cv2.line(grid, (query_x, query_y), (train_x, train_y), color, 1, lineType=cv2.LINE_AA)
                
                # Draw small circles at endpoints
                cv2.circle(grid, (query_x, query_y), 2, color, -1)
                cv2.circle(grid, (train_x, train_y), 2, color, -1)

    return grid

# Flow visualization functions moved to flow_visualizations.py


def draw_stats_panel(frame, stats, fps=None):
    """Draw statistics panel similar to the Swift app"""
    height, width = frame.shape[:2]
    
    # Create semi-transparent overlay for stats
    overlay = frame.copy()
    panel_height = 180
    cv2.rectangle(overlay, (10, 10), (350, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Text configuration
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 25
    
    y_offset = 35
    
    # Title
    cv2.putText(frame, "Arbit SLAM", (20, y_offset), font, 0.7, (255, 255, 255), 2)
    y_offset += line_height
    
    # FPS
    if fps is not None:
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset), font, font_scale, (0, 255, 0), thickness)
    y_offset += line_height
    
    # Tracked points
    cv2.putText(frame, f"Tracks: {stats['tracks']}", (20, y_offset), font, font_scale, (255, 255, 255), thickness)
    y_offset += line_height
    
    # Map statistics
    cv2.putText(frame, f"Keyframes: {stats['keyframes']}", (20, y_offset), font, font_scale, (255, 255, 255), thickness)
    y_offset += line_height
    
    cv2.putText(frame, f"Landmarks: {stats['landmarks']}", (20, y_offset), font, font_scale, (255, 255, 255), thickness)
    y_offset += line_height
    
    cv2.putText(frame, f"Anchors: {stats['anchors']}", (20, y_offset), font, font_scale, (255, 255, 255), thickness)
    
    return frame


def format_tracked_points(engine):
    """
    Convert engine's tracked points to a format suitable for visualization.
    Returns list of dicts with 'initial', 'refined', 'status', 'residual'
    """
    # Get tracked points from engine (up to 500)
    max_points = 500
    tracked = []
    
    # Import the ArbitTrackedPoint type and TrackStatus enum from arbit.types
    from arbit.types import ArbitTrackedPoint
    
    points_array = (ArbitTrackedPoint * max_points)()
    count = engine._lib.arbit_get_tracked_points(engine._handle, points_array, max_points)
    
    for i in range(count):
        pt = points_array[i]
        tracked.append({
            'initial': (pt.initial_x, pt.initial_y),
            'refined': (pt.refined_x, pt.refined_y),
            'residual': pt.residual,
            'iterations': pt.iterations,
            'status': arbit.TrackStatus(pt.status),  # Returns TrackStatus enum
            'track_id': pt.track_id,
        })
    
    return tracked


print("\n🚀 Arbit Video Processing with Visualization")
print("=" * 60)

# Initialize logging
print("\n0. Initializing logging...")
arbit.init_logging(verbose=False)
print("   ✓ Logging initialized")

# Initialize the engine
print("\n1. Initializing Arbit engine...")
engine = arbit.ArbitEngine()
print("   ✓ Engine initialized")

# Open video file
video_path = "5.MOV"
print(f"\n2. Loading video: {video_path}...")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"   ✗ Error: Could not open video file {video_path}")
    exit(1)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Detect video orientation and rotation needed
# For portrait videos (height > width), we may need to rotate
is_portrait = width > height
needs_rotation = True

# Read a test frame to check actual dimensions vs expected
ret_test, frame_test = cap.read()
if ret_test:
    actual_height, actual_width = frame_test.shape[:2]
    # If video metadata says landscape but frame is portrait, need rotation
    if width > height and actual_height > actual_width:
        needs_rotation = True
        width, height = height, width  # Swap dimensions
        print(f"   • Detected portrait video with rotation metadata")
    # Reset video to beginning
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print(f"   ✓ Video loaded successfully")
print(f"   • Resolution: {width}x{height} {'(portrait)' if height > width else '(landscape)'}")
print(f"   • FPS: {fps:.2f}")
print(f"   • Total frames: {frame_count}")
if needs_rotation:
    print(f"   • Will apply 90° rotation to correct orientation")

# Scale factor for processing (reduce resolution for better performance)
process_scale = 0.5
process_width = int(width * process_scale)
process_height = int(height * process_scale)

# Estimate camera intrinsics (typical iPhone camera) - scaled to match processing resolution
focal_length = process_width * 1.2  # Rough estimate
fx = fy = focal_length
cx = process_width / 2.0
cy = process_height / 2.0

print("\n3. Pre-loading and converting frames...")
max_frames = 1000000000  # Process first 100 frames
frames_bgra = []
frames_bgr = []

print(f"   • Loading {max_frames} frames...")
print(f"   • Downscaling to {process_width}x{process_height} ({process_scale*100:.0f}%)")
for i in range(max_frames):
    ret, frame_bgr = cap.read()
    if not ret:
        max_frames = i  # Adjust if video has fewer frames
        break
    
    # Apply rotation if needed (for portrait videos with rotation metadata)
    if needs_rotation:
        frame_bgr = cv2.rotate(frame_bgr, cv2.ROTATE_90_CLOCKWISE)
    
    # Downscale frame for processing
    # frame_bgr_scaled = cv2.resize(frame_bgr, (process_width, process_height))
    
    # Convert BGR to BGRA (add alpha channel) - done once outside processing loop
    frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    frames_bgra.append(frame_bgra)
    frames_bgr.append(frame_bgr)  # Keep BGR for visualization

print(f"   ✓ Pre-loaded {len(frames_bgra)} frames at {process_width}x{process_height}")

print("\n4. Processing video frames with visualization...")
print(f"   • Using estimated intrinsics: fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
print(f"   • Press 'q' to quit, 'SPACE' to pause/resume")
print(f"   • Press '1': Arrows, '2': HSV (standard), '3': Heatmap, '4': Circles+Arrows")
print(f"   • Keyframe grid shows feature matches between consecutive keyframes")

# Visualization mode
viz_mode = 4  # 1=arrows, 2=hsv, 3=heatmap, 4=circles+arrows
color_wheel_legend = create_color_wheel_legend(150)

processed_frames = 0
paused = False
processing_fps = 0.0
last_time = cv2.getTickCount()

keyframe_overlays: List[Tuple[int, np.ndarray, List[arbit.FeatDescriptor]]] = []
MAX_KEYFRAMES_DISPLAY = 12
keyframe_tile = (320, 240)
keyframe_cols = 4

# Create window
cv2.namedWindow('Arbit SLAM Visualization', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Arbit SLAM Visualization', process_width, process_height)
cv2.namedWindow('Keyframe Grid', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Keyframe Grid', keyframe_tile[0] * keyframe_cols, keyframe_tile[1] * 2)

while processed_frames < len(frames_bgra):
    if not paused:
        # sleep(1)
        frame_bgra = frames_bgra[processed_frames]
        frame_bgr = frames_bgr[processed_frames]
        
        # Calculate timestamp based on frame number
        timestamp = processed_frames / fps
        
        # Create CameraFrame for Arbit (using pre-converted BGRA frame)
        camera_frame = arbit.CameraFrame(
            timestamp=timestamp,
            image=frame_bgra,
            intrinsics=(fx, fy, cx, cy),
            pixel_format=arbit.PixelFormat.BGRA8
        )
        
        # Ingest frame into Arbit engine
        success = engine.ingest_frame(camera_frame)
        
        if success:
            processed_frames += 1
            
            # Get tracked points for visualization
            tracked_points = format_tracked_points(engine)
            
            # Get engine state
            state = engine.get_frame_state()
            
            # Calculate processing FPS
            current_time = cv2.getTickCount()
            time_elapsed = (current_time - last_time) / cv2.getTickFrequency()
            if time_elapsed > 0:
                processing_fps = 1.0 / time_elapsed
            last_time = current_time
            
            # Use frame directly for visualization (already at processing resolution)
            vis_frame = frame_bgr.copy()
            
            # Draw tracked points overlay based on mode
            vis_frame = draw_flow_arrows(vis_frame, tracked_points, arrow_style='detailed')
            # if viz_mode == 1:
            #     vis_frame = draw_flow_arrows(vis_frame, tracked_points, arrow_style='simple')
            # elif viz_mode == 2:
            #     vis_frame = draw_flow_dense_hsv(vis_frame, tracked_points, grid_size=32)
            #     # Add color wheel legend in corner
            #     legend_h, legend_w = color_wheel_legend.shape[:2]
            #     vis_frame[10:10+legend_h, process_width-legend_w-10:process_width-10] = color_wheel_legend
            # elif viz_mode == 3:
            #     vis_frame = draw_flow_heatmap(vis_frame, tracked_points)
            # else:  # viz_mode == 4
            #     vis_frame = draw_flow_circles_with_arrows(vis_frame, tracked_points)
            
            # Draw stats panel
            stats = {
                'tracks': state.track_count,
                'keyframes': state.keyframe_count,
                'landmarks': state.landmark_count,
                'anchors': state.anchor_count,
            }
            vis_frame = draw_stats_panel(vis_frame, stats, processing_fps)

            # Check if current frame is a keyframe (every 60 frames, matching engine logic)
            is_keyframe = processed_frames % 10 == 0
            if is_keyframe:
                descriptors = engine.get_descriptors(2048)
                if descriptors:  # Only add if descriptors are available
                    overlay = draw_keypoint_descriptors(frame_bgr, descriptors)
                    # Store keyframe with image, overlay, and descriptors
                    keyframe_overlays.append((processed_frames, overlay, descriptors))
                    if len(keyframe_overlays) > MAX_KEYFRAMES_DISPLAY:
                        keyframe_overlays.pop(0)
            
            # Add frame counter
            cv2.putText(vis_frame, f"Frame: {processed_frames}/{max_frames}", 
                       (process_width - 200, process_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    else:
        # If paused, just use the last frame
        pass
    
    # Display the frame
    cv2.imshow('Arbit SLAM Visualization', vis_frame)

    if keyframe_overlays:
        grid_image = build_keyframe_grid(keyframe_overlays, keyframe_tile, keyframe_cols)
        cv2.imshow('Keyframe Grid', grid_image)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\n   ⚠ User requested quit")
        break
    elif key == ord(' '):
        paused = not paused
        print(f"\n   {'⏸ Paused' if paused else '▶ Resumed'}")
    elif key == ord('1'):
        viz_mode = 1
        print("\n   🎨 Switched to: Arrow visualization")
    elif key == ord('2'):
        viz_mode = 2
        print("\n   🎨 Switched to: HSV color wheel (CV standard)")
    elif key == ord('3'):
        viz_mode = 3
        print("\n   🎨 Switched to: Magnitude heatmap")
    elif key == ord('4'):
        viz_mode = 4
        print("\n   🎨 Switched to: Circles with arrows (SLAM papers)")

sleep(100000)
cap.release()
cv2.destroyAllWindows()

print(f"\n   ✓ Processed {processed_frames} frames")

# Query final engine state
print("\n5. Final engine state...")
state = engine.get_frame_state()
print(f"   • Tracked points: {state.track_count}")
print(f"   • Keyframes: {state.keyframe_count}")
print(f"   • Landmarks: {state.landmark_count}")
print(f"   • Anchors: {state.anchor_count}")

# Get trajectory
print("\n6. Getting trajectory...")
trajectory = engine.get_trajectory()
print(f"   • Trajectory points: {len(trajectory)}")
if len(trajectory) > 0:
    print(f"   • Start position: {trajectory[0]}")
    print(f"   • End position: {trajectory[-1]}")
    # Calculate total distance traveled
    distances = [np.linalg.norm(trajectory[i+1] - trajectory[i]) 
                 for i in range(len(trajectory)-1)]
    total_distance = sum(distances)
    print(f"   • Total distance: {total_distance:.3f} meters")

# Save map
print("\n7. Saving map...")
map_data = engine.save_map()
print(f"   • Map size: {len(map_data)} bytes")

print("\n" + "=" * 60)
print("✅ Video processing completed successfully!")
print("\nResults:")
print(f"  - Processed {processed_frames} frames from real video")
print(f"  - Built map with {state.landmark_count} landmarks")
print(f"  - Tracked trajectory with {len(trajectory)} points")
print(f"  - Created {state.keyframe_count} keyframes")
