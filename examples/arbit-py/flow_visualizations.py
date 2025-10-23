"""
Different optical flow visualization techniques for comparison
"""
import numpy as np
import cv2
import arbit


def flow_to_hsv(flow_x, flow_y, max_magnitude=None):
    """
    Convert optical flow to HSV color wheel visualization (Middlebury standard).
    
    This is the most standard way to visualize optical flow in computer vision:
    - Hue (color) represents direction of motion
    - Saturation/Value represents magnitude of motion
    
    Args:
        flow_x: Horizontal flow component (array)
        flow_y: Vertical flow component (array)
        max_magnitude: Maximum magnitude for normalization (auto-computed if None)
    
    Returns:
        BGR image with color-coded flow
    """
    # Calculate magnitude and angle
    magnitude = np.sqrt(flow_x**2 + flow_y**2)
    angle = np.arctan2(flow_y, flow_x)
    
    # Normalize magnitude
    if max_magnitude is None:
        max_magnitude = np.max(magnitude)
    if max_magnitude == 0:
        max_magnitude = 1.0
    
    # Create HSV image
    hsv = np.zeros((magnitude.shape[0], magnitude.shape[1], 3), dtype=np.uint8)
    
    # Hue (0-180 in OpenCV) represents direction
    hsv[..., 0] = ((angle + np.pi) / (2 * np.pi) * 180).astype(np.uint8)
    
    # Saturation is full
    hsv[..., 1] = 255
    
    # Value represents magnitude
    hsv[..., 2] = np.clip(magnitude / max_magnitude * 255, 0, 255).astype(np.uint8)
    
    # Convert to BGR for display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return bgr


def draw_flow_arrows(frame, tracked_points, max_points=200, arrow_style='simple'):
    """
    Arrow-based visualization (more intuitive, matches Swift app).
    
    Args:
        frame: BGR image to draw on
        tracked_points: List of tracked points
        max_points: Maximum number of arrows to draw
        arrow_style: 'simple' or 'detailed'
    """
    overlay = frame.copy()
    
    for point in tracked_points[:max_points]:
        if point['status'] != arbit.TrackStatus.CONVERGED:  # Only show converged
            continue
            
        x0, y0 = point['initial']
        x1, y1 = point['refined']
        
        dx = x1 - x0
        dy = y1 - y0
        magnitude = np.sqrt(dx * dx + dy * dy)
        
        # Color based on magnitude
        if magnitude < 5.0:
            color = (0, 255, 0)  # Green
        elif magnitude < 15.0:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
        
        pt0 = (int(x0), int(y0))
        pt1 = (int(x1), int(y1))
        
        if arrow_style == 'simple':
            # Simple arrow
            if magnitude > 1.0:
                cv2.arrowedLine(overlay, pt0, pt1, color, 2, tipLength=0.3)
                cv2.circle(overlay, pt1, 2, (255, 255, 255), -1)
            else:
                cv2.circle(overlay, pt1, 4, color, -1)
        else:
            # Detailed with initial point
            cv2.circle(overlay, pt0, 3, color, -1)
            if magnitude > 1.0:
                cv2.arrowedLine(overlay, pt0, pt1, color, 2, tipLength=0.3)
                cv2.circle(overlay, pt1, 2, (255, 255, 255), -1)

        track_id = point.get('track_id')
        if track_id:
            label = f"#{int(track_id)}"
            label_pos = (int(x1) + 5, int(y1) - 5)
            cv2.putText(
                overlay,
                label,
                label_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
    
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_flow_dense_hsv(frame, tracked_points, grid_size=32):
    """
    Dense HSV flow visualization with color wheel.
    
    Creates a dense flow field by interpolating sparse tracked points,
    then visualizes using the standard color wheel.
    
    Args:
        frame: BGR image
        tracked_points: List of tracked points
        grid_size: Size of interpolation grid
    """
    if not tracked_points:
        return frame
    
    height, width = frame.shape[:2]
    
    # Create sparse flow field
    flow_x = np.zeros((height, width), dtype=np.float32)
    flow_y = np.zeros((height, width), dtype=np.float32)
    weights = np.zeros((height, width), dtype=np.float32)
    
    # Fill in flow from tracked points (with Gaussian weighting)
    sigma = grid_size / 2.0
    for point in tracked_points:
        if point['status'] != arbit.TrackStatus.CONVERGED:
            continue
            
        x0, y0 = point['initial']
        x1, y1 = point['refined']
        
        dx = x1 - x0
        dy = y1 - y0
        
        # Integer coordinates for grid
        ix, iy = int(x0), int(y0)
        
        # Apply flow to nearby grid cells with Gaussian falloff
        for j in range(max(0, iy - grid_size), min(height, iy + grid_size)):
            for i in range(max(0, ix - grid_size), min(width, ix + grid_size)):
                dist_sq = (i - x0)**2 + (j - y0)**2
                weight = np.exp(-dist_sq / (2 * sigma**2))
                
                flow_x[j, i] += dx * weight
                flow_y[j, i] += dy * weight
                weights[j, i] += weight
    
    # Normalize by weights
    mask = weights > 0.01
    flow_x[mask] /= weights[mask]
    flow_y[mask] /= weights[mask]
    
    # Convert to HSV color wheel
    flow_hsv = flow_to_hsv(flow_x, flow_y, max_magnitude=20.0)
    
    # Blend with original frame
    alpha = 0.6
    result = cv2.addWeighted(flow_hsv, alpha, frame, 1 - alpha, 0)
    
    return result


def draw_flow_heatmap(frame, tracked_points):
    """
    Magnitude heatmap visualization.
    Shows only the speed of motion as a color intensity map.
    """
    if not tracked_points:
        return frame
    
    height, width = frame.shape[:2]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # Create magnitude field
    for point in tracked_points:
        if point['status'] != arbit.TrackStatus.CONVERGED:
            continue
            
        x0, y0 = point['initial']
        x1, y1 = point['refined']
        
        dx = x1 - x0
        dy = y1 - y0
        magnitude = np.sqrt(dx * dx + dy * dy)
        
        # Draw with Gaussian splat
        ix, iy = int(x0), int(y0)
        radius = 20
        sigma = radius / 2.5
        
        for j in range(max(0, iy - radius), min(height, iy + radius)):
            for i in range(max(0, ix - radius), min(width, ix + radius)):
                dist_sq = (i - x0)**2 + (j - y0)**2
                weight = np.exp(-dist_sq / (2 * sigma**2))
                heatmap[j, i] = max(heatmap[j, i], magnitude * weight)
    
    # Normalize and convert to color
    if heatmap.max() > 0:
        heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    else:
        heatmap = heatmap.astype(np.uint8)
    
    # Apply colormap (hot, jet, etc.)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Blend with original
    alpha = 0.5
    result = cv2.addWeighted(heatmap_color, alpha, frame, 1 - alpha, 0)
    
    return result


def draw_flow_circles_with_arrows(frame, tracked_points, max_points=200):
    """
    Circle with arrow visualization - very common in SLAM/visual odometry papers.
    
    This style shows:
    - Circle: Feature region/uncertainty or feature type
    - Arrow: Motion vector from center
    - Colors: Status or magnitude
    
    Args:
        frame: BGR image to draw on
        tracked_points: List of tracked points
        max_points: Maximum number of features to draw
    """
    if not tracked_points:
        return frame
    
    overlay = frame.copy()
    
    for point in tracked_points[:max_points]:
        if point['status'] != arbit.TrackStatus.CONVERGED:  # Only show converged
            continue
            
        x0, y0 = point['initial']
        x1, y1 = point['refined']
        
        dx = x1 - x0
        dy = y1 - y0
        magnitude = np.sqrt(dx * dx + dy * dy)
        
        # Center of circle (initial position)
        center = (int(x0), int(y0))
        
        # Circle size based on magnitude or fixed
        radius = max(8, min(20, int(magnitude * 2)))
        
        # Colors based on magnitude/status
        if magnitude < 5.0:
            circle_color = (0, 255, 0)  # Green - small motion
            arrow_color = (0, 200, 0)
        elif magnitude < 15.0:
            circle_color = (0, 255, 255)  # Yellow - medium motion  
            arrow_color = (0, 200, 200)
        else:
            circle_color = (0, 0, 255)  # Red - large motion
            arrow_color = (0, 0, 200)
        
        # Draw circle (outline + fill with transparency)
        cv2.circle(overlay, center, radius, circle_color, 2)
        cv2.circle(overlay, center, radius-2, circle_color, -1)
        
        # Draw arrow from center to refined position
        if magnitude > 1.0:
            end_point = (int(x1), int(y1))
            cv2.arrowedLine(overlay, center, end_point, arrow_color, 2, tipLength=0.3)
            
            # Small dot at refined position
            cv2.circle(overlay, end_point, 2, (255, 255, 255), -1)
        else:
            # For very small motion, just a brighter center
            cv2.circle(overlay, center, 3, (255, 255, 255), -1)
    
    # Blend overlay with original frame
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame


def create_color_wheel_legend(size=200):
    """
    Create a color wheel legend showing direction encoding.
    This is standard in optical flow papers.
    """
    legend = np.zeros((size, size, 3), dtype=np.uint8)
    center = size // 2
    
    for y in range(size):
        for x in range(size):
            dx = x - center
            dy = y - center
            
            angle = np.arctan2(dy, dx)
            magnitude = np.sqrt(dx**2 + dy**2)
            
            if magnitude > center:
                continue
            
            # HSV encoding
            hue = int(((angle + np.pi) / (2 * np.pi) * 180))
            sat = 255
            val = int(min(magnitude / center * 255, 255))
            
            hsv_pixel = np.uint8([[[hue, sat, val]]])
            bgr_pixel = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)
            legend[y, x] = bgr_pixel[0, 0]
    
    # Add directional labels
    cv2.putText(legend, "→", (center + center//2, center), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(legend, "←", (10, center), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(legend, "↑", (center, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(legend, "↓", (center, size - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return legend
