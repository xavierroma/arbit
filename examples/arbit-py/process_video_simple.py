"""
Simple Arbit visualization with feature tracking overlay
"""
import numpy as np
import cv2
import arbit
import time
from flow_visualizations import draw_flow_circles_with_arrows

print("\n🚀 Arbit Simple Visualization")
print("=" * 60)

# Initialize logging
arbit.init_logging(verbose=False)

# Initialize engine
engine = arbit.ArbitEngine()

# Open video
video_path = "1.MOV"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"❌ Could not open video: {video_path}")
    exit(1)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"📹 Video: {width}x{height} @ {fps:.1f}fps")

# Camera intrinsics
focal_length = max(width, height) * 0.8
fx = fy = focal_length
cx = width / 2.0
cy = height / 2.0

print(f"📐 Intrinsics: fx={fx:.0f}, fy={fy:.0f}")

# Display settings
DISPLAY_SCALE = 0.5
display_width = int(width * DISPLAY_SCALE)
display_height = int(height * DISPLAY_SCALE)

print(f"🖥️ Display: {display_width}x{display_height}")
print(f"⚡ Controls: 'q'=quit, 'SPACE'=pause")

# Create window
cv2.namedWindow('Arbit Simple', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Arbit Simple', display_width, display_height)

# Simple stats
frame_count = 0
processed_count = 0
last_time = time.time()

print(f"\n▶️ Starting simple processing...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"\n🏁 End of video")
            break
        
        frame_count += 1
        
        tracked_points = []
        track_count = 0
        state_name = "Initializing"
        
        # Convert to BGRA for SLAM (every 3rd frame to keep it simple)
        if frame_count % 3 == 0:
            frame_bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            
            # Create camera frame
            camera_frame = arbit.CameraFrame(
                timestamp=frame_count / fps,
                image=frame_bgra,
                intrinsics=(fx, fy, cx, cy),
                pixel_format=arbit.PixelFormat.BGRA8
            )
            
            # Process with SLAM
            try:
                success = engine.ingest_frame(camera_frame)
                if success:
                    processed_count += 1
                    
                    # Get state
                    state = engine.get_frame_state()
                    track_count = state.track_count
                    state_name = state.state_name
                    
                    # Get tracked points for visualization
                    from arbit.types import ArbitTrackedPoint
                    max_points = 200
                    tracked_points_ffi = (ArbitTrackedPoint * max_points)()
                    num_tracked = engine.get_frame_state(tracked_points_ffi, max_points)
                    
                    for i in range(num_tracked):
                        pt = tracked_points_ffi[i]
                        tracked_points.append({
                            'initial': (pt.initial_x, pt.initial_y),
                            'refined': (pt.refined_x, pt.refined_y),
                            'residual': pt.residual,
                            'iterations': pt.iterations,
                            'status': pt.status,
                        })
                    
            except Exception as e:
                print(f"SLAM error: {e}")
        
        # Always display the frame
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Scale tracked points for display
        if tracked_points:
            scaled_points = []
            for pt in tracked_points:
                scaled_points.append({
                    'initial': (pt['initial'][0] * DISPLAY_SCALE, pt['initial'][1] * DISPLAY_SCALE),
                    'refined': (pt['refined'][0] * DISPLAY_SCALE, pt['refined'][1] * DISPLAY_SCALE),
                    'residual': pt['residual'],
                    'iterations': pt['iterations'],
                    'status': pt['status'],
                })
            
            # Draw feature visualization
            display_frame = draw_flow_circles_with_arrows(display_frame, scaled_points)
        
        # Add text overlay
        cv2.putText(display_frame, f"Frame: {frame_count} | Tracks: {track_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"State: {state_name}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display
        cv2.imshow('Arbit Simple', display_frame)
        
        # Handle input
        key = cv2.waitKey(30) & 0xFF  # 30ms delay for ~30 FPS
        if key == ord('q'):
            break
        
        # Stats every 5 seconds
        if frame_count % 150 == 0:
            current_time = time.time()
            elapsed = current_time - last_time
            display_fps = 150 / elapsed
            print(f"📊 {display_fps:.1f} FPS, processed {processed_count}/{frame_count}")
            last_time = current_time

finally:
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Completed:")
    print(f"   • Frames: {frame_count}")
    print(f"   • Processed: {processed_count}")
