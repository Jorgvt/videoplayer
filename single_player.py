# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
# ]
# ///

import os
import sys
import time
import cv2
from pathlib import Path

# Config
DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
DEFAULT_VIDEO = "marbles_reference.mp4"

def main():
    # Allow choosing video via command line
    video_filename = DEFAULT_VIDEO
    if len(sys.argv) > 1:
        video_filename = sys.argv[1]
        
    video_path = DATASET_DIR / video_filename
    if not video_path.exists():
        # Search directory
        print(f"Error: {video_path} does not exist.")
        if DATASET_DIR.exists():
            print("\nAvailable videos in GAIM240:")
            for f in sorted(os.listdir(DATASET_DIR)):
                if f.endswith(".mp4"):
                    print(f"  {f}")
        return

    print(f"\n==========================================")
    print(f"Loading single video: {video_filename}")
    print(f"==========================================")
    
    # Preload video frames into RAM
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Resolution: {int(width)}x{int(height)} | File FPS: {fps}")
    print("Preloading frames into RAM...", end="", flush=True)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    num_frames = len(frames)
    print(f" Loaded {num_frames} frames.")
    
    # Playback parameters
    target_fps = 240.0
    frame_time = 1.0 / target_fps
    
    window_name = "GAIM240 Single Player (RAM)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    
    # State variables
    frame_idx = 0
    playing = True
    fps_history = []
    current_fps = 0.0
    
    print("\nControls:")
    print("  [Space]       : Play / Pause")
    print("  [Left/Right]  : Step 1 frame")
    print("  [Q] or [Esc]  : Quit")
    print("\nPlaying at target 240 FPS...")
    
    last_render_time = time.perf_counter()
    play_start_time = time.perf_counter()
    play_start_frame_idx = 0
    
    while True:
        # Calculate current frame index (time-based progression)
        if playing:
            now = time.perf_counter()
            elapsed = now - play_start_time
            frame_idx = int(play_start_frame_idx + elapsed * target_fps)
            if frame_idx >= num_frames:
                # Loop around
                play_start_time = now
                play_start_frame_idx = 0
                frame_idx = 0
                
        # Draw frame
        active_frame = frames[frame_idx % num_frames]
        cv2.imshow(window_name, active_frame)
        
        # Calculate FPS
        now = time.perf_counter()
        fps_history.append(now)
        fps_history = [t for t in fps_history if now - t <= 1.0]
        if len(fps_history) > 1:
            current_fps = len(fps_history) / (fps_history[-1] - fps_history[0])
            
        # Update window title with FPS (extremely cheap compared to putText)
        cv2.setWindowTitle(window_name, f"Single Player - {video_filename} | {current_fps:.1f} FPS")
        
        # Event Polling: waitKey(1) every 4 frames (60Hz) to save OS overhead
        if frame_idx % 4 == 0:
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255
            
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            playing = not playing
            play_start_time = time.perf_counter()
            play_start_frame_idx = frame_idx
        elif key == 83 or key == 3 or key == ord('d'): # Right Arrow or d
            playing = False
            frame_idx = (frame_idx + 1) % num_frames
            play_start_time = time.perf_counter()
            play_start_frame_idx = frame_idx
        elif key == 81 or key == 2 or key == ord('a'): # Left Arrow or a
            playing = False
            frame_idx = (frame_idx - 1) % num_frames
            play_start_time = time.perf_counter()
            play_start_frame_idx = frame_idx
            
        # Performance output in console
        if playing and frame_idx % 240 == 0:
            print(f"  [Console Status] Playing at {current_fps:.1f} FPS")
            
        # Precise playback pacing
        if playing:
            now = time.perf_counter()
            time_elapsed = now - last_render_time
            time_remaining = frame_time - time_elapsed
            if time_remaining > 0:
                time.sleep(time_remaining)
        else:
            time.sleep(0.03)
            
        last_render_time = time.perf_counter()
        
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
