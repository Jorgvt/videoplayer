# /// script
# dependencies = [
#   "opencv-python",
#   "numpy",
# ]
# ///

import os
import re
import time
import cv2
import numpy as np
from pathlib import Path

# Paths
DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
SCENES = ["marbles", "pink_room", "subway", "zeroday"]

class DatasetPlayer:
    def __init__(self):
        self.scene_idx = 0
        self.metric_idx = 0
        self.level_idx = 1  # level1 default
        
        self.videos = {}  # scene -> list of video dicts
        self.scan_dataset()
        
        self.frames_a = []
        self.frames_b = []
        self.meta_a = None
        self.meta_b = None
        
        self.frame_idx = 0
        self.playing = False
        self.layout = "side-by-side" # "side-by-side" or "overlay" or "single-a" or "single-b"
        self.show_b_in_overlay = False
        
        # Playback speed controls (speed is scale factor of FPS, e.g., 0.25 means 60 FPS)
        self.speeds = [0.1, 0.25, 0.5, 1.0, 2.0]
        self.speed_idx = 3 # 1.0x by default. Note: 240 FPS real-time can be heavy, 0.25x matches 60Hz.
        
        # Interactive Zoom / Pan state (synced)
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Dragging state
        self.drag_start = None
        self.drag_start_pan = None
        
        # FPS calculation state
        self.fps_history = []
        self.current_fps = 0.0
        
        self.load_active_videos()
        
    def scan_dataset(self):
        for scene in SCENES:
            self.videos[scene] = []
            
        if not DATASET_DIR.exists():
            print(f"Dataset directory not found: {DATASET_DIR}")
            return
            
        for file in sorted(os.listdir(DATASET_DIR)):
            if not file.endswith(".mp4"):
                continue
                
            # Find scene
            scene = None
            for s in SCENES:
                if file.startswith(s + "_"):
                    scene = s
                    break
            if not scene:
                continue
                
            # Parse rest
            rest = file[len(scene) + 1:-4]
            if rest == "reference":
                self.videos[scene].append({
                    "filename": file,
                    "path": str(DATASET_DIR / file),
                    "metric": "reference",
                    "level": None,
                    "label": "Reference"
                })
            else:
                match = re.match(r"^(.*?)(?:_level([0-2]))?$", rest)
                if match:
                    metric = match.group(1)
                    lvl_num = match.group(2)
                    level = f"level{lvl_num}" if lvl_num is not None else None
                    metric_label = metric.replace("_", " ").replace("-", " ").title()
                    label = f"{metric_label} ({level.capitalize()})" if level else metric_label
                    
                    self.videos[scene].append({
                        "filename": file,
                        "path": str(DATASET_DIR / file),
                        "metric": metric,
                        "level": level,
                        "label": label
                    })

    def get_current_video_list(self):
        scene = SCENES[self.scene_idx]
        return self.videos[scene]

    def load_active_videos(self):
        vids = self.get_current_video_list()
        if not vids:
            return
            
        # Video A is always reference if available
        vid_a = next((v for v in vids if v["metric"] == "reference"), vids[0])
        
        # Video B is current metric + level
        non_ref = [v for v in vids if v["metric"] != "reference"]
        if not non_ref:
            vid_b = vid_a
        else:
            # Group metrics
            metrics = list(dict.fromkeys([v["metric"] for v in non_ref]))
            self.metric_idx = self.metric_idx % len(metrics)
            active_metric = metrics[self.metric_idx]
            
            # Find matching level
            level_str = f"level{self.level_idx}"
            vid_b = next((v for v in non_ref if v["metric"] == active_metric and v["level"] == level_str), None)
            if not vid_b:
                # Fallback to any level of this metric
                vid_b = next((v for v in non_ref if v["metric"] == active_metric), non_ref[0])
                
        self.meta_a = vid_a
        self.meta_b = vid_b
        
        print(f"\n--- Loading Video Pair ---")
        print(f"Scene: {SCENES[self.scene_idx].upper()}")
        print(f"Video A (Base): {vid_a['filename']}")
        print(f"Video B (Comp): {vid_b['filename']}")
        
        self.frames_a = self.preload_video(vid_a["path"])
        self.frames_b = self.preload_video(vid_b["path"])
        
        # Reset frames
        self.frame_idx = 0
        
    def preload_video(self, path):
        print(f"Preloading {Path(path).name} into RAM...", end="", flush=True)
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f" Loaded {len(frames)} frames.")
        return frames

    def apply_zoom_pan(self, img):
        if self.zoom_scale == 1.0:
            return img
            
        h, w = img.shape[:2]
        
        # Calculate crop size
        crop_w = int(w / self.zoom_scale)
        crop_h = int(h / self.zoom_scale)
        
        # Center of the crop window
        center_x = w // 2 + self.pan_x
        center_y = h // 2 + self.pan_y
        
        # Bounds check
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        # Adjust if bounds hit edges
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
            
        cropped = img[y1:y2, x1:x2]
        # Resize using Nearest Neighbor to keep pixel edges sharp for inspection
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized

    def get_display_frame(self):
        if not self.frames_a or not self.frames_b:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Update FPS counter
        now = time.perf_counter()
        self.fps_history.append(now)
        self.fps_history = [t for t in self.fps_history if now - t <= 1.0]
        if len(self.fps_history) > 1:
            self.current_fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
        else:
            self.current_fps = 0.0
            
        # Get frames
        idx_a = self.frame_idx % len(self.frames_a)
        idx_b = self.frame_idx % len(self.frames_b)
        
        frame_a = self.frames_a[idx_a]
        frame_b = self.frames_b[idx_b]
        
        # Apply Zoom & Pan
        frame_a_disp = self.apply_zoom_pan(frame_a)
        frame_b_disp = self.apply_zoom_pan(frame_b)
        
        h, w = frame_a.shape[:2]
        
        # Combine according to layout
        if self.layout == "side-by-side":
            combined = np.hstack((frame_a_disp, frame_b_disp))
            display_w = w * 2
        elif self.layout == "overlay":
            combined = frame_b_disp.copy() if self.show_b_in_overlay else frame_a_disp.copy()
            display_w = w
        elif self.layout == "single-a":
            combined = frame_a_disp.copy()
            display_w = w
        else: # single-b
            combined = frame_b_disp.copy()
            display_w = w
            
        # Draw UI overlay
        self.draw_ui(combined, h, display_w)
        return combined

    def draw_ui(self, img, h, w):
        # Semi-transparent dark bar at the top
        cv2.rectangle(img, (0, 0), (w, 40), (15, 15, 20), -1)
        
        # Details text
        scene_name = SCENES[self.scene_idx].upper()
        speed = self.speeds[self.speed_idx]
        
        status = "PLAYING" if self.playing else "PAUSED"
        text = f"[{status}] Scene: {scene_name} | Speed: {speed}x | Layout: {self.layout.upper()} | FPS: {self.current_fps:.1f}"
        cv2.putText(img, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
        
        # Video source labels
        v_a_lbl = f"A: {self.meta_a['label']}"
        v_b_lbl = f"B: {self.meta_b['label']}"
        
        if self.layout == "side-by-side":
            cv2.putText(img, v_a_lbl, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(img, v_b_lbl, (w // 2 + 15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_AA)
        else:
            active_label = v_b_lbl if (self.layout == "single-b" or (self.layout == "overlay" and self.show_b_in_overlay)) else v_a_lbl
            cv2.putText(img, active_label, (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 120), 1, cv2.LINE_AA)
            
        # Frame counter
        total_frames = max(len(self.frames_a), len(self.frames_b))
        frame_text = f"Frame: {self.frame_idx + 1}/{total_frames} (Zoom: {int(self.zoom_scale*100)}%)"
        cv2.putText(img, frame_text, (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_start_pan = (self.pan_x, self.pan_y)
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                # Pan moves in opposite direction of drag to behave naturally
                self.pan_x = self.drag_start_pan[0] - int(dx / self.zoom_scale)
                self.pan_y = self.drag_start_pan[1] - int(dy / self.zoom_scale)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.drag_start_pan = None
            
        elif event == cv2.EVENT_MOUSEWHEEL:
            # cv2 wheel delta is typically +/- 120 (windows) or +/- 1 (linux)
            delta = flags
            if delta > 0:
                self.zoom_scale = min(20.0, self.zoom_scale * 1.15)
            else:
                self.zoom_scale = max(1.0, self.zoom_scale / 1.15)
                if self.zoom_scale == 1.0:
                    self.pan_x = 0
                    self.pan_y = 0

    def run(self):
        window_name = "GAIM240 Video Quality Comparer (RAM Playback)"
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(window_name, self.on_mouse)
        
        print("\n=== Native Player Started ===")
        print("Controls:")
        print("  [Space]       : Play / Pause")
        print("  [Left/Right]  : Frame step")
        print("  [1/2/3/4]     : Switch Scene (Marbles, Pink Room, Subway, Zeroday)")
        print("  [Tab]         : Press/Release to Swap A/B (Overlay mode)")
        print("  [L]           : Toggle Layout (Side-by-Side, Overlay, Single A, Single B)")
        print("  [M] / [N]     : Switch Comparison Metric (Next / Previous)")
        print("  [0/1/2]       : Switch Artifact Level (0, 1, 2)")
        print("  [-] / [+]     : Playback speed (0.1x, 0.25x, 0.5x, 1x, 2x)")
        print("  [R]           : Reset Zoom & Pan")
        print("  [Q] or [Esc]  : Quit")
        
        while True:
            frame = self.get_display_frame()
            cv2.imshow(window_name, frame)
            
            # Base video is 240 FPS.
            # Real-time base frame time = 4.16 ms
            # Delay in ms depends on selected speed scale
            speed_mult = self.speeds[self.speed_idx]
            base_fps = 240.0
            delay_ms = int(1000.0 / (base_fps * speed_mult))
            
            # Bound delay
            delay_ms = max(1, delay_ms)
            
            key = cv2.waitKey(delay_ms if self.playing else 30) & 0xFF
            
            if key == ord('q') or key == 27: # Q or Esc
                break
            elif key == ord(' '):
                self.playing = not self.playing
            elif key == 83 or key == 3: # Right Arrow (Windows/Linux cv2 mappings) or ord('d')
                self.frame_idx += 1
                self.playing = False
            elif key == 81 or key == 2: # Left Arrow or ord('a')
                self.frame_idx = max(0, self.frame_idx - 1)
                self.playing = False
            elif key == ord('r'):
                self.zoom_scale = 1.0
                self.pan_x = 0
                self.pan_y = 0
            elif key == ord('l'):
                layouts = ["side-by-side", "overlay", "single-a", "single-b"]
                curr_idx = layouts.index(self.layout)
                self.layout = layouts[(curr_idx + 1) % len(layouts)]
            elif key == 9: # Tab
                # Toggle swap state in overlay
                self.show_b_in_overlay = not self.show_b_in_overlay
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.scene_idx = key - ord('1')
                self.load_active_videos()
            elif key == ord('0'):
                self.level_idx = 0
                self.load_active_videos()
            elif key == ord('1'):
                self.level_idx = 1
                self.load_active_videos()
            elif key == ord('2'):
                self.level_idx = 2
                self.load_active_videos()
            elif key == ord('='): # Plus
                self.speed_idx = min(len(self.speeds) - 1, self.speed_idx + 1)
            elif key == ord('-'):
                self.speed_idx = max(0, self.speed_idx - 1)
            elif key == ord('m'): # Next metric
                self.metric_idx += 1
                self.load_active_videos()
            elif key == ord('n'): # Prev metric
                self.metric_idx = max(0, self.metric_idx - 1)
                self.load_active_videos()
                
            # If playing, increment frame
            if self.playing:
                self.frame_idx += 1
                total_frames = max(len(self.frames_a), len(self.frames_b))
                if self.frame_idx >= total_frames:
                    self.frame_idx = 0 # loop
                    
        cv2.destroyAllWindows()

if __name__ == "__main__":
    player = DatasetPlayer()
    player.run()
