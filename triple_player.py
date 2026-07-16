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

class TripleDatasetPlayer:
    def __init__(self):
        self.scene_idx = 0
        self.metric_idx_a = 0  # Left video metric index
        self.metric_idx_c = 1  # Right video metric index (default to second metric)
        self.level_idx_a = 1   # Left video distortion level
        self.level_idx_c = 1   # Right video distortion level
        
        self.videos = {}  # scene -> list of video dicts
        self.scan_dataset()
        
        self.frames_a = []
        self.frames_ref = []
        self.frames_c = []
        self.frames_triple = []
        
        self.meta_a = None
        self.meta_ref = None
        self.meta_c = None
        
        self.frame_idx = 0
        self.playing = False
        self.show_ui = True
        
        # Playback speed controls
        self.speeds = [0.1, 0.25, 0.5, 1.0, 2.0]
        self.speed_idx = 3 # 1.0x default
        
        # Timing anchors
        self.play_start_time = 0.0
        self.play_start_frame_idx = 0
        
        # Zoom & Pan coordinate state (synced)
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        self.drag_start_pan = None
        
        # Pre-allocated buffers for zoom mode
        self.buffer_triple = None
        
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
            
        # 1. Video B is ALWAYS reference
        vid_ref = next((v for v in vids if v["metric"] == "reference"), vids[0])
        
        # Distorted variants only
        non_ref = [v for v in vids if v["metric"] != "reference"]
        if not non_ref:
            vid_a = vid_ref
            vid_c = vid_ref
        else:
            # Group unique metrics
            metrics = list(dict.fromkeys([v["metric"] for v in non_ref]))
            
            # Select A (Left)
            self.metric_idx_a = self.metric_idx_a % len(metrics)
            active_metric_a = metrics[self.metric_idx_a]
            level_str_a = f"level{self.level_idx_a}"
            vid_a = next((v for v in non_ref if v["metric"] == active_metric_a and v["level"] == level_str_a), None)
            if not vid_a:
                vid_a = next((v for v in non_ref if v["metric"] == active_metric_a), non_ref[0])
                
            # Select C (Right)
            self.metric_idx_c = self.metric_idx_c % len(metrics)
            active_metric_c = metrics[self.metric_idx_c]
            level_str_c = f"level{self.level_idx_c}"
            vid_c = next((v for v in non_ref if v["metric"] == active_metric_c and v["level"] == level_str_c), None)
            if not vid_c:
                vid_c = next((v for v in non_ref if v["metric"] == active_metric_c), non_ref[0])
                
        self.meta_a = vid_a
        self.meta_ref = vid_ref
        self.meta_c = vid_c
        
        print(f"\n--- Loading Triple Video Set ---")
        print(f"Scene           : {SCENES[self.scene_idx].upper()}")
        print(f"Left Video (A)  : {vid_a['filename']}")
        print(f"Center Video (R): {vid_ref['filename']} [LOCKED]")
        print(f"Right Video (C) : {vid_c['filename']}")
        
        self.frames_a = self.preload_video(vid_a["path"])
        self.frames_ref = self.preload_video(vid_ref["path"])
        self.frames_c = self.preload_video(vid_c["path"])
        
        # Pre-combine all frames in a 16:9 Pyramid layout (Ref on top, Distorted below)
        print("Pre-stacking Triple frames into Pyramid layout...", flush=True)
        self.frames_triple = []
        len_a = len(self.frames_a)
        len_ref = len(self.frames_ref)
        len_c = len(self.frames_c)
        num_frames = max(len_a, len_ref, len_c)
        
        h, w = self.frames_ref[0].shape[:2]
        for i in range(num_frames):
            fa = self.frames_a[i % len_a]
            fref = self.frames_ref[i % len_ref]
            fc = self.frames_c[i % len_c]
            
            # Construct 2x2 grid (Ref centered on top, A and C on bottom)
            frame = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)
            frame[:h, w//2 : w//2 + w] = fref
            frame[h:, :w] = fa
            frame[h:, w:] = fc
            self.frames_triple.append(frame)
            
            if (i + 1) % 100 == 0 or i == num_frames - 1:
                print(f"  Stacked {i + 1}/{num_frames} frames...", flush=True)
        print(" Done.")
        
        # Reset playback state
        self.frame_idx = 0
        self.update_playback_anchor()
        
        # Pre-allocate frames (used for zoom mode)
        if self.frames_a:
            self.buffer_triple = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

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

    def update_playback_anchor(self):
        self.play_start_time = time.perf_counter()
        self.play_start_frame_idx = self.frame_idx

    def apply_zoom_pan(self, img):
        if self.zoom_scale == 1.0:
            return img
            
        h, w = img.shape[:2]
        crop_w = int(w / self.zoom_scale)
        crop_h = int(h / self.zoom_scale)
        
        center_x = w // 2 + self.pan_x
        center_y = h // 2 + self.pan_y
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
            
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized

    def get_display_frame(self):
        if not self.frames_a or not self.frames_ref or not self.frames_c:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        # Update FPS history
        now = time.perf_counter()
        self.fps_history.append(now)
        self.fps_history = [t for t in self.fps_history if now - t <= 1.0]
        if len(self.fps_history) > 1:
            self.current_fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
        else:
            self.current_fps = 0.0
            
        idx_a = self.frame_idx % len(self.frames_a)
        idx_ref = self.frame_idx % len(self.frames_ref)
        idx_c = self.frame_idx % len(self.frames_c)
        
        h, w = self.frames_ref[idx_ref].shape[:2]
        
        # Zero-copy rendering paths
        if self.zoom_scale == 1.0:
            combined = self.frames_triple[self.frame_idx % len(self.frames_triple)]
            display_w = w * 2
        else:
            # Zoom active: Crop & resize allocation (fallback)
            frame_a_disp = self.apply_zoom_pan(self.frames_a[idx_a])
            frame_ref_disp = self.apply_zoom_pan(self.frames_ref[idx_ref])
            frame_c_disp = self.apply_zoom_pan(self.frames_c[idx_c])
            
            self.buffer_triple[:h, :w//2] = 0
            self.buffer_triple[:h, w//2 + w:] = 0
            self.buffer_triple[:h, w//2 : w//2 + w] = frame_ref_disp
            self.buffer_triple[h:, :w] = frame_a_disp
            self.buffer_triple[h:, w:] = frame_c_disp
            combined = self.buffer_triple
            display_w = w * 2
            
        # Draw UI overlay if enabled
        if self.show_ui:
            combined_disp = combined.copy()
            self.draw_ui(combined_disp, h, display_w)
            return combined_disp
            
        return combined

    def draw_ui(self, img, h, w):
        # Semi-transparent dark bar at the top
        cv2.rectangle(img, (0, 0), (w, 40), (15, 15, 20), -1)
        
        scene_name = SCENES[self.scene_idx].upper()
        speed = self.speeds[self.speed_idx]
        status = "PLAYING" if self.playing else "PAUSED"
        
        text = f"[{status}] Scene: {scene_name} | Speed: {speed}x | Layout: PYRAMID COMPARISON | FPS: {self.current_fps:.1f}"
        cv2.putText(img, text, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_8)
        
        # Labels for sources
        lbl_a = f"Left (A): {self.meta_a['label']}"
        lbl_ref = f"Top (Reference)"
        lbl_c = f"Right (C): {self.meta_c['label']}"
        
        # Reference is on top row (centered: w/4 to 3*w/4)
        cv2.putText(img, lbl_ref, (w // 4 + 15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_8)
        # Distorted A is bottom-left (0 to w/2)
        cv2.putText(img, lbl_a, (15, h * 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_8)
        # Distorted C is bottom-right (w/2 to w)
        cv2.putText(img, lbl_c, (w // 2 + 15, h * 2 - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1, cv2.LINE_8)
        
        # Frame counter
        total_frames = len(self.frames_triple)
        frame_text = f"Frame: {self.frame_idx + 1}/{total_frames} (Zoom: {int(self.zoom_scale*100)}%)"
        cv2.putText(img, frame_text, (w - 260, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_8)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_start_pan = (self.pan_x, self.pan_y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drag_start is not None:
                dx = x - self.drag_start[0]
                dy = y - self.drag_start[1]
                self.pan_x = self.drag_start_pan[0] - int(dx / self.zoom_scale)
                self.pan_y = self.drag_start_pan[1] - int(dy / self.zoom_scale)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drag_start = None
            self.drag_start_pan = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            delta = flags
            if delta > 0:
                self.zoom_scale = min(20.0, self.zoom_scale * 1.15)
            else:
                self.zoom_scale = max(1.0, self.zoom_scale / 1.15)
                if self.zoom_scale == 1.0:
                    self.pan_x = 0
                    self.pan_y = 0

    def run(self):
        window_name = "GAIM240 Video Quality Comparer (Pyramid View)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # 16:9 aspect ratio fits 2K/1080p monitors perfectly
        cv2.resizeWindow(window_name, 1280, 720)
        cv2.setMouseCallback(window_name, self.on_mouse)
        
        print("\n=== Native Triple Video Player Started ===")
        print("Controls:")
        print("  [Space]       : Play / Pause")
        print("  [Left/Right]  : Frame step")
        print("  [1/2/3/4]     : Switch Scene (Marbles, Pink Room, Subway, Zeroday)")
        print("  [A] / [Z]     : Cycle LEFT video metric (A) [Prev / Next]")
        print("  [K] / [L]     : Cycle RIGHT video metric (C) [Prev / Next]")
        print("  [0/1/2]       : Set LEFT video (A) distortion level")
        print("  [7/8/9]       : Set RIGHT video (C) distortion level")
        print("  [-] / [+]     : Playback speed (0.1x, 0.25x, 0.5x, 1x, 2x)")
        print("  [U]           : Toggle UI Overlay (hides text rendering for maximum FPS)")
        print("  [R]           : Reset Zoom & Pan")
        print("  [Q] or [Esc]  : Quit")
        
        last_render_time = time.perf_counter()
        while True:
            speed_mult = self.speeds[self.speed_idx]
            base_fps = 240.0
            target_fps = base_fps * speed_mult
            frame_time = 1.0 / target_fps
            
            # 1. Update playback state (time-based progression)
            if self.playing:
                now = time.perf_counter()
                elapsed = now - self.play_start_time
                total_frames = len(self.frames_triple)
                
                self.frame_idx = self.play_start_frame_idx + int(elapsed * target_fps)
                if self.frame_idx >= total_frames:
                    self.play_start_time = now
                    self.play_start_frame_idx = 0
                    self.frame_idx = 0
                    
            # 2. Get and show display frame
            frame = self.get_display_frame()
            cv2.imshow(window_name, frame)
            
            # 3. GUI Events & wait (every 4 frames)
            if self.frame_idx % 4 == 0:
                key = cv2.waitKey(1) & 0xFF
            else:
                key = 255
            
            total_frames = len(self.frames_triple)
            
            # Console FPS status
            if self.playing and self.frame_idx % 240 == 0:
                print(f"  [Performance] {self.current_fps:.1f} FPS (UI: {'ON' if self.show_ui else 'OFF'})")
            
            if key == ord('q') or key == 27:
                break
            elif key == ord(' '):
                self.playing = not self.playing
                self.update_playback_anchor()
            elif key == 83 or key == 3 or key == ord('d'): # Right Arrow or d
                self.playing = False
                self.frame_idx = (self.frame_idx + 1) % total_frames
                self.update_playback_anchor()
            elif key == 81 or key == 2 or key == ord('a'): # Left Arrow or a
                self.playing = False
                self.frame_idx = (self.frame_idx - 1) % total_frames
                self.update_playback_anchor()
            elif key == ord('r'):
                self.zoom_scale = 1.0
                self.pan_x = 0
                self.pan_y = 0
            elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                self.scene_idx = key - ord('1')
                self.load_active_videos()
            elif key == ord('0'):
                self.level_idx_a = 0
                self.load_active_videos()
            elif key == ord('1'):
                self.level_idx_a = 1
                self.load_active_videos()
            elif key == ord('2'):
                self.level_idx_a = 2
                self.load_active_videos()
            elif key == ord('7'):
                self.level_idx_c = 0
                self.load_active_videos()
            elif key == ord('8'):
                self.level_idx_c = 1
                self.load_active_videos()
            elif key == ord('9'):
                self.level_idx_c = 2
                self.load_active_videos()
            elif key == ord('z'): # Left metric next
                self.metric_idx_a += 1
                self.load_active_videos()
            elif key == ord('a'): # Left metric prev
                # Check mapping overlap
                # Since Left Arrow is also mapped to 'a' on some Linux setups, let's keep it safe
                self.metric_idx_a = max(0, self.metric_idx_a - 1)
                self.load_active_videos()
            elif key == ord('l'): # Right metric next
                self.metric_idx_c += 1
                self.load_active_videos()
            elif key == ord('k'): # Right metric prev
                self.metric_idx_c = max(0, self.metric_idx_c - 1)
                self.load_active_videos()
            elif key == ord('='):
                self.speed_idx = min(len(self.speeds) - 1, self.speed_idx + 1)
                self.update_playback_anchor()
            elif key == ord('-'):
                self.speed_idx = max(0, self.speed_idx - 1)
                self.update_playback_anchor()
            elif key == ord('u'):
                self.show_ui = not self.show_ui
                print(f"UI Overlay toggled: {'ON' if self.show_ui else 'OFF'}")
                
            # 4. High-Precision Playback Pacing
            if self.playing:
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
    player = TripleDatasetPlayer()
    player.run()
