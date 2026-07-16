# /// script
# dependencies = [
#   "glfw",
#   "PyOpenGL",
#   "opencv-python",
#   "numpy",
# ]
# ///

import os
import re
import sys
import time
import csv
import glfw
import cv2
import numpy as np
from pathlib import Path
from OpenGL.GL import *

# Paths
DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
SCENES = ["marbles", "pink_room", "subway", "zeroday"]

class ProfileTriplePlayer:
    def __init__(self):
        self.scene_idx = 0
        self.metric_idx_a = 0
        self.metric_idx_c = 1
        self.level_idx_a = 1
        self.level_idx_c = 1
        
        # Check command-line arguments for VSync control
        self.vsync = True
        if "--no-vsync" in sys.argv:
            self.vsync = False
            
        self.videos = {}
        self.scan_dataset()
        
        self.frames_a = []
        self.frames_ref = []
        self.frames_c = []
        
        self.meta_a = None
        self.meta_ref = None
        self.meta_c = None
        
        self.frame_idx = 0
        
        # Interactive Zoom / Pan state
        self.zoom_scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        
        # Textures
        self.tex_a = None
        self.tex_ref = None
        self.tex_c = None
        
        # Multi-stage profiling arrays
        self.frame_indices = []
        self.get_image_durations = []     # t1 - t0 (ms) - Time to fetch frame from RAM
        self.upload_draw_durations = []   # t2 - t1 (ms) - OpenGL texture upload & draw commands
        self.flip_durations = []          # t3 - t2 (ms) - time spent waiting in glfw.swap_buffers()
        self.total_loop_durations = []    # t3 - t0 (ms) - total time for one full loop cycle
        self.swap_timestamps = []         # t3 (sec) - timestamp immediately after buffer flip
        
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
                
            scene = None
            for s in SCENES:
                if file.startswith(s + "_"):
                    scene = s
                    break
            if not scene:
                continue
                
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
        return self.videos[SCENES[self.scene_idx]]

    def load_active_videos(self):
        vids = self.get_current_video_list()
        if not vids:
            return
            
        vid_ref = next((v for v in vids if v["metric"] == "reference"), vids[0])
        non_ref = [v for v in vids if v["metric"] != "reference"]
        
        metrics = list(dict.fromkeys([v["metric"] for v in non_ref]))
        
        # A (Left)
        self.metric_idx_a = self.metric_idx_a % len(metrics)
        active_metric_a = metrics[self.metric_idx_a]
        vid_a = next((v for v in non_ref if v["metric"] == active_metric_a and v["level"] == f"level{self.level_idx_a}"), non_ref[0])
        
        # C (Right)
        self.metric_idx_c = self.metric_idx_c % len(metrics)
        active_metric_c = metrics[self.metric_idx_c]
        vid_c = next((v for v in non_ref if v["metric"] == active_metric_c and v["level"] == f"level{self.level_idx_c}"), non_ref[0])
        
        self.meta_a = vid_a
        self.meta_ref = vid_ref
        self.meta_c = vid_c
        
        print(f"\n--- Profiler: Preloading Scene {SCENES[self.scene_idx].upper()} ---")
        self.frames_a = self.preload_video(vid_a["path"])
        self.frames_ref = self.preload_video(vid_ref["path"])
        self.frames_c = self.preload_video(vid_c["path"])
        
    def preload_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    def setup_textures(self):
        glEnable(GL_TEXTURE_2D)
        self.tex_a, self.tex_ref, self.tex_c = glGenTextures(3)
        for tex in [self.tex_a, self.tex_ref, self.tex_c]:
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

    def upload_texture(self, tex_id, frame):
        h, w = frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_BGR, GL_UNSIGNED_BYTE, frame.data)

    def draw_quad(self, x1, y1, x2, y2):
        w_tex = 1.0 / self.zoom_scale
        h_tex = 1.0 / self.zoom_scale
        
        tc_x1 = 0.5 + self.pan_x - w_tex / 2.0
        tc_y1 = 0.5 + self.pan_y - h_tex / 2.0
        tc_x2 = tc_x1 + w_tex
        tc_y2 = tc_y1 + h_tex
        
        glBegin(GL_QUADS)
        glTexCoord2f(tc_x1, tc_y2); glVertex2f(x1, y1)
        glTexCoord2f(tc_x2, tc_y2); glVertex2f(x2, y1)
        glTexCoord2f(tc_x2, tc_y1); glVertex2f(x2, y2)
        glTexCoord2f(tc_x1, tc_y1); glVertex2f(x1, y2)
        glEnd()

    def render_pyramid(self, window, frame_a, frame_ref, frame_c):
        W, H = glfw.get_framebuffer_size(window)
        target_aspect = 16.0 / 9.0
        
        w_view = W
        h_view = int(W / target_aspect)
        if h_view > H:
            h_view = H
            w_view = int(H * target_aspect)
            
        x_offset = (W - w_view) // 2
        y_offset = (H - h_view) // 2
        
        glViewport(0, 0, W, H)
        glClearColor(0.02, 0.02, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        glViewport(x_offset, y_offset, w_view, h_view)
        
        # Upload active frames to textures on GPU
        self.upload_texture(self.tex_a, frame_a)
        self.upload_texture(self.tex_ref, frame_ref)
        self.upload_texture(self.tex_c, frame_c)
        
        # Reference (top-center)
        glBindTexture(GL_TEXTURE_2D, self.tex_ref)
        self.draw_quad(-0.5, 0.0, 0.5, 1.0)
        
        # Left (A)
        glBindTexture(GL_TEXTURE_2D, self.tex_a)
        self.draw_quad(-1.0, -1.0, 0.0, 0.0)
        
        # Right (C)
        glBindTexture(GL_TEXTURE_2D, self.tex_c)
        self.draw_quad(0.0, -1.0, 1.0, 0.0)

    def generate_report(self):
        print("\n" + "="*55)
        print("    VISUAL PERCEPTION EXPERIMENT REPORT (MULTI-STAGE)    ")
        print("="*55)
        
        total_logged = len(self.swap_timestamps)
        if total_logged < 2:
            print("Not enough frames rendered to generate profile statistics.")
            return
            
        timestamps = np.array(self.swap_timestamps)
        intervals_ms = np.diff(timestamps) * 1000.0
        
        target_interval_ms = 1000.0 / 240.0 # 4.167ms
        
        # Display swap stats
        mean_int = np.mean(intervals_ms)
        std_int = np.std(intervals_ms)
        min_int = np.min(intervals_ms)
        max_int = np.max(intervals_ms)
        median_int = np.median(intervals_ms)
        
        # Dropped frames
        dropped_frames_threshold = target_interval_ms * 1.5
        dropped_count = np.sum(intervals_ms > dropped_frames_threshold)
        dropped_percentage = (dropped_count / len(intervals_ms)) * 100.0
        
        # Stage metrics
        get_img_arr = np.array(self.get_image_durations)
        upload_draw_arr = np.array(self.upload_draw_durations)
        flip_arr = np.array(self.flip_durations)
        loop_arr = np.array(self.total_loop_durations)
        
        print(f"Target Frame Rate       : 240 FPS")
        print(f"Target Frame Interval   : {target_interval_ms:.3f} ms")
        print(f"VSync Configured        : {'ENABLED (swap_interval=1)' if self.vsync else 'DISABLED (swap_interval=0)'}")
        print(f"Total Frames Profileed  : {total_logged}")
        print("-"*55)
        print(f"Display Swap Interval Metrics (Stimulus Presentation):")
        print(f"  Mean Interval         : {mean_int:.3f} ms (Actual Frame Rate: {1000.0/mean_int:.2f} Hz)")
        print(f"  Median Interval       : {median_int:.3f} ms")
        print(f"  Min Interval          : {min_int:.3f} ms")
        print(f"  Max Interval          : {max_int:.3f} ms")
        print(f"  Jitter (Std Dev)      : {std_int:.3f} ms")
        print(f"  Dropped Frames (>6.25ms): {dropped_count} / {len(intervals_ms)} ({dropped_percentage:.2f}%)")
        print("-"*55)
        print("Rendering Pipeline Stages Durations (Mean ± Std Dev):")
        print(f"  1. Fetch Frame (RAM)  : {np.mean(get_img_arr):.4f} ± {np.std(get_img_arr):.4f} ms")
        print(f"  2. Upload & Draw (GPU): {np.mean(upload_draw_arr):.4f} ± {np.std(upload_draw_arr):.4f} ms")
        print(f"  3. Swap/VSync (Block) : {np.mean(flip_arr):.4f} ± {np.std(flip_arr):.4f} ms")
        print(f"  4. Total Loop Overhead: {np.mean(loop_arr):.4f} ± {np.std(loop_arr):.4f} ms")
        print("-"*55)
        
        # ASCII Histogram of Display Intervals
        print("Inter-Frame Interval Distribution (ms):")
        bins = [0, 2.0, 3.5, 4.0, 4.3, 4.5, 5.0, 6.25, 8.3, 10.0, float('inf')]
        counts, edges = np.histogram(intervals_ms, bins=bins)
        
        max_chars = 30
        max_count = max(counts) if max(counts) > 0 else 1
        for i in range(len(counts)):
            bin_start = f"{edges[i]:.2f}"
            bin_end = f"{edges[i+1]:.2f}" if edges[i+1] != float('inf') else "inf"
            label = f"  [{bin_start} - {bin_end}]".ljust(18)
            bar = "#" * int((counts[i] / max_count) * max_chars)
            print(f"{label} : {counts[i]:<5} {bar}")
            
        # Write to CSV
        csv_filename = "profile_results.csv"
        print("-"*55)
        print(f"Saving multi-stage profiling data to: {csv_filename}...")
        
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                "FrameIndex", 
                "GetImageDuration_ms", 
                "UploadDrawDuration_ms", 
                "FlipDuration_ms", 
                "TotalLoopDuration_ms",
                "SwapTimestamp_sec", 
                "InterFrameInterval_ms"
            ])
            
            # First frame has no interval
            writer.writerow([
                self.frame_indices[0], 
                f"{self.get_image_durations[0]:.6f}", 
                f"{self.upload_draw_durations[0]:.6f}", 
                f"{self.flip_durations[0]:.6f}", 
                f"{self.total_loop_durations[0]:.6f}", 
                f"{self.swap_timestamps[0]:.6f}", 
                "N/A"
            ])
            for idx in range(1, len(self.swap_timestamps)):
                writer.writerow([
                    self.frame_indices[idx],
                    f"{self.get_image_durations[idx]:.6f}",
                    f"{self.upload_draw_durations[idx]:.6f}",
                    f"{self.flip_durations[idx]:.6f}",
                    f"{self.total_loop_durations[idx]:.6f}",
                    f"{self.swap_timestamps[idx]:.6f}",
                    f"{intervals_ms[idx-1]:.6f}"
                ])
                
        print("Report generation complete.")
        print("="*55)

    def run(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            return
            
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        window = glfw.create_window(1280, 720, "GAIM240 Video Quality Comparer (Triple OpenGL Profiler)", None, None)
        if not window:
            glfw.terminate()
            print("Failed to create GLFW window")
            return
            
        glfw.make_context_current(window)
        if self.vsync:
            glfw.swap_interval(1)
            print("VSync: ENABLED (locked to monitor refresh rate)")
        else:
            glfw.swap_interval(0)
            print("VSync: DISABLED (unlocked frame rate benchmark)")
            
        self.setup_textures()
        
        total_frames_to_profile = max(len(self.frames_a), len(self.frames_ref), len(self.frames_c))
        
        print("\n=== Native OpenGL Triple Player Profiling Mode ===")
        print(f"Pacing: Playback speed locked to 1.0x.")
        print(f"Target duration: 1 full scene loop ({total_frames_to_profile} frames).")
        print("Profiling starts automatically. Please do NOT resize or interact with the window during run to preserve precision.")
        print("Press [Q] or [Esc] to stop early and print current profile data.")
        
        # Start immediately
        self.play_start_time = time.perf_counter()
        self.play_start_frame_idx = 0
        
        target_fps = 240.0
        frame_time = 1.0 / target_fps
        
        last_render_time = time.perf_counter()
        
        while not glfw.window_should_close(window) and self.frame_idx < total_frames_to_profile:
            t0 = time.perf_counter() # Loop start
            
            # 1. Update playback frame pointer & get images
            elapsed = t0 - self.play_start_time
            self.frame_idx = int(elapsed * target_fps)
            if self.frame_idx >= total_frames_to_profile:
                break
                
            idx_a = self.frame_idx % len(self.frames_a)
            idx_ref = self.frame_idx % len(self.frames_ref)
            idx_c = self.frame_idx % len(self.frames_c)
            
            frame_a = self.frames_a[idx_a]
            frame_ref = self.frames_ref[idx_ref]
            frame_c = self.frames_c[idx_c]
            
            t1 = time.perf_counter() # After image retrieval
            
            # 2. Render & submit GPU draw commands
            self.render_pyramid(window, frame_a, frame_ref, frame_c)
            
            t2 = time.perf_counter() # Right before buffer swap
            
            # 3. Swap buffers (blocks here waiting for VSync)
            glfw.swap_buffers(window)
            
            t3 = time.perf_counter() # Immediately after swap
            
            # Record durations in milliseconds
            self.frame_indices.append(self.frame_idx)
            self.get_image_durations.append((t1 - t0) * 1000.0)
            self.upload_draw_durations.append((t2 - t1) * 1000.0)
            self.flip_durations.append((t3 - t2) * 1000.0)
            self.total_loop_durations.append((t3 - t0) * 1000.0)
            self.swap_timestamps.append(t3)
            
            glfw.poll_events()
            
            # Check key presses to exit early
            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
                print("\nProfiling stopped early by user.")
                break
                
            # Title bar simple FPS status
            if self.frame_idx % 60 == 0:
                glfw.set_window_title(window, f"GAIM240 Profiler | Progress: {self.frame_idx}/{total_frames_to_profile} frames")
                
            # 4. High-Precision Playback Pacing
            time_elapsed = time.perf_counter() - t0
            time_remaining = frame_time - time_elapsed
            if time_remaining > 0:
                time.sleep(time_remaining)
                
            last_render_time = time.perf_counter()
            
        glfw.destroy_window(window)
        glfw.terminate()
        
        # Generate the results report
        self.generate_report()

if __name__ == "__main__":
    player = ProfileTriplePlayer()
    player.run()
