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
import time
import glfw
import cv2
import numpy as np
from pathlib import Path
from OpenGL.GL import *

# Paths
DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
SCENES = ["marbles", "pink_room", "subway", "zeroday"]

class OpenGLTriplePlayer:
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
        
        self.meta_a = None
        self.meta_ref = None
        self.meta_c = None
        
        self.frame_idx = 0
        self.playing = False
        
        # Playback speed controls
        self.speeds = [0.1, 0.25, 0.5, 1.0, 2.0]
        self.speed_idx = 3 # 1.0x by default
        
        # Interactive Zoom / Pan state (synced, handled on GPU texture coordinates)
        self.zoom_scale = 1.0
        self.pan_x = 0.0  # Normalized texture offset [-0.5, 0.5]
        self.pan_y = 0.0
        
        # Mouse dragging state
        self.drag_start = None
        self.drag_start_pan = None
        
        # Textures
        self.tex_a = None
        self.tex_ref = None
        self.tex_c = None
        
        # Timing anchors
        self.play_start_time = 0.0
        self.play_start_frame_idx = 0
        
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
            
        # Video B is ALWAYS reference
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
        
        # Reset playback state
        self.frame_idx = 0
        self.update_playback_anchor()
        
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

    def setup_textures(self):
        glEnable(GL_TEXTURE_2D)
        self.tex_a, self.tex_ref, self.tex_c = glGenTextures(3)
        for tex in [self.tex_a, self.tex_ref, self.tex_c]:
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_BGR, GL_UNSIGNED_BYTE, None)

    def upload_texture(self, tex_id, frame):
        h, w = frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, tex_id)
        
        if self.zoom_scale > 1.0:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        else:
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, frame.data)

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

    def render(self, window):
        W, H = glfw.get_framebuffer_size(window)
        # Target aspect ratio is 16:9 for the pyramid layout
        target_aspect = 16.0 / 9.0
        
        w_view = W
        h_view = int(W / target_aspect)
        if h_view > H:
            h_view = H
            w_view = int(H * target_aspect)
            
        x_offset = (W - w_view) // 2
        y_offset = (H - h_view) // 2
        
        # Clear the entire window space first
        glViewport(0, 0, W, H)
        glClearColor(0.02, 0.02, 0.03, 1.0)
        glClear(GL_COLOR_BUFFER_BIT)
        
        # Set active viewport to letterbox bounds
        glViewport(x_offset, y_offset, w_view, h_view)
        
        if not self.frames_a or not self.frames_ref or not self.frames_c:
            return
            
        idx_a = self.frame_idx % len(self.frames_a)
        idx_ref = self.frame_idx % len(self.frames_ref)
        idx_c = self.frame_idx % len(self.frames_c)
        
        # Upload active frames to textures on GPU
        self.upload_texture(self.tex_a, self.frames_a[idx_a])
        self.upload_texture(self.tex_ref, self.frames_ref[idx_ref])
        self.upload_texture(self.tex_c, self.frames_c[idx_c])
        
        # 16:9 Pyramid Layout:
        # Reference (top-center): x=[-0.5, 0.5], y=[0, 1]
        # Distorted A (bottom-left): x=[-1.0, 0.0], y=[-1, 0]
        # Distorted C (bottom-right): x=[0.0, 1.0], y=[-1, 0]
        
        # Draw Top Center (Reference)
        glBindTexture(GL_TEXTURE_2D, self.tex_ref)
        self.draw_quad(-0.5, 0.0, 0.5, 1.0)
        
        # Draw Bottom Left (A)
        glBindTexture(GL_TEXTURE_2D, self.tex_a)
        self.draw_quad(-1.0, -1.0, 0.0, 0.0)
        
        # Draw Bottom Right (C)
        glBindTexture(GL_TEXTURE_2D, self.tex_c)
        self.draw_quad(0.0, -1.0, 1.0, 0.0)

    # Interactive input handlers
    def on_scroll(self, window, xoffset, yoffset):
        if yoffset > 0:
            self.zoom_scale = min(30.0, self.zoom_scale * 1.15)
        else:
            self.zoom_scale = max(1.0, self.zoom_scale / 1.15)
            if self.zoom_scale == 1.0:
                self.pan_x = 0.0
                self.pan_y = 0.0

    def on_mouse_button(self, window, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            if action == glfw.PRESS:
                x, y = glfw.get_cursor_pos(window)
                self.drag_start = (x, y)
                self.drag_start_pan = (self.pan_x, self.pan_y)
            elif action == glfw.RELEASE:
                self.drag_start = None
                self.drag_start_pan = None

    def on_mouse_move(self, window, xpos, ypos):
        if self.drag_start is not None:
            width, height = glfw.get_window_size(window)
            dx = xpos - self.drag_start[0]
            dy = ypos - self.drag_start[1]
            
            # Map screen pixel dragging to normalized OpenGL texture space offsets
            # Divide by zoom scale so movement maps naturally to speed
            self.pan_x = self.drag_start_pan[0] - (dx / width) / self.zoom_scale
            self.pan_y = self.drag_start_pan[1] + (dy / height) / self.zoom_scale
            
            # Clamp pan ranges to keep textures within bounds
            max_pan_x = 0.5 * (1.0 - 1.0 / self.zoom_scale)
            max_pan_y = 0.5 * (1.0 - 1.0 / self.zoom_scale)
            self.pan_x = np.clip(self.pan_x, -max_pan_x, max_pan_x)
            self.pan_y = np.clip(self.pan_y, -max_pan_y, max_pan_y)

    def on_key(self, window, key, scancode, action, mods):
        if action != glfw.PRESS:
            return
            
        total_frames = max(len(self.frames_a), len(self.frames_ref), len(self.frames_c))
        
        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
            
        elif key == glfw.KEY_SPACE:
            self.playing = not self.playing
            self.update_playback_anchor()
            
        elif key == glfw.KEY_RIGHT or key == glfw.KEY_D:
            self.playing = False
            self.frame_idx = (self.frame_idx + 1) % total_frames
            self.update_playback_anchor()
            
        elif key == glfw.KEY_LEFT or key == glfw.KEY_A:
            self.playing = False
            self.frame_idx = (self.frame_idx - 1) % total_frames
            self.update_playback_anchor()
            
        elif key == glfw.KEY_R:
            self.zoom_scale = 1.0
            self.pan_x = 0.0
            self.pan_y = 0.0
            
        elif key in [glfw.KEY_1, glfw.KEY_2, glfw.KEY_3, glfw.KEY_4]:
            self.scene_idx = key - glfw.KEY_1
            self.load_active_videos()
            
        # Left Video (A) level selectors
        elif key == glfw.KEY_0:
            self.level_idx_a = 0
            self.load_active_videos()
        elif key == glfw.KEY_1:
            self.level_idx_a = 1
            self.load_active_videos()
        elif key == glfw.KEY_2:
            self.level_idx_a = 2
            self.load_active_videos()
            
        # Right Video (C) level selectors
        elif key == glfw.KEY_7:
            self.level_idx_c = 0
            self.load_active_videos()
        elif key == glfw.KEY_8:
            self.level_idx_c = 1
            self.load_active_videos()
        elif key == glfw.KEY_9:
            self.level_idx_c = 2
            self.load_active_videos()
            
        # Left Video (A) metric cycle
        elif key == glfw.KEY_Z:
            self.metric_idx_a += 1
            self.load_active_videos()
        elif key == glfw.KEY_X:
            self.metric_idx_a = max(0, self.metric_idx_a - 1)
            self.load_active_videos()
            
        # Right Video (C) metric cycle
        elif key == glfw.KEY_L:
            self.metric_idx_c += 1
            self.load_active_videos()
        elif key == glfw.KEY_K:
            self.metric_idx_c = max(0, self.metric_idx_c - 1)
            self.load_active_videos()
            
        # Playback Speed
        elif key == glfw.KEY_EQUAL: # Plus
            self.speed_idx = min(len(self.speeds) - 1, self.speed_idx + 1)
            self.update_playback_anchor()
        elif key == glfw.KEY_MINUS:
            self.speed_idx = max(0, self.speed_idx - 1)
            self.update_playback_anchor()

    def run(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            return
            
        borderless = "--borderless" in sys.argv
        windowed = "--windowed" in sys.argv
        
        monitor = glfw.get_primary_monitor()
        if monitor:
            mode = glfw.get_video_mode(monitor)
            if hasattr(mode, 'size'):
                mon_w, mon_h = mode.size.width, mode.size.height
            elif hasattr(mode, 'width'):
                mon_w, mon_h = mode.width, mode.height
            else:
                mon_w, mon_h = mode[0], mode[1]
        else:
            mon_w, mon_h = 1280, 720
            
        glfw.default_window_hints()
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
        if borderless:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            window = glfw.create_window(mon_w, mon_h, "GAIM240 Video Quality Comparer (Triple OpenGL/GPU)", None, None)
            if window:
                glfw.set_window_pos(window, 0, 0)
        elif windowed or not monitor:
            window = glfw.create_window(1280, 720, "GAIM240 Video Quality Comparer (Triple OpenGL/GPU)", None, None)
        else:
            window = glfw.create_window(mon_w, mon_h, "GAIM240 Video Quality Comparer (Triple OpenGL/GPU)", monitor, None)
            if not window:
                glfw.window_hint(glfw.DECORATED, glfw.FALSE)
                window = glfw.create_window(mon_w, mon_h, "GAIM240 Video Quality Comparer (Triple OpenGL/GPU)", None, None)
                if window:
                    glfw.set_window_pos(window, 0, 0)
            
        if not window:
            glfw.terminate()
            print("Failed to create GLFW window")
            return
            
        glfw.make_context_current(window)
        glfw.swap_interval(1)  # VSync locked to 240Hz screen
        
        glfw.set_scroll_callback(window, self.on_scroll)
        glfw.set_mouse_button_callback(window, self.on_mouse_button)
        glfw.set_cursor_pos_callback(window, self.on_mouse_move)
        glfw.set_key_callback(window, self.on_key)
        
        self.setup_textures()
        
        print("\n=== Native GLFW/OpenGL Triple Player (Pyramid Layout) Started ===")
        print("Controls:")
        print("  [Space]       : Play / Pause")
        print("  [Left/Right]  : Frame step")
        print("  [1/2/3/4]     : Switch Scene (Marbles, Pink Room, Subway, Zeroday)")
        print("  [Z] / [X]     : Cycle LEFT video metric (A) [Next / Previous]")
        print("  [L] / [K]     : Cycle RIGHT video metric (C) [Next / Previous]")
        print("  [0/1/2]       : Set LEFT video (A) distortion level")
        print("  [7/8/9]       : Set RIGHT video (C) distortion level")
        print("  [-] / [+]     : Playback speed (0.1x, 0.25x, 0.5x, 1x, 2x)")
        print("  [R]           : Reset Zoom & Pan")
        print("  [Q] or [Esc]  : Quit")
        
        last_render_time = time.perf_counter()
        
        while not glfw.window_should_close(window):
            speed_mult = self.speeds[self.speed_idx]
            base_fps = 240.0
            target_fps = base_fps * speed_mult
            frame_time = 1.0 / target_fps
            
            now = time.perf_counter()
            
            # 1. Update playback state (time-based progression)
            if self.playing:
                elapsed = now - self.play_start_time
                total_frames = max(len(self.frames_a), len(self.frames_ref), len(self.frames_c))
                self.frame_idx = self.play_start_frame_idx + int(elapsed * target_fps)
                if self.frame_idx >= total_frames:
                    self.play_start_time = now
                    self.play_start_frame_idx = 0
                    self.frame_idx = 0
            
            # 2. Render OpenGL textures to window
            self.render(window)
            
            # 3. Swap buffers & poll events
            glfw.swap_buffers(window)
            glfw.poll_events()
            
            # Update FPS rolling counter
            fps_now = time.perf_counter()
            self.fps_history.append(fps_now)
            self.fps_history = [t for t in self.fps_history if fps_now - t <= 1.0]
            if len(self.fps_history) > 1:
                self.current_fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
            else:
                self.current_fps = 0.0
                
            # Update window title with FPS and metadata
            title = f"GAIM240 Triple Comparer (OpenGL) | {self.current_fps:.1f} FPS | A: {self.meta_a['label']} | C: {self.meta_c['label']}"
            glfw.set_window_title(window, title)
            
            # Console performance status
            if self.playing and self.frame_idx % 240 == 0:
                print(f"  [Performance] {self.current_fps:.1f} FPS (VSync locked)")
                
            # 4. High-Precision Playback Pacing
            if self.playing:
                time_elapsed = time.perf_counter() - now
                time_remaining = frame_time - time_elapsed
                if time_remaining > 0:
                    time.sleep(time_remaining)
            else:
                time.sleep(0.03)
                
            last_render_time = time.perf_counter()
            
        glfw.terminate()

if __name__ == "__main__":
    player = OpenGLTriplePlayer()
    player.run()
