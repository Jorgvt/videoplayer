# /// script
# dependencies = [
#   "glfw",
#   "PyOpenGL",
#   "opencv-python",
#   "numpy",
#   "pandas",
# ]
# ///

import os
import re
import sys
import time
import csv
import gc
import glfw
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from OpenGL.GL import *

DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
SCENES = ["attic", "bistro_exterior", "bistro_interior", "classroom", "landscape", "marbles", "pink_room", "subway", "zeroday"]

class DatasetBenchmarker:
    def __init__(self, quick_mode=False, vsync=True, borderless=False):
        self.quick_mode = quick_mode
        self.vsync = vsync
        self.borderless = borderless
        self.videos = {}
        self.scan_dataset()

    def scan_dataset(self):
        for scene in SCENES:
            self.videos[scene] = []
            
        if not DATASET_DIR.exists():
            print(f"Error: Dataset directory not found: {DATASET_DIR}")
            sys.exit(1)
            
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

    def preload_video(self, path, max_frames=None):
        cap = cv2.VideoCapture(path)
        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
            if max_frames and count >= max_frames:
                break
        cap.release()
        return frames

    def create_isolated_window(self, title):
        monitor = glfw.get_primary_monitor()
        fullscreen = not self.borderless and ("--windowed" not in sys.argv)
        
        if fullscreen and monitor:
            mode = glfw.get_video_mode(monitor)
            if hasattr(mode, 'size'):
                mon_w, mon_h = mode.size.width, mode.size.height
            elif hasattr(mode, 'width'):
                mon_w, mon_h = mode.width, mode.height
            else:
                mon_w, mon_h = mode[0], mode[1]
        else:
            mon_w, mon_h = 1280, 720
            monitor = None
            
        glfw.default_window_hints()
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
        
        if self.borderless and monitor:
            glfw.window_hint(glfw.DECORATED, glfw.FALSE)
            window = glfw.create_window(mon_w, mon_h, title, None, None)
            if window:
                glfw.set_window_pos(window, 0, 0)
        elif fullscreen and monitor:
            window = glfw.create_window(mon_w, mon_h, title, monitor, None)
            if not window:
                glfw.window_hint(glfw.DECORATED, glfw.FALSE)
                window = glfw.create_window(mon_w, mon_h, title, None, None)
                if window:
                    glfw.set_window_pos(window, 0, 0)
        else:
            window = glfw.create_window(1280, 720, title, None, None)
            
        if not window:
            print(f"Failed to create GLFW window for {title}")
            return None
            
        glfw.make_context_current(window)
        glfw.show_window(window)
        glfw.poll_events()
        glfw.swap_interval(1 if self.vsync else 0)
        
        # Setup Textures
        glEnable(GL_TEXTURE_2D)
        tex_a, tex_ref, tex_c = glGenTextures(3)
        for tex in [tex_a, tex_ref, tex_c]:
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1280, 720, 0, GL_BGR, GL_UNSIGNED_BYTE, None)
            
        return window, tex_a, tex_ref, tex_c

    def upload_texture(self, tex_id, frame):
        h, w = frame.shape[:2]
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, w, h, GL_BGR, GL_UNSIGNED_BYTE, frame.data)

    def draw_quad(self, x1, y1, x2, y2):
        glBegin(GL_QUADS)
        glTexCoord2f(0.0, 1.0); glVertex2f(x1, y1)
        glTexCoord2f(1.0, 1.0); glVertex2f(x2, y1)
        glTexCoord2f(1.0, 0.0); glVertex2f(x2, y2)
        glTexCoord2f(0.0, 0.0); glVertex2f(x1, y2)
        glEnd()

    def render_pyramid(self, window, tex_a, tex_ref, tex_c, frame_a, frame_ref, frame_c):
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
        
        self.upload_texture(tex_a, frame_a)
        self.upload_texture(tex_ref, frame_ref)
        self.upload_texture(tex_c, frame_c)
        
        # Reference (top-center)
        glBindTexture(GL_TEXTURE_2D, tex_ref)
        self.draw_quad(-0.5, 0.0, 0.5, 1.0)
        
        # Left (A)
        glBindTexture(GL_TEXTURE_2D, self.tex_a if hasattr(self, 'tex_a') else tex_a, frame_a)
        # Note: fix texture binding
        glBindTexture(GL_TEXTURE_2D, tex_a)
        self.draw_quad(-1.0, -1.0, 0.0, 0.0)
        
        # Right (C)
        glBindTexture(GL_TEXTURE_2D, tex_c)
        self.draw_quad(0.0, -1.0, 1.0, 0.0)

    def run_single_isolated_test(self, scene, vid_a, vid_ref, vid_c):
        # 1. PRELOAD PHASE into RAM
        max_f = 120 if self.quick_mode else None
        frames_a = self.preload_video(vid_a["path"], max_frames=max_f)
        frames_ref = self.preload_video(vid_ref["path"], max_frames=max_f)
        frames_c = self.preload_video(vid_c["path"], max_frames=max_f)
        
        total_frames = max(len(frames_a), len(frames_ref), len(frames_c))
        
        # 2. CREATE ISOLATED WINDOW & OPENGL CONTEXT
        window_title = f"GAIM240 Profiler | {scene.upper()} - {vid_a['metric']}"
        res_win = self.create_isolated_window(window_title)
        if not res_win:
            return None
        window, tex_a, tex_ref, tex_c = res_win
        
        # Warm-up rendering phase (60 frames) to let X11 window unredirection & GPU driver pipelines settle
        warmup_frames = 60
        for step in range(warmup_frames):
            fa = frames_a[step % len(frames_a)]
            fref = frames_ref[step % len(frames_ref)]
            fc = frames_c[step % len(frames_c)]
            
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
            
            self.upload_texture(tex_a, fa)
            self.upload_texture(tex_ref, fref)
            self.upload_texture(tex_c, fc)
            
            glBindTexture(GL_TEXTURE_2D, tex_ref)
            self.draw_quad(-0.5, 0.0, 0.5, 1.0)
            glBindTexture(GL_TEXTURE_2D, tex_a)
            self.draw_quad(-1.0, -1.0, 0.0, 0.0)
            glBindTexture(GL_TEXTURE_2D, tex_c)
            self.draw_quad(0.0, -1.0, 1.0, 0.0)
            
            glfw.swap_buffers(window)
            glfw.poll_events()

        get_img_durations = []
        upload_draw_durations = []
        flip_durations = []
        loop_durations = []
        swap_timestamps = []
        
        # 3. BENCHMARK RENDER LOOP (Recorded)
        for step in range(total_frames):
            t0 = time.perf_counter()
            
            fa = frames_a[step % len(frames_a)]
            fref = frames_ref[step % len(frames_ref)]
            fc = frames_c[step % len(frames_c)]
            
            t1 = time.perf_counter()
            
            # Render
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
            
            self.upload_texture(tex_a, fa)
            self.upload_texture(tex_ref, fref)
            self.upload_texture(tex_c, fc)
            
            glBindTexture(GL_TEXTURE_2D, tex_ref)
            self.draw_quad(-0.5, 0.0, 0.5, 1.0)
            glBindTexture(GL_TEXTURE_2D, tex_a)
            self.draw_quad(-1.0, -1.0, 0.0, 0.0)
            glBindTexture(GL_TEXTURE_2D, tex_c)
            self.draw_quad(0.0, -1.0, 1.0, 0.0)
            
            t2 = time.perf_counter()
            
            glfw.swap_buffers(window)
            t3 = time.perf_counter()
            
            glfw.poll_events()
            if glfw.window_should_close(window) or glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.destroy_window(window)
                return None
                
            get_img_durations.append((t1 - t0) * 1000.0)
            upload_draw_durations.append((t2 - t1) * 1000.0)
            flip_durations.append((t3 - t2) * 1000.0)
            loop_durations.append((t3 - t0) * 1000.0)
            swap_timestamps.append(t3)
            
        # Destroy window immediately after test finishes
        glfw.destroy_window(window)
        
        # 4. STATISTICAL SUMMARY & CSV LOGGING
        timestamps = np.array(swap_timestamps)
        intervals_ms = np.diff(timestamps) * 1000.0 if len(timestamps) > 1 else np.array([0.0])
        
        mean_int = np.mean(intervals_ms) if len(intervals_ms) > 0 else 0.0
        std_int = np.std(intervals_ms) if len(intervals_ms) > 0 else 0.0
        min_int = np.min(intervals_ms) if len(intervals_ms) > 0 else 0.0
        max_int = np.max(intervals_ms) if len(intervals_ms) > 0 else 0.0
        
        dropped_count = np.sum(intervals_ms > (4.167 * 1.5)) if len(intervals_ms) > 0 else 0
        dropped_pct = (dropped_count / len(intervals_ms)) * 100.0 if len(intervals_ms) > 0 else 0.0
        actual_fps = 1000.0 / mean_int if mean_int > 0 else 0.0

        # Save individual raw profile CSV for this test
        logs_dir = Path("profile_logs")
        logs_dir.mkdir(exist_ok=True)
        vsync_suffix = "vsync_on" if self.vsync else "vsync_off"
        raw_csv_path = logs_dir / f"profile_{scene}_{vid_a['metric']}_{vsync_suffix}.csv"
        
        with open(raw_csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "FrameIndex", 
                "GetImageDuration_ms", 
                "UploadDrawDuration_ms", 
                "FlipDuration_ms", 
                "TotalLoopDuration_ms", 
                "SwapTimestamp_sec", 
                "InterFrameInterval_ms"
            ])
            writer.writerow([
                0,
                f"{get_img_durations[0]:.6f}",
                f"{upload_draw_durations[0]:.6f}",
                f"{flip_durations[0]:.6f}",
                f"{loop_durations[0]:.6f}",
                f"{swap_timestamps[0]:.6f}",
                "N/A"
            ])
            for idx in range(1, len(swap_timestamps)):
                writer.writerow([
                    idx,
                    f"{get_img_durations[idx]:.6f}",
                    f"{upload_draw_durations[idx]:.6f}",
                    f"{flip_durations[idx]:.6f}",
                    f"{loop_durations[idx]:.6f}",
                    f"{swap_timestamps[idx]:.6f}",
                    f"{intervals_ms[idx-1]:.6f}"
                ])
                
        # 5. DOWNTIME & CLEANUP
        del frames_a, frames_ref, frames_c
        gc.collect()
        time.sleep(0.2) # Short downtime between tests
        
        return {
            "Scene": scene,
            "Metric": vid_a["metric"],
            "LeftLevel": vid_a["level"],
            "RightLevel": vid_c["level"],
            "VSync": "ENABLED" if self.vsync else "DISABLED",
            "TotalFrames": len(swap_timestamps),
            "ActualFPS": round(actual_fps, 2),
            "MeanInterval_ms": round(mean_int, 4),
            "Jitter_ms": round(std_int, 4),
            "MinInterval_ms": round(min_int, 4),
            "MaxInterval_ms": round(max_int, 4),
            "DroppedFrames": dropped_count,
            "DroppedPercentage": round(dropped_pct, 2),
            "MeanRAMFetch_ms": round(np.mean(get_img_durations), 4),
            "MeanGPUDraw_ms": round(np.mean(upload_draw_durations), 4),
            "MeanVSyncBlock_ms": round(np.mean(flip_durations), 4),
        }

    def run_full_suite(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            sys.exit(1)
            
        results = []
        
        total_tests = 0
        for scene in SCENES:
            vids = self.videos[scene]
            non_ref = [v for v in vids if v["metric"] != "reference"]
            metrics = list(dict.fromkeys([v["metric"] for v in non_ref]))
            total_tests += len(metrics)
            
        print("\n" + "="*65)
        print(f"      GAIM240 DATASET COMPREHENSIVE BENCHMARK SUITE       ")
        print("="*65)
        print(f"Mode: {'Quick Dry Run (120 frames/test)' if self.quick_mode else 'Full Test (1200 frames/test)'}")
        print(f"VSync: {'ENABLED' if self.vsync else 'DISABLED'}")
        print(f"Total Test Configurations: {total_tests}")
        print("="*65 + "\n")
        
        test_counter = 0
        for scene in SCENES:
            vids = self.videos[scene]
            if not vids:
                continue
                
            vid_ref = next((v for v in vids if v["metric"] == "reference"), vids[0])
            non_ref = [v for v in vids if v["metric"] != "reference"]
            metrics = list(dict.fromkeys([v["metric"] for v in non_ref]))
            
            for metric in metrics:
                test_counter += 1
                vid_a = next((v for v in non_ref if v["metric"] == metric and v["level"] == "level1"), non_ref[0])
                vid_c = next((v for v in non_ref if v["metric"] == metric and v["level"] == "level2"), non_ref[0])
                
                print(f"[{test_counter}/{total_tests}] Preloading & Testing {scene.upper()} | Metric: {metric} (VSync: {'ON' if self.vsync else 'OFF'})...", end="", flush=True)
                
                t_start = time.perf_counter()
                res = self.run_single_isolated_test(scene, vid_a, vid_ref, vid_c)
                t_elapsed = time.perf_counter() - t_start
                
                if res is None:
                    print(" Aborted by user.")
                    break
                    
                results.append(res)
                print(f" Done ({t_elapsed:.1f}s) -> {res['ActualFPS']} FPS | Jitter: {res['Jitter_ms']}ms | Drops: {res['DroppedFrames']}")
                
        glfw.terminate()
        return results

def get_summary_filename():
    tags = []
    if "--quick" in sys.argv:
        tags.append("quick")
    else:
        tags.append("full")
        
    if "--compare-vsync" in sys.argv:
        tags.append("compare-vsync")
    elif "--no-vsync" in sys.argv:
        tags.append("no-vsync")
    else:
        tags.append("vsync")
        
    if "--borderless" in sys.argv:
        tags.append("borderless")
    elif "--windowed" in sys.argv:
        tags.append("windowed")
    else:
        tags.append("fullscreen")
        
    tag_str = "_".join(tags)
    return f"benchmark_summary_{tag_str}.csv"

if __name__ == "__main__":
    quick = "--quick" in sys.argv
    no_vsync = "--no-vsync" in sys.argv
    borderless = "--borderless" in sys.argv
    compare_vsync = "--compare-vsync" in sys.argv
    
    summary_filename = get_summary_filename()
    
    if compare_vsync:
        print(f"\n--- Running Dual VSync Comparison Suite (Pass 1: VSync ON | Pass 2: VSync OFF) ---")
        bench1 = DatasetBenchmarker(quick_mode=quick, vsync=True, borderless=borderless)
        res1 = bench1.run_full_suite()
        
        bench2 = DatasetBenchmarker(quick_mode=quick, vsync=False, borderless=borderless)
        res2 = bench2.run_full_suite()
        
        all_res = (res1 or []) + (res2 or [])
        if all_res:
            df_res = pd.DataFrame(all_res)
            df_res.to_csv(summary_filename, index=False)
            print("\n" + "="*65)
            print(f"Dual VSync Comparison Complete! Summary saved to: {Path(summary_filename).resolve()}")
            print("="*65)
    else:
        benchmarker = DatasetBenchmarker(quick_mode=quick, vsync=not no_vsync, borderless=borderless)
        res = benchmarker.run_full_suite()
        if res:
            df_res = pd.DataFrame(res)
            df_res.to_csv(summary_filename, index=False)
            print("\n" + "="*65)
            print(f"Benchmark suite finished! Summary saved to: {Path(summary_filename).resolve()}")
            print("="*65)
            print("\nTop Performance Highlights:")
            print(f"  Overall Mean Frame Rate: {df_res['ActualFPS'].mean():.2f} FPS")
            print(f"  Overall Mean Jitter    : {df_res['Jitter_ms'].mean():.4f} ms")
            print(f"  Total Dropped Frames   : {df_res['DroppedFrames'].sum()}")
            print(f"  GPU Draw Overhead Mean : {df_res['MeanGPUDraw_ms'].mean():.4f} ms")
            print("="*65)
