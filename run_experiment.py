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
import random
import gc
import glfw
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from OpenGL.GL import *

from platform_utils import get_dataset_dir

DATASET_DIR = get_dataset_dir()
SCENES = ["attic", "bistro_exterior", "bistro_interior", "classroom", "landscape", "marbles", "pink_room", "subway", "zeroday"]
TRIAL_BANK_CSV = Path("all_trials_bank.csv")

def scan_dataset():
    videos = {scene: [] for scene in SCENES}
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
            videos[scene].append({
                "filename": file,
                "path": str(DATASET_DIR / file),
                "metric": "reference",
                "level": None
            })
        else:
            match = re.match(r"^(.*?)(?:_level([0-2]))?$", rest)
            if match:
                metric = match.group(1)
                lvl_num = match.group(2)
                level = f"level{lvl_num}" if lvl_num is not None else None
                videos[scene].append({
                    "filename": file,
                    "path": str(DATASET_DIR / file),
                    "metric": metric,
                    "level": level
                })
    return videos

def generate_master_trial_bank():
    """Scans dataset and generates all_trials_bank.csv containing all possible 2AFC pairings."""
    import itertools
    videos = scan_dataset()
    raw_bank = []
    trial_counter = 0
    
    for scene in SCENES:
        vids = videos[scene]
        if not vids:
            continue
        ref_vid = next((v for v in vids if v["metric"] == "reference"), None)
        if not ref_vid:
            continue
            
        non_ref = [v for v in vids if v["metric"] != "reference"]
        non_ref = sorted(non_ref, key=lambda x: (x["metric"], x["level"]))
        
        pairs = list(itertools.combinations(non_ref, 2))
        for v1, v2 in pairs:
            trial_counter += 1
            comp_type = "INTRA-DISTORTION" if v1["metric"] == v2["metric"] else "INTER-DISTORTION"
            raw_bank.append({
                "MasterTrialID": trial_counter,
                "Scene": scene,
                "ComparisonType": comp_type,
                "RefFilename": ref_vid["filename"],
                "RefPath": ref_vid["path"],
                "Vid1_Filename": v1["filename"],
                "Vid1_Metric": v1["metric"],
                "Vid1_Level": v1["level"],
                "Vid1_Path": v1["path"],
                "Vid2_Filename": v2["filename"],
                "Vid2_Metric": v2["metric"],
                "Vid2_Level": v2["level"],
                "Vid2_Path": v2["path"]
            })
                    
    df_bank = pd.DataFrame(raw_bank)
    df_bank.to_csv(TRIAL_BANK_CSV, index=False)
    print(f"Master Trial Bank generated with {len(df_bank)} unique 2AFC pairings -> {TRIAL_BANK_CSV.resolve()}")
    return df_bank

class PerceptionExperiment:
    def __init__(self, subject_id, exp_mode="full", target_scenes=None, quick_mode=False, vsync=True, borderless=False):
        self.subject_id = subject_id
        self.exp_mode = exp_mode.lower()
        self.target_scenes = target_scenes if target_scenes else SCENES
        self.quick_mode = quick_mode
        self.vsync = vsync
        self.borderless = borderless
        
        self.trials = []
        self.load_and_shuffle_trials()
        
        self.results = []
        self.current_trial_choice = None
        self.quit_requested = False

    def load_and_shuffle_trials(self):
        """Loads master trial bank CSV, filters by mode/scenes, shuffles order and counterbalances sides."""
        if not TRIAL_BANK_CSV.exists():
            print(f"Generating master trial bank CSV...")
            df_bank = generate_master_trial_bank()
        else:
            df_bank = pd.read_csv(TRIAL_BANK_CSV)
            
        df_bank = df_bank[df_bank["Scene"].isin(self.target_scenes)].copy()
        
        if self.exp_mode == "intra":
            df_bank = df_bank[df_bank["ComparisonType"] == "INTRA-DISTORTION"].copy()
        elif self.exp_mode == "inter":
            df_bank = df_bank[df_bank["ComparisonType"] == "INTER-DISTORTION"].copy()
            
        records = df_bank.to_dict("records")
        
        random.seed(hash(self.subject_id))
        random.shuffle(records)
        
        for i, row in enumerate(records):
            flip_side = random.choice([True, False])
            
            left_vid = {
                "filename": row["Vid2_Filename"] if flip_side else row["Vid1_Filename"],
                "metric": row["Vid2_Metric"] if flip_side else row["Vid1_Metric"],
                "level": row["Vid2_Level"] if flip_side else row["Vid1_Level"],
                "path": row["Vid2_Path"] if flip_side else row["Vid1_Path"]
            }
            right_vid = {
                "filename": row["Vid1_Filename"] if flip_side else row["Vid2_Filename"],
                "metric": row["Vid1_Metric"] if flip_side else row["Vid2_Metric"],
                "level": row["Vid1_Level"] if flip_side else row["Vid2_Level"],
                "path": row["Vid1_Path"] if flip_side else row["Vid2_Path"]
            }
            ref_vid = {
                "filename": row["RefFilename"],
                "path": row["RefPath"],
                "metric": "reference",
                "level": None
            }
            
            self.trials.append({
                "trial_idx": i + 1,
                "master_trial_id": row["MasterTrialID"],
                "scene": row["Scene"],
                "comparison_type": row["ComparisonType"],
                "ref_vid": ref_vid,
                "left_vid": left_vid,
                "right_vid": right_vid
            })

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
            return None
            
        glfw.make_context_current(window)
        glfw.show_window(window)
        glfw.poll_events()
        glfw.swap_interval(1 if self.vsync else 0)
        
        glfw.set_key_callback(window, self.on_key)
        
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

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key in (glfw.KEY_LEFT, glfw.KEY_A, glfw.KEY_1, glfw.KEY_KP_1):
                self.current_trial_choice = "LEFT"
            elif key in (glfw.KEY_RIGHT, glfw.KEY_D, glfw.KEY_2, glfw.KEY_KP_2):
                self.current_trial_choice = "RIGHT"
            elif key in (glfw.KEY_ESCAPE, glfw.KEY_Q):
                self.quit_requested = True

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
        
        glBindTexture(GL_TEXTURE_2D, tex_ref)
        self.draw_quad(-0.5, 0.0, 0.5, 1.0)
        
        glBindTexture(GL_TEXTURE_2D, tex_a)
        self.draw_quad(-1.0, -1.0, 0.0, 0.0)
        
        glBindTexture(GL_TEXTURE_2D, tex_c)
        self.draw_quad(0.0, -1.0, 1.0, 0.0)

    def run_trial(self, trial_info):
        self.current_trial_choice = None
        
        # On-the-fly Video Streaming (ZERO startup preloading delay!)
        cap_left = cv2.VideoCapture(trial_info["left_vid"]["path"])
        cap_ref = cv2.VideoCapture(trial_info["ref_vid"]["path"])
        cap_right = cv2.VideoCapture(trial_info["right_vid"]["path"])
        
        title = f"Perception Experiment | Subject: {self.subject_id} | Trial {trial_info['trial_idx']}/{len(self.trials)}"
        res_win = self.create_isolated_window(title)
        if not res_win:
            cap_left.release()
            cap_ref.release()
            cap_right.release()
            return None
        window, tex_a, tex_ref, tex_c = res_win
        
        # Read Frame 0 for static 60-frame warm-up (X11 unredirection & 240Hz sync)
        ret_l, fa_0 = cap_left.read()
        ret_r, fref_0 = cap_ref.read()
        ret_c, fc_0 = cap_right.read()
        
        if not (ret_l and ret_r and ret_c):
            print("Error reading video streams.")
            glfw.destroy_window(window)
            return None
            
        for _ in range(60):
            self.render_pyramid(window, tex_a, tex_ref, tex_c, fa_0, fref_0, fc_0)
            glfw.swap_buffers(window)
            glfw.poll_events()
            
        start_time = time.perf_counter()
        swap_timestamps = []
        
        # On-the-fly 240Hz streaming loop
        while not self.current_trial_choice and not self.quit_requested:
            ret_l, fa = cap_left.read()
            ret_r, fref = cap_ref.read()
            ret_c, fc = cap_right.read()
            
            # Loop seamless when reaching end of video
            if not ret_l or fa is None:
                cap_left.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, fa = cap_left.read()
            if not ret_r or fref is None:
                cap_ref.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, fref = cap_ref.read()
            if not ret_c or fc is None:
                cap_right.set(cv2.CAP_PROP_POS_FRAMES, 0)
                _, fc = cap_right.read()
                
            self.render_pyramid(window, tex_a, tex_ref, tex_c, fa, fref, fc)
            
            glfw.swap_buffers(window)
            t_swap = time.perf_counter()
            swap_timestamps.append(t_swap)
            
            glfw.poll_events()
            if glfw.window_should_close(window):
                self.quit_requested = True
                break
                
        response_time = time.perf_counter() - start_time
        choice = self.current_trial_choice
        
        cap_left.release()
        cap_ref.release()
        cap_right.release()
        glfw.destroy_window(window)
        
        timestamps = np.array(swap_timestamps)
        intervals_ms = np.diff(timestamps) * 1000.0 if len(timestamps) > 1 else np.array([0.0])
        mean_int = np.mean(intervals_ms) if len(intervals_ms) > 0 else 0.0
        actual_fps = 1000.0 / mean_int if mean_int > 0 else 0.0
        
        gc.collect()
        time.sleep(0.1)
        
        if self.quit_requested or not choice:
            return None
            
        chosen_vid = trial_info["left_vid"] if choice == "LEFT" else trial_info["right_vid"]
        rejected_vid = trial_info["right_vid"] if choice == "LEFT" else trial_info["left_vid"]
        
        return {
            "SubjectID": self.subject_id,
            "TrialIndex": trial_info["trial_idx"],
            "MasterTrialID": trial_info["master_trial_id"],
            "TotalTrials": len(self.trials),
            "ComparisonType": trial_info["comparison_type"],
            "Scene": trial_info["scene"],
            "LeftMetric": trial_info["left_vid"]["metric"],
            "LeftLevel": trial_info["left_vid"]["level"],
            "LeftVideo": trial_info["left_vid"]["filename"],
            "RightMetric": trial_info["right_vid"]["metric"],
            "RightLevel": trial_info["right_vid"]["level"],
            "RightVideo": trial_info["right_vid"]["filename"],
            "ChosenSide": choice,
            "ChosenMetric": chosen_vid["metric"],
            "ChosenLevel": chosen_vid["level"],
            "ChosenVideo": chosen_vid["filename"],
            "RejectedMetric": rejected_vid["metric"],
            "RejectedLevel": rejected_vid["level"],
            "RejectedVideo": rejected_vid["filename"],
            "ResponseTime_sec": round(response_time, 4),
            "PresentationFPS": round(actual_fps, 2)
        }

    def run_experiment(self):
        if not glfw.init():
            print("Failed to initialize GLFW")
            sys.exit(1)
            
        out_dir = Path("experiment_results")
        out_dir.mkdir(exist_ok=True)
        csv_file = out_dir / f"experiment_{self.subject_id}.csv"
        
        completed_master_ids = set()
        if csv_file.exists():
            try:
                df_existing = pd.read_csv(csv_file)
                if "MasterTrialID" in df_existing.columns:
                    completed_master_ids = set(df_existing["MasterTrialID"].dropna().astype(int))
                    self.results = df_existing.to_dict("records")
            except Exception as e:
                print(f"Warning: Could not read existing file {csv_file}: {e}")
                
        active_trials = [t for t in self.trials if t["master_trial_id"] not in completed_master_ids]
        if self.quick_mode:
            active_trials = active_trials[:10]
        
        print("\n" + "="*65)
        print(f"      GAIM240 HUMAN VISUAL PERCEPTION EXPERIMENT SUITE       ")
        print("="*65)
        print(f"Participant ID  : {self.subject_id}")
        print(f"Trial Bank      : {TRIAL_BANK_CSV}")
        print(f"Total Bank Size : {len(self.trials)}")
        if completed_master_ids:
            print(f"Status          : RESUME MODE ({len(completed_master_ids)} completed, {len(active_trials)} remaining)")
        else:
            print(f"Status          : NEW SESSION ({len(active_trials)} trials to present)")
        print(f"Experiment Mode : {self.exp_mode.upper()}")
        print(f"VSync           : {'ENABLED (240Hz Locked)' if self.vsync else 'DISABLED'}")
        print("="*65)
        
        if not active_trials:
            print(f"\nAll {len(self.trials)} trials for subject '{self.subject_id}' are already completed!")
            print(f"Results file is complete: {csv_file.resolve()}")
            glfw.terminate()
            return

        print("\nINSTRUCTIONS FOR PARTICIPANT:")
        print("  1. Look at the TOP (CENTER) video reference.")
        print("  2. Compare the BOTTOM LEFT vs BOTTOM RIGHT distorted videos.")
        print("  3. Choose the distorted video that has HIGHER visual quality:")
        print("     - Press [LEFT ARROW] or [A] or [1] for Left Video.")
        print("     - Press [RIGHT ARROW] or [D] or [2] for Right Video.")
        print("  4. Press [ESC] or [Q] at any time to save and quit.")
        print("="*65 + "\n")
        
        for idx_count, trial in enumerate(active_trials, 1):
            print(f"Executing Trial [{idx_count}/{len(active_trials)}] (Master ID: #{trial['master_trial_id']}) | Scene: {trial['scene'].upper()}...", end="", flush=True)
            
            res = self.run_trial(trial)
            if res is None:
                print("\nExperiment stopped early by participant/user. Progress saved.")
                break
                
            self.results.append(res)
            
            df = pd.DataFrame(self.results)
            df.to_csv(csv_file, index=False)
            
            print(f" Chose {res['ChosenSide']} ({res['ChosenMetric']} {res['ChosenLevel']}) in {res['ResponseTime_sec']}s (FPS: {res['PresentationFPS']})")
            
        glfw.terminate()
        
        if self.results:
            df = pd.DataFrame(self.results)
            df.to_csv(csv_file, index=False)
            print("\n" + "="*65)
            print(f"Subject Session Complete! Total completed trials: {len(df)} -> Saved to: {csv_file.resolve()}")
            print("="*65)

if __name__ == "__main__":
    if "--generate-bank" in sys.argv:
        generate_master_trial_bank()
        sys.exit(0)
        
    subject_id = "P01"
    exp_mode = "full"
    target_scenes = []
    
    for arg in sys.argv[1:]:
        if arg.startswith("--subject=") or arg.startswith("--participant="):
            subject_id = arg.split("=")[1]
        elif arg.startswith("--mode="):
            exp_mode = arg.split("=")[1]
        elif arg.startswith("--scenes="):
            target_scenes = arg.split("=")[1].split(",")
        elif not arg.startswith("--") and arg != sys.argv[0]:
            subject_id = arg
            
    quick = "--quick" in sys.argv
    no_vsync = "--no-vsync" in sys.argv
    borderless = "--borderless" in sys.argv
    
    exp = PerceptionExperiment(
        subject_id=subject_id, 
        exp_mode=exp_mode,
        target_scenes=target_scenes if target_scenes else SCENES,
        quick_mode=quick, 
        vsync=not no_vsync, 
        borderless=borderless
    )
    exp.run_experiment()
