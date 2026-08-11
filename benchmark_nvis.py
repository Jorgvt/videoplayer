#!/usr/bin/env python3
import os
import sys
import csv
import time
import argparse
import subprocess
import queue
import threading
from pathlib import Path

# Paths to the player and bank
NVIS_BIN = Path("../userstudy_v0.2_linux/userstudy_v0.2/userstudy_patched").resolve()
BANK_PATH = Path("all_trials_bank.csv").resolve()
DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
RUST_LIB_DIR = Path("rust_player/lib").resolve()

SCENES = ["attic", "bistro_exterior", "bistro_interior", "classroom", "landscape", "marbles", "pink_room", "subway", "zeroday"]

def enqueue_output(out, q):
    try:
        for line in iter(out.readline, ''):
            q.put(line)
    except Exception:
        pass
    finally:
        out.close()

def load_balanced_trials(bank_path, count_per_scene=8):
    """
    Loads master trials from bank and selects count_per_scene trials for each of the 9 scenes.
    """
    if not bank_path.exists():
        print(f"Error: Master trials bank not found at {bank_path}")
        sys.exit(1)
        
    scene_buckets = {scene: [] for scene in SCENES}
    with open(bank_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scene = row.get("Scene", "").lower()
            if scene in scene_buckets:
                scene_buckets[scene].append(row)
                
    selected_trials = []
    for scene, rows in scene_buckets.items():
        if len(rows) < count_per_scene:
            print(f"Warning: Scene {scene} only has {len(rows)} trials in the bank (needed {count_per_scene}). Taking all.")
            selected_trials.extend(rows)
        else:
            selected_trials.extend(rows[:count_per_scene])
            
    return selected_trials

def run_trial(trial_idx, total_trials, trial, vsync_mode, playback_fps):
    scene = trial["Scene"]
    print(f"\n=================================================================")
    print(f"  TRIAL [{trial_idx}/{total_trials}] | SCENE: {scene.upper()}")
    print(f"  Comparison: {trial['ComparisonType']}")
    print(f"=================================================================")
    
    # Resolve absolute paths
    left_path = str(DATASET_DIR / Path(trial["Vid1_Path"]).name)
    ref_path = str(DATASET_DIR / Path(trial["RefPath"]).name)
    right_path = str(DATASET_DIR / Path(trial["Vid2_Path"]).name)
    
    # Verify files exist
    for p in [left_path, ref_path, right_path]:
        if not os.path.exists(p):
            print(f"  Error: File not found: {p}")
            return None

    # Set up environment variables
    env = os.environ.copy()
    wrapper_bin_dir = str(Path(__file__).parent.resolve() / "bin")
    env["PATH"] = wrapper_bin_dir + ":" + str(RUST_LIB_DIR / "usr" / "bin") + ":" + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = str(RUST_LIB_DIR / "usr" / "lib" / "x86_64-linux-gnu") + ":" + env.get("LD_LIBRARY_PATH", "")
    env["__GL_SHOW_GRAPHICS_OSD"] = "1"

    # Nvidia player command
    cmd = [
        "stdbuf", "-oL", "-eL",
        str(NVIS_BIN),
        "-f", left_path,
        "-f", ref_path,
        "-f", right_path,
        "--vsync", vsync_mode,
        "--fps", str(playback_fps)
    ]

    print("  Pre-decoding video streams in background...", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    
    # Asynchronous stdout queue reader
    q = queue.Queue()
    t = threading.Thread(target=enqueue_output, args=(proc.stdout, q))
    t.daemon = True
    t.start()
    
    decoded_sequences_count = 0
    predecode_time = None
    
    # Wait until all 3 streams are loaded
    while proc.poll() is None:
        time.sleep(0.05)
        while not q.empty():
            try:
                line = q.get_nowait()
                if "Decoded image sequence" in line:
                    decoded_sequences_count += 1
            except queue.Empty:
                break
        
        if decoded_sequences_count >= 3 and predecode_time is None:
            predecode_time = time.time() - t0
            print(f"  --> Pre-decoding complete in {predecode_time:.2f} seconds.")
            print("  --> Nvidia Visualizer window is now active on your monitor!")
            print("      [Press SPACE to play, F2 to show GUI panel for actual FPS, Esc to close]")
            break

    # Block until the window is closed
    proc.wait()
    
    print("\n-----------------------------------------------------------------")
    print("  Visualizer closed. Please enter the performance metrics below:")
    print("-----------------------------------------------------------------")
    
    # Prompt user for the observed playback FPS
    while True:
        try:
            fps_str = input("  Enter the FPS observed in the GUI panel (default 240.0): ").strip()
            if not fps_str:
                fps = 240.0
                break
            fps = float(fps_str)
            if fps > 0:
                break
            print("  FPS must be a positive number!")
        except ValueError:
            print("  Please enter a valid numeric value.")
            
    return {
        "TrialIndex": trial_idx,
        "MasterTrialID": trial["MasterTrialID"],
        "Scene": scene,
        "ComparisonType": trial["ComparisonType"],
        "LeftVideo": Path(trial["Vid1_Path"]).name,
        "RightVideo": Path(trial["Vid2_Path"]).name,
        "VSync": vsync_mode,
        "PreDecodeTime_sec": round(predecode_time if predecode_time else (time.time() - t0), 4),
        "PlaybackFPS": fps
    }

def main():
    parser = argparse.ArgumentParser(description="GAIM240 Nvidia Visualizer Benchmark Suite")
    parser.add_argument("--vsync", type=str, choices=["on", "off", "mailbox"], default="off",
                        help="VSync configuration preference (default: off for uncapped FPS evaluation)")
    parser.add_argument("--fps", type=int, default=240, help="Target playback frame rate (default: 240)")
    parser.add_argument("--subject", type=str, default="BENCHMARK", help="Subject identifier for results")
    args = parser.parse_args()

    if not NVIS_BIN.exists():
        print(f"Error: Nvidia Visualizer executable not found at {NVIS_BIN}")
        print("Please verify the patched elf binary exists.")
        sys.exit(1)

    print("=================================================================")
    print("        GAIM240 NVIDIA VISUALIZER BENCHMARK SUITE")
    print("=================================================================")
    print(f"Visualizer Binary : {NVIS_BIN}")
    print(f"VSync Preference  : {args.vsync.upper()}")
    print("Benchmark Scope   : 72 trials (8 balanced trials per scene)")
    print("Output Log        : experiment_results/benchmark_nvis_results.csv")
    print("=================================================================")
    
    trials = load_balanced_trials(BANK_PATH, count_per_scene=8)
    total_trials = len(trials)
    
    os.makedirs("experiment_results", exist_ok=True)
    csv_out_path = Path("experiment_results") / f"{args.subject}_nvis_benchmark.csv"
    
    results = []
    try:
        for idx, trial in enumerate(trials, 1):
            res = run_trial(idx, total_trials, trial, args.vsync, args.fps)
            if res is None:
                print("Skipping trial due to error.")
                continue
            results.append(res)
            
            # Save progress incrementally in case of user abort
            with open(csv_out_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=res.keys())
                writer.writeheader()
                writer.writerows(results)
                
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user. Progress saved.")
        
    if not results:
        print("Keep results empty.")
        return
        
    # Calculate performance stats
    avg_load = sum(r["PreDecodeTime_sec"] for r in results) / len(results)
    avg_fps = sum(r["PlaybackFPS"] for r in results) / len(results)
    
    print("\n=================================================================")
    print("          BENCHMARK PERFORMANCE COMPLETE SUMMARY")
    print("=================================================================")
    print(f"Total Trials Runs        : {len(results)} out of 72")
    print(f"Average Pre-Decode Time  : {avg_load:.2f} seconds")
    print(f"Average Playback FPS     : {avg_fps:.2f} FPS")
    print(f"Detailed Logs Exported   : {csv_out_path.resolve()}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
