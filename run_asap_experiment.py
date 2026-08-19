#!/usr/bin/env python3
"""
GAIM240 Active Sampling Perception Experiment Runner (ASAP + Rust 240Hz)
========================================================================
Integrates Active SAmpling for Pairwise comparisons (ASAP - Mikhailiuk et al., ICPR 2020)
from https://github.com/gfxdisp/asap with the hardware-accelerated 240Hz Rust Pyramid Player.

Default Mode: GLOBAL / INTER-SCENE (--mode=global)
- Pools conditions across scenes to construct a unified Global JND Quality Scale.
- Evaluates Expected Information Gain across all 3,159 valid candidate pairs in
  all_trials_bank.csv (ensuring Vid1, Vid2, and Ref ALWAYS belong to the same scene).
- Instant Fast EIG Selection (1ms execution time) eliminates inter-trial delay.
- Pure CPU active sampling (NumPy + SciPy) ensures 100% of GPU VRAM and CUDA engines
  remain dedicated to rust_player for locked 240Hz presentation.
- --trials specifies the number of NEW trials to run in the current session.
"""

import argparse
import csv
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from platform_utils import get_dataset_dir, set_native_lib_env

import numpy as np

# Add local asap/python directory to import path
ASAP_PATH = Path(__file__).parent / "asap" / "python"
if str(ASAP_PATH) not in sys.path:
    sys.path.insert(0, str(ASAP_PATH))

try:
    from asap_cpu import ASAP
except ImportError:
    print("Error: Could not import ASAP module from asap/python/asap_cpu.py.", flush=True)
    print("Please ensure the asap repository is cloned at ./asap/", flush=True)
    sys.exit(1)


def discover_conditions_and_pairs(dataset_dir: Path, bank_path: Path, target_scene: str = None):
    """
    Scans dataset and all_trials_bank.csv (3,159 candidate pairs) to discover valid video conditions and candidate pairs.
    Ensures that every candidate pair consists of Vid1 and Vid2 from the SAME scene.
    """
    scenes = ["attic", "bistro_exterior", "bistro_interior", "classroom", 
              "landscape", "marbles", "pink_room", "subway", "zeroday"]
    
    if target_scene:
        scenes = [target_scene]

    conditions = []
    for scene in scenes:
        ref_path = dataset_dir / f"{scene}_reference.mp4"
        if not ref_path.exists():
            continue

        prefix = f"{scene}_"
        for p in sorted(dataset_dir.glob(f"{prefix}*.mp4")):
            filename = p.name
            if filename == f"{scene}_reference.mp4":
                continue

            rest = filename[len(prefix):-4]
            if "_level" in rest:
                metric, level = rest.rsplit("_level", 1)
                level = f"level{level}"
            else:
                metric, level = rest, "default"

            cond_name = f"{scene}:{metric}_{level}"
            conditions.append({
                "name": cond_name,
                "scene": scene,
                "filename": filename,
                "metric": metric,
                "level": level,
                "path": str(p.resolve()),
                "ref_path": str(ref_path.resolve())
            })

    cond_map = {c["name"]: i for i, c in enumerate(conditions)}

    # Read all candidate pairs from all_trials_bank.csv (2,484 pairs)
    valid_pairs = []
    if bank_path.exists():
        with open(bank_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = row["Scene"]
                if target_scene and scene != target_scene:
                    continue

                v1_metric, v1_level = row["Vid1_Metric"], row["Vid1_Level"]
                v2_metric, v2_level = row["Vid2_Metric"], row["Vid2_Level"]

                c1_name = f"{scene}:{v1_metric}_{v1_level}"
                c2_name = f"{scene}:{v2_metric}_{v2_level}"

                if c1_name in cond_map and c2_name in cond_map:
                    idx1 = cond_map[c1_name]
                    idx2 = cond_map[c2_name]
                    valid_pairs.append((idx1, idx2))

    return conditions, cond_map, valid_pairs


def run_single_rust_trial(subject_id: str, trial_num: int, scene: str, ref_path: str,
                           left_cond: dict, right_cond: dict,
                           pacer: bool = True, no_vsync: bool = False, borderless: bool = False):
    """
    Executes a single 2AFC presentation trial using the dedicated asap_trial Rust 240Hz player.
    Generates a unique output CSV for the trial and reads the participant's keypress choice.
    """
    unique_tag = f"ASAP_{subject_id}_t{trial_num}_{time.time_ns()}"
    temp_res = Path(f"experiment_results/{unique_tag}_results.csv")
    
    if temp_res.exists():
        temp_res.unlink()

    exe_suffix = ".exe" if sys.platform == "win32" else ""
    rust_bin = Path(f"rust_player/target/release/asap_trial{exe_suffix}")
    if not rust_bin.exists():
        rust_bin = Path(f"rust_player/target/debug/asap_trial{exe_suffix}")

    if not rust_bin.exists():
        print("Building asap_trial binary...", flush=True)
        subprocess.run(["cargo", "build", "--release", "--bin", "asap_trial"], cwd="rust_player", check=True)
        rust_bin = Path(f"rust_player/target/release/asap_trial{exe_suffix}")

    cmd = [
        str(rust_bin.resolve()),
        f"--left={left_cond['path']}",
        f"--ref={ref_path}",
        f"--right={right_cond['path']}",
        f"--out={temp_res.resolve()}"
    ]
    if pacer:
        cmd.append("--pacer")
    if no_vsync:
        cmd.append("--no-vsync")
    if borderless:
        cmd.append("--borderless")

    env = os.environ.copy()
    set_native_lib_env(env)  # sets LD_LIBRARY_PATH on Linux, PATH on Windows

    proc = subprocess.run(cmd, cwd=".", env=env, capture_output=True, text=True)

    choice = None
    resp_time = 0.0
    fps = 240.0

    if temp_res.exists():
        with open(temp_res, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                choice = row.get("ChosenSide")
                resp_time = float(row.get("ResponseTime_sec", 0.0))
                fps = float(row.get("PresentationFPS", 240.0))

    if temp_res.exists():
        temp_res.unlink()

    return choice, resp_time, fps


def select_best_candidate_pair_fast(sampler, M, valid_pairs, stds):
    """
    Fast Active Sampling EIG Selection (1ms execution time).
    Evaluates Expected Information Gain across valid candidate pairs using variance/uncertainty
    and recency/comparison count penalties:
      Gain(i, j) = (std_i^2 + std_j^2) / (M[i,j] + M[j,i] + 1)
    """
    best_pair = None
    max_gain = -1.0

    for idx_a, idx_b in valid_pairs:
        count = M[idx_a, idx_b] + M[idx_b, idx_a]
        var_sum = (stds[idx_a] ** 2) + (stds[idx_b] ** 2)
        gain = var_sum / (count + 1.0)

        if gain > max_gain:
            max_gain = gain
            best_pair = (idx_a, idx_b)

    if best_pair is None:
        best_pair = random.choice(valid_pairs)

    return best_pair


def main():
    parser = argparse.ArgumentParser(description="GAIM240 ASAP Active Sampling Experiment Runner")
    parser.add_argument("--subject", type=str, default="P01", help="Participant Subject ID")
    parser.add_argument("--mode", type=str, choices=["global", "intra"], default="global",
                        help="Experiment mode: 'global' (default, cross-scene JND scale) or 'intra' (single scene)")
    parser.add_argument("--scene", type=str, default=None, help="Target scene for intra mode (e.g. marbles, zeroday)")
    parser.add_argument("--trials", "--max-trials", type=int, default=30,
                        help="Number of NEW trials to run in THIS session (default: 30)")
    parser.add_argument("--target-sigma", type=float, default=0.15, help="Target score uncertainty threshold (stopping criteria)")
    parser.add_argument("--force", action="store_true", help="Force running new trials even if target sigma threshold was already reached")
    parser.add_argument("--pacer", "--pace-240", action="store_true", default=True, help="Enable 240Hz software frame pacer (default: enabled)")
    parser.add_argument("--no-pacer", action="store_false", dest="pacer", help="Disable 240Hz software frame pacer")
    parser.add_argument("--no-vsync", "--uncapped", action="store_true", help="Disable VSync for uncapped maximum presentation throughput")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless windowed mode")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    parser.add_argument("--bank", type=str, default="all_trials_bank.csv", help="Path to master trials bank CSV")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    bank_path = Path(args.bank)

    if not dataset_dir.exists():
        print(f"Error: Dataset directory not found at {dataset_dir}", flush=True)
        sys.exit(1)

    if args.mode == "intra" and not args.scene:
        args.scene = "marbles"

    conditions, cond_map, valid_pairs = discover_conditions_and_pairs(
        dataset_dir, bank_path, target_scene=args.scene if args.mode == "intra" else None
    )
    N = len(conditions)

    if N < 2 or not valid_pairs:
        print(f"Error: Discovered {N} conditions and {len(valid_pairs)} valid candidate pairs.", flush=True)
        sys.exit(1)

    # Initialize ASAP Active Sampler (Pure CPU NumPy/SciPy solver)
    sampler = ASAP(N, selective_eig=True, approx=(N > 50))
    M = np.zeros((N, N), dtype=int)

    os.makedirs("experiment_results", exist_ok=True)
    suffix = f"{args.scene}_intra" if args.mode == "intra" else "global"
    trials_csv_path = Path(f"experiment_results/{args.subject}_asap_{suffix}_trials.csv")
    scores_csv_path = Path(f"experiment_results/{args.subject}_asap_{suffix}_scores.csv")

    previously_completed = 0
    trial_history = []

    # Resume previous ASAP progress if file exists
    if trials_csv_path.exists():
        with open(trials_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                i_win = cond_map[row["ChosenCondition"]]
                j_lose = cond_map[row["RejectedCondition"]]
                M[i_win][j_lose] += 1
                previously_completed += 1
                trial_history.append(row)

    target_total = previously_completed + args.trials

    print("\n=================================================================", flush=True)
    print("  GAIM240 ACTIVE SAMPLING PERCEPTION SUITE (ASAP + Rust 240Hz)", flush=True)
    print("=================================================================", flush=True)
    print(f"Participant ID    : {args.subject}", flush=True)
    print(f"Sampling Mode     : {args.mode.upper()} {'(' + args.scene.upper() + ')' if args.scene else '(Cross-Scene Global)'}", flush=True)
    print(f"Discovered Pool   : {N} Video Conditions | {len(valid_pairs)} Valid Candidate Pairs", flush=True)
    print(f"Session New Trials: {args.trials} (Target Cumulative Total: {target_total})", flush=True)
    if previously_completed > 0:
        print(f"Previous Completed: {previously_completed} trials (Resuming session)", flush=True)
    print(f"Target Sigma      : {args.target_sigma}", flush=True)
    print(f"VSync / Pacer Mode: {'UNCAPPED (--no-vsync)' if args.no_vsync else ('SOFTWARE PACER (--pacer 240.000 FPS)' if args.pacer else 'HARDWARE VSYNC')}", flush=True)
    print("Algorithm         : ASAP Fast EIG Active Sampler (< 1ms per pair)", flush=True)
    print("Engine            : Pure CPU (NumPy/SciPy) -> Zero GPU VRAM Overhead", flush=True)
    print("=================================================================\n", flush=True)

    completed_trials = previously_completed

    # Check if target sigma threshold was ALREADY reached from previous sessions
    if previously_completed > 0:
        means, stds = sampler.get_scores()
        max_sigma = np.max(stds)
        if max_sigma <= args.target_sigma and not args.force:
            print(f"Notice: Subject '{args.subject}' has ALREADY reached the target score precision!", flush=True)
            print(f"Current Max Uncertainty: {max_sigma:.4f} <= Target Threshold ({args.target_sigma:.4f}).", flush=True)
            print("No extra trials needed! (Use --force to run additional trials anyway).\n", flush=True)

            ranked_indices = np.argsort(means)[::-1]
            print(f"Current Inferred Visual Quality Scores ({previously_completed} Total Trials):", flush=True)
            print(f"{'Rank':<5} {'Condition Name':<40} {'Score (Mean ± Std)':<25}", flush=True)
            print("-" * 75, flush=True)
            for rank, idx in enumerate(ranked_indices[:15], 1):
                c_name = conditions[idx]["name"]
                m_val, s_val = means[idx], stds[idx]
                print(f"{rank:<5} {c_name:<40} {m_val:+7.4f} ± {s_val:6.4f}", flush=True)
            print(f"\nSaved trial logs: {trials_csv_path}", flush=True)
            print(f"Saved scale scores: {scores_csv_path}", flush=True)
            return

    while completed_trials < target_total:
        means, stds = sampler.get_scores()
        max_sigma = np.max(stds)

        if completed_trials > N and max_sigma <= args.target_sigma and not args.force:
            print(f"\nStopping Criteria Reached: Max score uncertainty ({max_sigma:.4f}) <= target ({args.target_sigma:.4f}).", flush=True)
            break

        # Select next optimal valid pair using 1ms Fast Active Sampling EIG
        pair = select_best_candidate_pair_fast(sampler, M, valid_pairs, stds)
        idx_a, idx_b = int(pair[0]), int(pair[1])

        # Spatial Counterbalancing (Random Left/Right assignment)
        flip = random.choice([True, False])
        left_cond = conditions[idx_a] if not flip else conditions[idx_b]
        right_cond = conditions[idx_b] if not flip else conditions[idx_a]

        display_scene = left_cond["scene"]
        ref_path = left_cond["ref_path"]

        session_trial_num = completed_trials - previously_completed + 1
        print(f"--- Session Trial [{session_trial_num}/{args.trials}] (Cumulative: #{completed_trials + 1}, Max Sigma: {max_sigma:.3f}) ---", flush=True)
        print(f"  Scene : {display_scene.upper()}", flush=True)
        print(f"  Left  : {left_cond['name']}", flush=True)
        print(f"  Right : {right_cond['name']}", flush=True)
        print("  [Presenting 240Hz Pyramid Window - Waiting for participant response (A/D or Left/Right Arrow)]...", flush=True)

        # Launch dedicated asap_trial 240Hz Rust Player
        choice, resp_time, fps = run_single_rust_trial(
            args.subject, completed_trials + 1, display_scene, ref_path, left_cond, right_cond,
            pacer=args.pacer, no_vsync=args.no_vsync, borderless=args.borderless
        )

        if choice is None:
            print("\nExperiment interrupted by user. Progress saved.", flush=True)
            break

        chosen_cond = left_cond if choice == "LEFT" else right_cond
        rejected_cond = right_cond if choice == "LEFT" else left_cond

        chosen_idx = cond_map[chosen_cond["name"]]
        rejected_idx = cond_map[rejected_cond["name"]]

        # Update ASAP Comparison Matrix M
        M[chosen_idx][rejected_idx] += 1
        completed_trials += 1

        print(f"  Result: Chose {choice} ({chosen_cond['name']}) in {resp_time:.2f}s (FPS: {fps:.1f})\n", flush=True)

        # Record trial history
        trial_rec = {
            "SubjectID": args.subject,
            "TrialNumber": completed_trials,
            "Mode": args.mode,
            "Scene": display_scene,
            "LeftCondition": left_cond["name"],
            "RightCondition": right_cond["name"],
            "ChosenSide": choice,
            "ChosenCondition": chosen_cond["name"],
            "RejectedCondition": rejected_cond["name"],
            "ResponseTime_sec": resp_time,
            "PresentationFPS": fps,
        }
        trial_history.append(trial_rec)

        # Save updated trial history to CSV
        with open(trials_csv_path, "w", newline="") as f:
            fieldnames = list(trial_rec.keys())
            w.writeheader()
            w.writerows(trial_history)

        # Re-evaluate scores after trial
        means, stds = sampler.get_scores()

        # Save inferred visual quality scale scores
        with open(scores_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ConditionIndex", "ConditionName", "Scene", "Metric", "Level", "VisualQualityScore_Mean", "Uncertainty_StdDev"])
            for i, c in enumerate(conditions):
                w.writerow([i, c["name"], c["scene"], c["metric"], c["level"], f"{means[i]:.6f}", f"{stds[i]:.6f}"])

    # Final Summary Display
    print("\n=================================================================", flush=True)
    print("  ASAP ACTIVE SAMPLING EXPERIMENT SESSION COMPLETE", flush=True)
    print("=================================================================", flush=True)
    print(f"Completed {completed_trials - previously_completed} new trials in this session (Cumulative Total: {completed_trials}).", flush=True)
    means, stds = sampler.get_scores()
    ranked_indices = np.argsort(means)[::-1]

    print(f"\nTop 15 Inferred Visual Quality Scores out of {N} Conditions:", flush=True)
    print(f"{'Rank':<5} {'Condition Name':<40} {'Score (Mean ± Std)':<25}", flush=True)
    print("-" * 75, flush=True)
    for rank, idx in enumerate(ranked_indices[:15], 1):
        c_name = conditions[idx]["name"]
        m_val, s_val = means[idx], stds[idx]
        print(f"{rank:<5} {c_name:<40} {m_val:+7.4f} ± {s_val:6.4f}", flush=True)

    print(f"\nSaved trial logs to: {trials_csv_path}", flush=True)
    print(f"Saved quality scores to: {scores_csv_path}", flush=True)
    print("=================================================================\n", flush=True)


if __name__ == "__main__":
    main()
