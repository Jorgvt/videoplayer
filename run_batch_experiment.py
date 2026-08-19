#!/usr/bin/env python3
"""
run_batch_experiment.py
=======================
Executes pre-generated multi-scene ASAP presentation batches using the 240Hz Rust Pyramid Player.

Features:
- Reads presentation batch from `batches/batch_<subject>.csv`.
- Auto-resumes partially completed sessions from `experiment_results/trials_<subject>.csv`.
- Invokes hardware-accelerated `asap_trial` 240Hz presentation binary for each trial.
- Logs observer choices (A/D or Left/Right Arrow), response times, and presentation FPS.
- Automatically triggers prior updates across all 9 scenes upon batch completion.
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

from asap_batch_generator import generate_batch
from asap_prior_updater import update_priors_and_scores
from platform_utils import get_dataset_dir, set_native_lib_env


def run_single_presentation_trial(
    subject_id: str,
    trial_num: int,
    left_path: str,
    ref_path: str,
    right_path: str,
    pacer: bool = True,
    no_vsync: bool = False,
    borderless: bool = False
):
    """
    Executes a single 2AFC presentation trial using the dedicated asap_trial Rust 240Hz player.
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
        f"--left={left_path}",
        f"--ref={ref_path}",
        f"--right={right_path}",
        f"--out={temp_res.resolve()}"
    ]
    if pacer:
        cmd.append("--pacer")
    if no_vsync:
        cmd.append("--no-vsync")
    if borderless:
        cmd.append("--borderless")

    env = os.environ.copy()
    set_native_lib_env(env)

    proc = subprocess.run(cmd, cwd=".", env=env, capture_output=True, text=True)

    choice = None
    resp_time = 0.0
    fps = 240.0

    if temp_res.exists():
        with open(temp_res, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                choice = row.get("ChosenSide")
                resp_time = float(row.get("ResponseTime_sec", 0.0))
                fps = float(row.get("PresentationFPS", 240.0))
        temp_res.unlink()

    return choice, resp_time, fps


def run_batch_session(
    subject_id: str,
    batch_csv: Path = None,
    dataset_dir: Path = None,
    pacer: bool = True,
    no_vsync: bool = False,
    borderless: bool = False
):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    if batch_csv is None:
        batch_csv = Path(f"batches/batch_{subject_id}.csv")

    # Generate batch if it does not exist
    if not batch_csv.exists():
        print(f"[Run Batch Experiment] Batch file not found. Generating new batch: {batch_csv}")
        generate_batch(subject_id=subject_id, output_path=batch_csv, dataset_dir=dataset_dir)

    # Read batch trials
    batch_trials = []
    with open(batch_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_trials.append(row)

    total_batch_trials = len(batch_trials)
    if total_batch_trials == 0:
        print(f"Error: Batch file {batch_csv} contains no trials.", file=sys.stderr)
        sys.exit(1)

    results_dir = Path("experiment_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_csv_path = results_dir / f"trials_{subject_id}.csv"

    # Check for auto-resume
    completed_trials = []
    completed_numbers = set()
    if trials_csv_path.exists():
        with open(trials_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                completed_trials.append(row)
                try:
                    completed_numbers.add(int(row.get("TrialNumber", 0)))
                except ValueError:
                    pass

    num_previously_completed = len(completed_trials)

    print("\n=================================================================", flush=True)
    print("  GAIM240 ACTIVE SAMPLING BATCH EXPERIMENT (240Hz Rust Player)", flush=True)
    print("=================================================================", flush=True)
    print(f"Subject / Observer ID : {subject_id}", flush=True)
    print(f"Batch File            : {batch_csv}", flush=True)
    print(f"Total Batch Trials    : {total_batch_trials}", flush=True)
    if num_previously_completed > 0:
        print(f"Previously Completed  : {num_previously_completed}/{total_batch_trials} (Resuming)", flush=True)
    print(f"Presentation Rate     : {'UNCAPPED (--no-vsync)' if no_vsync else ('SOFTWARE PACER (240.0 FPS)' if pacer else 'HARDWARE VSYNC')}", flush=True)
    print("=================================================================\n", flush=True)

    if num_previously_completed >= total_batch_trials:
        print(f"Notice: Subject '{subject_id}' has already completed all {total_batch_trials} trials in this batch!", flush=True)
        print(f"Logged trials: {trials_csv_path.resolve()}\n", flush=True)
        update_priors_and_scores(dataset_dir=dataset_dir, results_dir=results_dir)
        return

    # Trial Execution Loop
    for row in batch_trials:
        trial_num = int(row["TrialNumber"])
        if trial_num in completed_numbers:
            continue

        scene = row["Scene"]
        left_name = row["LeftCondition"]
        right_name = row["RightCondition"]
        left_path = row["LeftPath"]
        ref_path = row["RefPath"]
        right_path = row["RightPath"]

        print(f"--- Trial [{trial_num}/{total_batch_trials}] (Scene: {scene.upper()}) ---", flush=True)
        print(f"  Left  : {left_name}", flush=True)
        print(f"  Right : {right_name}", flush=True)
        print("  [Presenting 240Hz Pyramid Window - Waiting for response (A/D or Left/Right Arrow)]...", flush=True)

        choice, resp_time, fps = run_single_presentation_trial(
            subject_id=subject_id,
            trial_num=trial_num,
            left_path=left_path,
            ref_path=ref_path,
            right_path=right_path,
            pacer=pacer,
            no_vsync=no_vsync,
            borderless=borderless
        )

        if choice is None:
            print(f"\n[Run Batch Experiment] Session interrupted by user at trial {trial_num}. Progress saved.", flush=True)
            break

        chosen_cond = left_name if choice == "LEFT" else right_name
        rejected_cond = right_name if choice == "LEFT" else left_name

        rec = {
            "SubjectID": subject_id,
            "TrialNumber": trial_num,
            "Scene": scene,
            "LeftCondition": left_name,
            "RightCondition": right_name,
            "ChosenSide": choice,
            "ChosenCondition": chosen_cond,
            "RejectedCondition": rejected_cond,
            "ResponseTime_sec": resp_time,
            "PresentationFPS": fps,
        }
        completed_trials.append(rec)
        completed_numbers.add(trial_num)

        print(f"  -> Result: Chose {choice} ({chosen_cond}) in {resp_time:.2f}s (FPS: {fps:.1f})\n", flush=True)

        # Save progress incrementally
        with open(trials_csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = list(rec.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(completed_trials)

    print("\n=================================================================", flush=True)
    print("  BATCH PRESENTATION SESSION FINISHED", flush=True)
    print("=================================================================", flush=True)
    print(f"Completed {len(completed_trials)}/{total_batch_trials} trials for subject '{subject_id}'.", flush=True)
    print(f"Trial log saved: {trials_csv_path.resolve()}\n", flush=True)

    # Automatically trigger prior updates across all 9 scenes
    print("[Run Batch Experiment] Updating cumulative ASAP models & scale scores...", flush=True)
    update_priors_and_scores(dataset_dir=dataset_dir, results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description="GAIM240 ASAP Multi-Scene Batch Runner (240Hz)")
    parser.add_argument("--subject", type=str, required=True, help="Subject/Observer ID (e.g. P01, P02)")
    parser.add_argument("--batch", type=str, default=None, help="Path to batch CSV file (default: batches/batch_<subject>.csv)")
    parser.add_argument("--pacer", "--pace-240", action="store_true", default=True, help="Enable 240Hz software frame pacer (default: enabled)")
    parser.add_argument("--no-pacer", action="store_false", dest="pacer", help="Disable 240Hz software frame pacer")
    parser.add_argument("--no-vsync", "--uncapped", action="store_true", help="Disable VSync for uncapped maximum presentation throughput")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless windowed mode")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    args = parser.parse_args()

    batch_p = Path(args.batch) if args.batch else None
    run_batch_session(
        subject_id=args.subject,
        batch_csv=batch_p,
        dataset_dir=Path(args.dataset),
        pacer=args.pacer,
        no_vsync=args.no_vsync,
        borderless=args.borderless
    )


if __name__ == "__main__":
    main()
