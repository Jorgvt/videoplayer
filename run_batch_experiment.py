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
    total_trials: int,
    left_path: str,
    ref_path: str,
    right_path: str,
    pacer: bool = True,
    no_vsync: bool = False,
    borderless: bool = False,
    feedback_ms: int = 300,
    warmup_ms: int = 500
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
        f"--out={temp_res.resolve()}",
        f"--trial={trial_num}",
        f"--total-trials={total_trials}",
        f"--feedback-ms={feedback_ms}",
        f"--warmup-ms={warmup_ms}"
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
    borderless: bool = False,
    feedback_ms: int = 300,
    warmup_ms: int = 0
):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    if batch_csv is None:
        batch_csv = Path(f"batches/batch_{subject_id}.csv")

    # Generate batch if it does not exist
    if not batch_csv.exists():
        print(f"[Run Batch Experiment] Batch file not found. Generating new batch: {batch_csv}")
        generate_batch(subject_id=subject_id, output_path=batch_csv, dataset_dir=dataset_dir)

    results_dir = Path("experiment_results")
    results_dir.mkdir(parents=True, exist_ok=True)
    trials_csv_path = results_dir / f"trials_{subject_id}.csv"

    exe_suffix = ".exe" if sys.platform == "win32" else ""
    rust_bin = Path(f"rust_player/target/release/asap_batch{exe_suffix}")
    if not rust_bin.exists():
        rust_bin = Path(f"rust_player/target/debug/asap_batch{exe_suffix}")

    if not rust_bin.exists():
        print("Building asap_batch binary...", flush=True)
        subprocess.run(["cargo", "build", "--release", "--bin", "asap_batch"], cwd="rust_player", check=True)
        rust_bin = Path(f"rust_player/target/release/asap_batch{exe_suffix}")

    cmd = [
        str(rust_bin.resolve()),
        f"--batch={batch_csv.resolve()}",
        f"--out={trials_csv_path.resolve()}",
        f"--feedback-ms={feedback_ms}",
        f"--warmup-ms={warmup_ms}"
    ]
    if pacer:
        cmd.append("--pacer")
    if no_vsync:
        cmd.append("--no-vsync")
    if borderless:
        cmd.append("--borderless")

    env = os.environ.copy()
    set_native_lib_env(env)

    # Launch seamless single-window batch presentation
    subprocess.run(cmd, cwd=".", env=env)

    # Automatically trigger prior updates across all 9 scenes upon session completion
    print("[Run Batch Experiment] Updating cumulative ASAP models & scale scores...", flush=True)
    update_priors_and_scores(dataset_dir=dataset_dir, results_dir=results_dir)


def main():
    parser = argparse.ArgumentParser(description="GAIM240 ASAP Multi-Scene Batch Runner (240Hz Single-Window)")
    parser.add_argument("--subject", type=str, required=True, help="Subject/Observer ID (e.g. P01, P02)")
    parser.add_argument("--batch", type=str, default=None, help="Path to batch CSV file (default: batches/batch_<subject>.csv)")
    parser.add_argument("--pacer", "--pace-240", action="store_true", default=True, help="Enable 240Hz software frame pacer (default: enabled)")
    parser.add_argument("--no-pacer", action="store_false", dest="pacer", help="Disable 240Hz software frame pacer")
    parser.add_argument("--no-vsync", "--uncapped", action="store_true", help="Disable VSync for uncapped maximum presentation throughput")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless windowed mode")
    parser.add_argument("--feedback-ms", type=int, default=300, help="Visual feedback duration in ms (default: 300 ms green border)")
    parser.add_argument("--warmup-ms", type=int, default=0, help="Optional minimum progress display duration in ms (default: 0 ms dynamic until loaded)")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    args = parser.parse_args()

    batch_p = Path(args.batch) if args.batch else None
    run_batch_session(
        subject_id=args.subject,
        batch_csv=batch_p,
        dataset_dir=Path(args.dataset),
        pacer=args.pacer,
        no_vsync=args.no_vsync,
        borderless=args.borderless,
        feedback_ms=args.feedback_ms,
        warmup_ms=args.warmup_ms
    )


if __name__ == "__main__":
    main()
