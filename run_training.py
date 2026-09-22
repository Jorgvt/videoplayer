#!/usr/bin/env python3
"""
run_training.py
===============
Executes a dedicated 18-trial training/familiarization session for observers
using the 240Hz Rust Pyramid Player.

Key behaviors:
- Uses the standardized `training/training_batch.csv` master template.
- Generates a participant-scoped batch in `training/batches/batch_<subject>.csv`.
- Saves participant results to `training/results/trials_<subject>.csv`.
- Analyzes and displays the number of correctly assessed pairs and accuracy.
- Completely isolated from the main experiment: DOES NOT touch `experiment_results/`
  and DOES NOT update ASAP experiment priors or psychometric scales.
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from platform_utils import get_dataset_dir, set_native_lib_env


def prepare_subject_training_batch(
    master_batch_csv: Path,
    subject_id: str,
    output_batch_csv: Path,
    dataset_dir: Path
) -> Path:
    """
    Reads the master training batch CSV and writes a copy with the participant's
    SubjectID and resolved dataset paths.
    """
    if not master_batch_csv.exists():
        from training.generate_training_batch import generate_training_csv
        print(f"[Training] Master batch not found. Generating {master_batch_csv}...", flush=True)
        generate_training_csv(output_path=master_batch_csv, dataset_dir=dataset_dir)

    output_batch_csv.parent.mkdir(parents=True, exist_ok=True)

    with open(master_batch_csv, "r", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Replace SubjectID in all rows
    for row in rows:
        row["SubjectID"] = subject_id

    with open(output_batch_csv, "w", newline="", encoding="utf-8") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return output_batch_csv


def print_training_summary(trials_csv_path: Path):
    """
    Reads the output CSV and prints an assessment accuracy summary.
    """
    if not trials_csv_path.exists():
        print(f"  [Notice] No results file found at {trials_csv_path}")
        return

    trials = []
    with open(trials_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trials.append(row)

    if not trials:
        print("  [Notice] No trials completed.")
        return

    total_trials = len(trials)
    correct_count = 0
    missed_trials = []
    response_times = []

    for row in trials:
        chosen_cond = row.get("ChosenCondition", "")
        rejected_cond = row.get("RejectedCondition", "")
        scene = row.get("Scene", "")
        trial_num = row.get("TrialNumber", "")
        rt = float(row.get("ResponseTime_sec", 0.0))
        response_times.append(rt)

        # In training, participant is expected to identify the pristine reference
        # over the level 2 distortion.
        if ":reference" in chosen_cond:
            correct_count += 1
        else:
            missed_trials.append((trial_num, scene, chosen_cond, rejected_cond))

    accuracy = (correct_count / total_trials) * 100.0 if total_trials > 0 else 0.0
    avg_rt = (sum(response_times) / len(response_times)) if response_times else 0.0

    print(f"  Total Trials Completed:      {total_trials}")
    print(f"  Correctly Assessed Pairs:    {correct_count} / {total_trials} ({accuracy:.1f}%)")
    print(f"  Average Response Time:       {avg_rt:.2f} seconds")

    if missed_trials:
        print("\n  Missed / Inverted Pairs:")
        for t_num, scene, chosen, rejected in missed_trials:
            # Extract clean distortion name from condition
            chosen_clean = chosen.split(":")[-1] if ":" in chosen else chosen
            rejected_clean = rejected.split(":")[-1] if ":" in rejected else rejected
            print(f"    - Trial #{t_num} ({scene}): Chose '{chosen_clean}' over pristine '{rejected_clean}'")
    else:
        print("  Status:                      All distortions successfully recognized! (100%)")


def run_training_session(
    subject_id: str,
    master_batch_csv: Path = None,
    dataset_dir: Path = None,
    pacer: bool = True,
    no_vsync: bool = False,
    borderless: bool = False,
    feedback_ms: int = 300,
    warmup_ms: int = 0
):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    if master_batch_csv is None:
        master_batch_csv = Path("training/training_batch.csv")

    training_dir = Path("training")
    results_dir = training_dir / "results"
    batches_dir = training_dir / "batches"
    results_dir.mkdir(parents=True, exist_ok=True)
    batches_dir.mkdir(parents=True, exist_ok=True)

    subject_batch_csv = batches_dir / f"batch_{subject_id}.csv"
    prepare_subject_training_batch(
        master_batch_csv=master_batch_csv,
        subject_id=subject_id,
        output_batch_csv=subject_batch_csv,
        dataset_dir=dataset_dir
    )

    trials_csv_path = results_dir / f"trials_{subject_id}.csv"

    exe_suffix = ".exe" if sys.platform == "win32" else ""
    rust_bin = Path(f"rust_player/target/release/asap_batch{exe_suffix}")
    if not rust_bin.exists():
        rust_bin = Path(f"rust_player/target/debug/asap_batch{exe_suffix}")

    if not rust_bin.exists():
        print("[Training] Building asap_batch binary...", flush=True)
        subprocess.run(["cargo", "build", "--release", "--bin", "asap_batch"], cwd="rust_player", check=True)
        rust_bin = Path(f"rust_player/target/release/asap_batch{exe_suffix}")

    cmd = [
        str(rust_bin.resolve()),
        f"--batch={subject_batch_csv.resolve()}",
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

    print("\n" + "=" * 65)
    print(f"  STARTING TRAINING / PRACTICE SESSION FOR SUBJECT: {subject_id}")
    print("  Controls: [A] / [Left Arrow] = Select Left Video")
    print("            [D] / [Right Arrow] = Select Right Video")
    print("            [Q] / [ESC] = Exit / Pause Session")
    print("=" * 65 + "\n", flush=True)

    # Launch seamless single-window batch presentation
    subprocess.run(cmd, cwd=".", env=env)

    print("\n" + "=" * 65)
    print("  TRAINING SESSION FINISHED")
    print("=" * 65)
    print_training_summary(trials_csv_path)
    print(f"\n  Results logged to:           {trials_csv_path}")
    print("  (Main experiment priors remain untouched)")
    print("=" * 65 + "\n", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GAIM240 Observer Training Session Runner (240Hz Single-Window)")
    parser.add_argument("--subject", type=str, required=True, help="Subject/Observer ID (e.g. P01, test_subject)")
    parser.add_argument("--batch", type=str, default=None, help="Path to training batch CSV file (default: training/training_batch.csv)")
    parser.add_argument("--pacer", "--pace-240", action="store_true", default=True, help="Enable 240Hz software frame pacer (default: enabled)")
    parser.add_argument("--no-pacer", action="store_false", dest="pacer", help="Disable 240Hz software frame pacer")
    parser.add_argument("--no-vsync", "--uncapped", action="store_true", help="Disable VSync for uncapped maximum presentation throughput")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless windowed mode")
    parser.add_argument("--feedback-ms", type=int, default=300, help="Visual feedback duration in ms (default: 300 ms green border)")
    parser.add_argument("--warmup-ms", type=int, default=0, help="Optional minimum progress display duration in ms (default: 0 ms dynamic until loaded)")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    args = parser.parse_args()

    batch_p = Path(args.batch) if args.batch else None
    run_training_session(
        subject_id=args.subject,
        master_batch_csv=batch_p,
        dataset_dir=Path(args.dataset),
        pacer=args.pacer,
        no_vsync=args.no_vsync,
        borderless=args.borderless,
        feedback_ms=args.feedback_ms,
        warmup_ms=args.warmup_ms
    )


if __name__ == "__main__":
    main()
