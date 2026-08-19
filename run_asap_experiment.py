#!/usr/bin/env python3
"""
run_asap_experiment.py
======================
GAIM240 Active Sampling Perception Experiment Runner (ASAP Batch Mode + Rust 240Hz)
----------------------------------------------------------------------------------
Unified CLI and master entry point for conducting the multi-observer perception experiment.

Architecture:
- 9 Independent ASAP models (one per scene: attic, bistro_exterior, bistro_interior,
  classroom, landscape, marbles, pink_room, subway, zeroday).
- Active Sampling in Batch Mode: Generates a 234-trial presentation batch per observer
  using Minimum Spanning Trees (MST) across each scene.
- Interleaves scenes in pseudo-random rounds to maintain perceptual freshness.
- Hardware-accelerated 240Hz presentation with software pacing and millisecond keypress capture.
- Sequential prior updates: Aggregates observer results after each session, re-fits
  TrueSkill posteriors, and generates next-generation batches for subsequent observers.

Usage:
  # End-to-end observer session (Generate batch -> Run 240Hz experiment -> Update priors):
  python run_asap_experiment.py --subject P01

  # Run specific steps:
  python run_asap_experiment.py --subject P01 --generate-only
  python run_asap_experiment.py --subject P01 --run-only
  python run_asap_experiment.py --update-only
"""

import argparse
import sys
from pathlib import Path

from asap_batch_generator import generate_batch
from asap_prior_updater import update_priors_and_scores
from platform_utils import get_dataset_dir
from run_batch_experiment import run_batch_session


def main():
    parser = argparse.ArgumentParser(
        description="GAIM240 ASAP Multi-Scene Batch Perception Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_asap_experiment.py --subject P01
  python run_asap_experiment.py --subject P02 --borderless
  python run_asap_experiment.py --subject P03 --generate-only
  python run_asap_experiment.py --update-only
        """
    )
    parser.add_argument("--subject", type=str, default=None, help="Participant / Observer ID (e.g. P01, P02)")
    parser.add_argument("--batch", type=str, default=None, help="Explicit batch CSV path (default: batches/batch_<subject>.csv)")
    parser.add_argument("--pairs-per-scene", type=int, default=None, help="Number of pairs per scene in batch (default: MST 26)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for batch generation")
    parser.add_argument("--pacer", "--pace-240", action="store_true", default=True, help="Enable 240Hz software frame pacer (default: enabled)")
    parser.add_argument("--no-pacer", action="store_false", dest="pacer", help="Disable 240Hz software frame pacer")
    parser.add_argument("--no-vsync", "--uncapped", action="store_true", help="Disable VSync for uncapped maximum presentation throughput")
    parser.add_argument("--borderless", action="store_true", help="Enable borderless windowed presentation mode")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")

    # Workflow mode switches
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--generate-only", action="store_true", help="Only generate the presentation batch CSV without running the experiment")
    mode_group.add_argument("--run-only", action="store_true", help="Only run the experiment with an existing batch CSV without regenerating")
    mode_group.add_argument("--update-only", action="store_true", help="Only aggregate history and update TrueSkill/ASAP scores without running")

    args = parser.parse_args()
    dataset_dir = Path(args.dataset)

    # 1. Update only mode
    if args.update_only:
        print("[Unified Runner] Mode: UPDATE-ONLY")
        update_priors_and_scores(dataset_dir=dataset_dir)
        return

    if not args.subject:
        parser.error("--subject is required unless --update-only is specified.")

    batch_path = Path(args.batch) if args.batch else Path(f"batches/batch_{args.subject}.csv")

    # 2. Generate only mode
    if args.generate_only:
        print(f"[Unified Runner] Mode: GENERATE-ONLY for subject '{args.subject}'")
        generate_batch(
            subject_id=args.subject,
            output_path=batch_path,
            dataset_dir=dataset_dir,
            pairs_per_scene=args.pairs_per_scene,
            seed=args.seed
        )
        return

    # 3. Run only mode
    if args.run_only:
        print(f"[Unified Runner] Mode: RUN-ONLY for subject '{args.subject}' using {batch_path}")
        run_batch_session(
            subject_id=args.subject,
            batch_csv=batch_path,
            dataset_dir=dataset_dir,
            pacer=args.pacer,
            no_vsync=args.no_vsync,
            borderless=args.borderless
        )
        return

    # 4. Standard End-to-End Orchestration: Generate -> Run -> Update
    print(f"[Unified Runner] Mode: FULL EXPERIMENT SESSION for subject '{args.subject}'")
    if not batch_path.exists():
        generate_batch(
            subject_id=args.subject,
            output_path=batch_path,
            dataset_dir=dataset_dir,
            pairs_per_scene=args.pairs_per_scene,
            seed=args.seed
        )
    else:
        print(f"[Unified Runner] Using existing batch: {batch_path}")

    run_batch_session(
        subject_id=args.subject,
        batch_csv=batch_path,
        dataset_dir=dataset_dir,
        pacer=args.pacer,
        no_vsync=args.no_vsync,
        borderless=args.borderless
    )


if __name__ == "__main__":
    main()
