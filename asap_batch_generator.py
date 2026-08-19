#!/usr/bin/env python3
"""
asap_batch_generator.py
=======================
Generates presentation batches for the GAIM240 Active Sampling Perception Experiment.

Features:
- Maintains 9 independent ASAP models (one per scene: attic, bistro_exterior,
  bistro_interior, classroom, landscape, marbles, pink_room, subway, zeroday).
- Learns from cumulative prior trials across all observers (from all_trials_history.csv
  or per-subject logs).
- Uses Minimum Spanning Tree (MST) active sampling per scene (26 pairs/scene -> 234 total trials).
- Applies spatial counterbalancing (50/50 randomized left/right assignment).
- Interleaves scenes using a stratified pseudo-random mixer (max 2 consecutive trials per scene).
- Exports batch to `batches/batch_<subject>.csv`.
"""

import argparse
import csv
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from platform_utils import get_dataset_dir

# Add asap/python to sys.path
ASAP_PATH = Path(__file__).parent / "asap" / "python"
if str(ASAP_PATH) not in sys.path:
    sys.path.insert(0, str(ASAP_PATH))

try:
    from asap_cpu import ASAP
except ImportError:
    print("Error: Could not import ASAP from asap/python/asap_cpu.py", file=sys.stderr)
    sys.exit(1)

SCENES = [
    "attic",
    "bistro_exterior",
    "bistro_interior",
    "classroom",
    "landscape",
    "marbles",
    "pink_room",
    "subway",
    "zeroday",
]


def discover_scene_conditions(dataset_dir: Path, scene: str) -> Tuple[List[Dict], Dict[str, int]]:
    """
    Discovers all distortion conditions for a given scene in the dataset directory.
    Returns (conditions_list, condition_name_to_index_map).
    """
    ref_path = dataset_dir / f"{scene}_reference.mp4"
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference video missing for scene '{scene}': {ref_path}")

    prefix = f"{scene}_"
    conditions = []
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
            "ref_path": str(ref_path.resolve()),
        })

    cond_map = {c["name"]: idx for idx, c in enumerate(conditions)}
    return conditions, cond_map


def load_cumulative_matrices(
    dataset_dir: Path,
    history_csv: Path = Path("experiment_results/all_trials_history.csv"),
    results_dir: Path = Path("experiment_results")
) -> Tuple[Dict[str, List[Dict]], Dict[str, Dict[str, int]], Dict[str, np.ndarray]]:
    """
    Builds conditions, condition maps, and comparison matrices M for all 9 scenes.
    Populates M from all_trials_history.csv or individual observer logs.
    """
    scene_conditions: Dict[str, List[Dict]] = {}
    scene_cond_maps: Dict[str, Dict[str, int]] = {}
    scene_matrices: Dict[str, np.ndarray] = {}

    for scene in SCENES:
        conds, cond_map = discover_scene_conditions(dataset_dir, scene)
        n = len(conds)
        scene_conditions[scene] = conds
        scene_cond_maps[scene] = cond_map
        scene_matrices[scene] = np.zeros((n, n), dtype=int)

    # 1. Try reading master aggregated history
    trials_loaded = 0
    if history_csv.exists():
        with open(history_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene = row.get("Scene")
                chosen = row.get("ChosenCondition")
                rejected = row.get("RejectedCondition")
                if scene in scene_matrices and chosen in scene_cond_maps[scene] and rejected in scene_cond_maps[scene]:
                    i_win = scene_cond_maps[scene][chosen]
                    j_lose = scene_cond_maps[scene][rejected]
                    scene_matrices[scene][i_win, j_lose] += 1
                    trials_loaded += 1
    # 2. Otherwise scan individual trials_*.csv files in results_dir
    elif results_dir.exists():
        for p in results_dir.glob("trials_*.csv"):
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    scene = row.get("Scene")
                    chosen = row.get("ChosenCondition")
                    rejected = row.get("RejectedCondition")
                    if scene in scene_matrices and chosen in scene_cond_maps[scene] and rejected in scene_cond_maps[scene]:
                        i_win = scene_cond_maps[scene][chosen]
                        j_lose = scene_cond_maps[scene][rejected]
                        scene_matrices[scene][i_win, j_lose] += 1
                        trials_loaded += 1

    return scene_conditions, scene_cond_maps, scene_matrices


def generate_scene_mst_pairs(
    conditions: List[Dict],
    M: np.ndarray,
    pairs_per_scene: int = None
) -> List[Tuple[int, int]]:
    """
    Runs ASAP on the scene comparison matrix M to extract optimal Minimum Spanning Tree pairs.
    """
    n = len(conditions)
    asap_model = ASAP(n, selective_eig=True, approx=False)
    raw_pairs = asap_model.run_asap(M, mst_mode=True)

    pairs_list = [(int(u), int(v)) for u, v in raw_pairs]

    if pairs_per_scene is not None and pairs_per_scene > 0:
        if len(pairs_list) > pairs_per_scene:
            pairs_list = pairs_list[:pairs_per_scene]
        elif len(pairs_list) < pairs_per_scene:
            # If more requested than MST, fill with candidate pairs
            all_possible = [(i, j) for i in range(n) for j in range(i + 1, n)]
            random.shuffle(all_possible)
            for p in all_possible:
                if p not in pairs_list and (p[1], p[0]) not in pairs_list:
                    pairs_list.append(p)
                if len(pairs_list) >= pairs_per_scene:
                    break

    return pairs_list


def interleave_scenes_stratified(
    scene_trial_items: Dict[str, List[Dict]],
    max_consecutive_same_scene: int = 2,
    seed: int = None
) -> List[Dict]:
    """
    Interleaves trials from multiple scenes in pseudo-random rounds,
    guaranteeing that no scene appears more than `max_consecutive_same_scene` times consecutively.
    """
    rng = random.Random(seed)
    
    # Make mutable queues for each scene
    queues = {s: list(items) for s, items in scene_trial_items.items()}
    for s in queues:
        rng.shuffle(queues[s])

    interleaved = []
    consecutive_scene = None
    consecutive_count = 0

    total_remaining = sum(len(q) for q in queues.values())

    while total_remaining > 0:
        # Candidate scenes that have remaining trials
        available = [s for s, q in queues.items() if len(q) > 0]

        # Filter out scene if it reached max consecutive limit and alternatives exist
        if consecutive_count >= max_consecutive_same_scene and len(available) > 1:
            candidates = [s for s in available if s != consecutive_scene]
        else:
            candidates = available

        # Pick randomly among candidates
        chosen_scene = rng.choice(candidates)
        trial = queues[chosen_scene].pop(0)
        interleaved.append(trial)

        if chosen_scene == consecutive_scene:
            consecutive_count += 1
        else:
            consecutive_scene = chosen_scene
            consecutive_count = 1

        total_remaining -= 1

    return interleaved


def generate_batch(
    subject_id: str,
    output_path: Path = None,
    dataset_dir: Path = None,
    pairs_per_scene: int = None,
    seed: int = None
) -> Path:
    """
    Generates a complete multi-scene active sampling presentation batch for a subject.
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    if output_path is None:
        output_dir = Path("batches")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"batch_{subject_id}.csv"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    print(f"[ASAP Batch Generator] Loading priors & dataset: {dataset_dir}", flush=True)
    scene_conditions, scene_cond_maps, scene_matrices = load_cumulative_matrices(dataset_dir)

    total_prior_comparisons = sum(m.sum() for m in scene_matrices.values())
    print(f"[ASAP Batch Generator] Total historical comparisons loaded: {total_prior_comparisons}", flush=True)

    scene_trial_items: Dict[str, List[Dict]] = {}
    total_pairs = 0

    for scene in SCENES:
        conds = scene_conditions[scene]
        M = scene_matrices[scene]
        prior_scene_cmps = int(M.sum())

        pairs = generate_scene_mst_pairs(conds, M, pairs_per_scene=pairs_per_scene)
        total_pairs += len(pairs)
        print(f"  - Scene '{scene:<16}': {len(conds)} conditions | {prior_scene_cmps} prior cmps -> Generated {len(pairs)} MST pairs", flush=True)

        trials_for_scene = []
        for idx_a, idx_b in pairs:
            cond_a = conds[idx_a]
            cond_b = conds[idx_b]

            # Spatial counterbalancing: 50% random left/right swap
            if rng.random() > 0.5:
                left_cond, right_cond = cond_a, cond_b
            else:
                left_cond, right_cond = cond_b, cond_a

            trials_for_scene.append({
                "SubjectID": subject_id,
                "Scene": scene,
                "RefFilename": f"{scene}_reference.mp4",
                "RefPath": cond_a["ref_path"],
                "LeftCondition": left_cond["name"],
                "LeftFilename": left_cond["filename"],
                "LeftMetric": left_cond["metric"],
                "LeftLevel": left_cond["level"],
                "LeftPath": left_cond["path"],
                "RightCondition": right_cond["name"],
                "RightFilename": right_cond["filename"],
                "RightMetric": right_cond["metric"],
                "RightLevel": right_cond["level"],
                "RightPath": right_cond["path"],
            })

        scene_trial_items[scene] = trials_for_scene

    # Interleave scenes
    interleaved_trials = interleave_scenes_stratified(
        scene_trial_items,
        max_consecutive_same_scene=2,
        seed=seed
    )

    # Assign sequential TrialNumber
    for idx, t in enumerate(interleaved_trials, 1):
        t["TrialNumber"] = idx

    fieldnames = [
        "TrialNumber",
        "SubjectID",
        "Scene",
        "RefFilename",
        "RefPath",
        "LeftCondition",
        "LeftFilename",
        "LeftMetric",
        "LeftLevel",
        "LeftPath",
        "RightCondition",
        "RightFilename",
        "RightMetric",
        "RightLevel",
        "RightPath",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(interleaved_trials)

    print(f"\n[ASAP Batch Generator] Successfully generated {len(interleaved_trials)} total interleaved trials for subject '{subject_id}'", flush=True)
    print(f"[ASAP Batch Generator] Batch saved to: {output_path.resolve()}\n", flush=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="GAIM240 ASAP Multi-Scene Batch Generator")
    parser.add_argument("--subject", type=str, required=True, help="Subject/Observer ID (e.g. P01, P02)")
    parser.add_argument("--out", type=str, default=None, help="Output batch CSV path (default: batches/batch_<subject>.csv)")
    parser.add_argument("--pairs-per-scene", type=int, default=None, help="Number of pairs per scene (default: MST N-1, i.e. 26)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducible shuffling / spatial balance")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    args = parser.parse_args()

    out_p = Path(args.out) if args.out else None
    generate_batch(
        subject_id=args.subject,
        output_path=out_p,
        dataset_dir=Path(args.dataset),
        pairs_per_scene=args.pairs_per_scene,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
