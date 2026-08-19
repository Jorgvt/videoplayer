#!/usr/bin/env python3
"""
asap_prior_updater.py
=====================
Aggregates perception experiment results from all observers, updates ASAP
comparison matrices per scene, solves TrueSkill scores, and exports global scale results.

Outputs:
- `experiment_results/all_trials_history.csv` (Aggregated master history)
- `experiment_results/scene_scores.csv` (Per-condition JND quality scores & uncertainties)
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from asap_batch_generator import SCENES, discover_scene_conditions
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


def aggregate_all_observer_trials(results_dir: Path = Path("experiment_results")) -> List[Dict]:
    """
    Scans all observer CSV files (`trials_*.csv`) in results_dir and aggregates them.
    Deduplicates trials based on SubjectID + TrialNumber + Scene + Conditions.
    """
    all_trials = []
    seen_keys = set()

    for p in sorted(results_dir.glob("trials_*.csv")):
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sub = row.get("SubjectID", "")
                t_num = row.get("TrialNumber", "")
                scene = row.get("Scene", "")
                c_win = row.get("ChosenCondition", "")
                c_lose = row.get("RejectedCondition", "")

                if not (sub and scene and c_win and c_lose):
                    continue

                dedup_key = (sub, t_num, scene, c_win, c_lose)
                if dedup_key not in seen_keys:
                    seen_keys.add(dedup_key)
                    all_trials.append(row)

    return all_trials


def update_priors_and_scores(
    dataset_dir: Path = None,
    results_dir: Path = Path("experiment_results"),
    history_csv: Path = Path("experiment_results/all_trials_history.csv"),
    scores_csv: Path = Path("experiment_results/scene_scores.csv")
) -> Dict[str, Dict]:
    """
    Updates comparison matrices for all 9 scenes, solves TrueSkill scores,
    and saves updated aggregated history and score tables.
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Aggregate trials across all observers
    all_trials = aggregate_all_observer_trials(results_dir)
    print(f"[ASAP Prior Updater] Aggregated {len(all_trials)} total valid comparison trials across all observers.")

    if all_trials:
        # Save master history CSV
        fieldnames = list(all_trials[0].keys())
        with open(history_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_trials)
        print(f"[ASAP Prior Updater] Saved aggregated history: {history_csv.resolve()}")

    # 2. Rebuild conditions & matrices per scene
    scene_results = {}
    all_score_rows = []

    for scene in SCENES:
        conds, cond_map = discover_scene_conditions(dataset_dir, scene)
        n = len(conds)
        M = np.zeros((n, n), dtype=int)
        cond_cmp_counts = np.zeros(n, dtype=int)

        # Fill M from all_trials
        for row in all_trials:
            if row.get("Scene") == scene:
                c_win = row.get("ChosenCondition")
                c_lose = row.get("RejectedCondition")
                if c_win in cond_map and c_lose in cond_map:
                    idx_w = cond_map[c_win]
                    idx_l = cond_map[c_lose]
                    M[idx_w, idx_l] += 1
                    cond_cmp_counts[idx_w] += 1
                    cond_cmp_counts[idx_l] += 1

        # Solve TrueSkill scores
        asap_model = ASAP(n, selective_eig=True, approx=False)
        G = asap_model.unroll_mat(M.copy())
        if len(G) > 0:
            means, vars_ = asap_model.ts_solver.solve(G, num_iters=8, save=True)
            stds = np.sqrt(vars_)
        else:
            means = np.zeros(n)
            stds = np.sqrt(asap_model.ts_solver.Vs)

        scene_results[scene] = {
            "conditions": conds,
            "M": M,
            "total_cmps": int(M.sum()),
            "means": means,
            "stds": stds,
            "counts": cond_cmp_counts
        }

        # Format rows for scores CSV
        ranked_indices = np.argsort(means)[::-1]
        for rank, idx in enumerate(ranked_indices, 1):
            c = conds[idx]
            all_score_rows.append({
                "Scene": scene,
                "RankInScene": rank,
                "ConditionName": c["name"],
                "Filename": c["filename"],
                "Metric": c["metric"],
                "Level": c["level"],
                "VisualQuality_Mean": f"{means[idx]:+.5f}",
                "Uncertainty_StdDev": f"{stds[idx]:.5f}",
                "ComparisonsCount": cond_cmp_counts[idx]
            })

    # 3. Export scores CSV
    if all_score_rows:
        with open(scores_csv, "w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "Scene",
                "RankInScene",
                "ConditionName",
                "Filename",
                "Metric",
                "Level",
                "VisualQuality_Mean",
                "Uncertainty_StdDev",
                "ComparisonsCount"
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_score_rows)
        print(f"[ASAP Prior Updater] Saved per-scene quality scores: {scores_csv.resolve()}\n")

    # 4. Display terminal summary
    print("================================================================================")
    print("           GAIM240 PER-SCENE ACTIVE SAMPLING (ASAP) POSTERIOR SUMMARY           ")
    print("================================================================================")
    print(f"{'Scene':<18} {'Total Cmps':<12} {'Avg Uncertainty (StdDev)':<25} {'Max Uncertainty':<16}")
    print("-" * 75)
    for scene in SCENES:
        res = scene_results[scene]
        avg_std = np.mean(res["stds"])
        max_std = np.max(res["stds"])
        print(f"{scene:<18} {res['total_cmps']:<12} {avg_std:<25.4f} {max_std:<16.4f}")
    print("================================================================================\n")

    return scene_results


def main():
    parser = argparse.ArgumentParser(description="GAIM240 ASAP Multi-Scene Prior & Score Updater")
    parser.add_argument("--results-dir", type=str, default="experiment_results", help="Directory containing observer trial logs")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    args = parser.parse_args()

    update_priors_and_scores(
        dataset_dir=Path(args.dataset),
        results_dir=Path(args.results_dir)
    )


if __name__ == "__main__":
    main()
