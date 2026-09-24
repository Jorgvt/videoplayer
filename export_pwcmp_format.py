#!/usr/bin/env python3
"""
export_pwcmp_format.py
======================
Exports gathered experimental results into Rafal Mantiuk's `pwcmp` package format
(both CSV and MATLAB .mat files) to share directly with collaborators.

Outputs:
1. `experiment_results/pwcmp_data.csv` - Standard pwcmp CSV table
2. `experiment_results/pwcmp_data.mat` - MATLAB .mat format with comparison matrices & metadata
3. `experiment_results/run_pwcmp_matlab.m` - Ready-to-run MATLAB script for pwcmp scaling
"""

import argparse
import csv
from pathlib import Path
import numpy as np
import scipy.io

from asap_batch_generator import SCENES, discover_scene_conditions
from platform_utils import get_dataset_dir
from asap_prior_updater import aggregate_all_observer_trials


def clean_cond_name(cond_str: str) -> str:
    """Removes scene prefix from condition string (e.g. 'attic:restir_level0' -> 'restir_level0')."""
    if ":" in cond_str:
        return cond_str.split(":", 1)[1]
    return cond_str


def parse_metric_and_level(cond_str: str):
    """Extracts (metric, level) from clean condition name (e.g. 'restir_level0' -> ('restir', 'level0'))."""
    clean = clean_cond_name(cond_str)
    if "_level" in clean:
        parts = clean.rsplit("_level", 1)
        return parts[0], f"level{parts[1]}"
    return clean, "level0"


def export_pwcmp(
    dataset_dir: Path = None,
    results_dir: Path = Path("experiment_results"),
    output_csv: Path = Path("experiment_results/pwcmp_data.csv"),
    output_mat: Path = Path("experiment_results/pwcmp_data.mat"),
    matlab_script: Path = Path("experiment_results/run_pwcmp_matlab.m")
):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()

    results_dir.mkdir(parents=True, exist_ok=True)
    all_trials = aggregate_all_observer_trials(results_dir)
    print(f"[Export pwcmp] Loaded {len(all_trials)} aggregated trials.")

    if not all_trials:
        print("[Export pwcmp] No trials found in experiment_results/")
        return

    # 1. Prepare and save CSV in pwcmp format
    # pwcmp standard columns: observer, scene, condition_A, condition_B, is_A_selected
    csv_rows = []
    for t in all_trials:
        sub = t.get("SubjectID", "")
        scene = t.get("Scene", "")
        c_left = t.get("LeftCondition", "")
        c_right = t.get("RightCondition", "")
        c_win = t.get("ChosenCondition", "")
        side = t.get("ChosenSide", "").upper()
        rt = float(t.get("ResponseTime_sec", 0.0))

        cond_a = clean_cond_name(c_left)
        cond_b = clean_cond_name(c_right)
        is_a_selected = 1 if side == "LEFT" or c_win == c_left else 0

        metric_a, level_a = parse_metric_and_level(cond_a)
        metric_b, level_b = parse_metric_and_level(cond_b)

        csv_rows.append({
            "observer": sub,
            "scene": scene,
            "condition_A": cond_a,
            "condition_B": cond_b,
            "is_A_selected": is_a_selected,
            "metric_A": metric_a,
            "level_A": level_a,
            "metric_B": metric_b,
            "level_B": level_b,
            "response_time_sec": f"{rt:.4f}"
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "observer",
            "scene",
            "condition_A",
            "condition_B",
            "is_A_selected",
            "metric_A",
            "level_A",
            "metric_B",
            "level_B",
            "response_time_sec"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"[Export pwcmp] Saved CSV format: {output_csv.resolve()}")

    # 2. Build MATLAB .mat structures and matrices
    mat_dict = {
        "scenes": np.array(SCENES, dtype=object),
        "total_trials": len(csv_rows),
    }

    # Store per-scene pairwise matrices M and condition lists
    scene_matrices = {}
    scene_condition_names = {}

    for scene in SCENES:
        conds, cond_map = discover_scene_conditions(dataset_dir, scene)
        n = len(conds)
        M = np.zeros((n, n), dtype=np.int32)
        cond_names_clean = [clean_cond_name(c["name"]) for c in conds]

        for r in csv_rows:
            if r["scene"] == scene:
                ca = r["condition_A"]
                cb = r["condition_B"]
                if ca in cond_names_clean and cb in cond_names_clean:
                    ia = cond_names_clean.index(ca)
                    ib = cond_names_clean.index(cb)
                    if r["is_A_selected"] == 1:
                        M[ia, ib] += 1
                    else:
                        M[ib, ia] += 1

        scene_matrices[f"M_{scene}"] = M
        scene_condition_names[f"conds_{scene}"] = np.array(cond_names_clean, dtype=object)

    mat_dict.update(scene_matrices)
    mat_dict.update(scene_condition_names)

    scipy.io.savemat(str(output_mat), mat_dict)
    print(f"[Export pwcmp] Saved MATLAB .mat format: {output_mat.resolve()}")

    # 3. Create MATLAB helper script for collaborators
    matlab_code = f"""% run_pwcmp_matlab.m
% Automatically generated script to scale GAIM240 pairwise comparison data using pwcmp.
% Requires pwcmp from https://github.com/mantiuk/pwcmp

clear; clc;

csv_file = 'pwcmp_data.csv';
fprintf('Loading pairwise comparison results from %s...\\n', csv_file);
T = readtable(csv_file);

disp('Dataset Overview:');
summary(T);

% If pwcmp toolbox is on the path, perform scaling per scene:
if exist('pw_scale', 'file') || exist('pw_scale_bml', 'file')
    scenes = unique(T.scene);
    for s = 1:length(scenes)
        cur_scene = scenes{{s}};
        fprintf('\\n=== Scaling Scene: %s ===\\n', cur_scene);
        T_scene = T(strcmp(T.scene, cur_scene), :);
        
        % Run pwcmp scaling (Thurstone Case V / BML / B-T)
        try
            [scale_jod, stats] = pw_scale(T_scene);
            disp(stats);
        catch ME
            fprintf('Error during scaling: %s\\n', ME.message);
        end
    end
else
    fprintf('\\nNOTE: pwcmp toolbox is not detected on your MATLAB path.\\n');
    fprintf('Please clone https://github.com/mantiuk/pwcmp and add it to your MATLAB path.\\n');
end
"""
    with open(matlab_script, "w", encoding="utf-8") as f:
        f.write(matlab_code)

    print(f"[Export pwcmp] Saved MATLAB starter script: {matlab_script.resolve()}\n")
    print(f"Successfully exported {len(csv_rows)} comparisons across {len(SCENES)} scenes.")


def main():
    parser = argparse.ArgumentParser(description="Export GAIM240 Perception Trials into pwcmp format")
    parser.add_argument("--results-dir", type=str, default="experiment_results", help="Directory containing observer trial logs")
    parser.add_argument("--dataset", type=str, default=str(get_dataset_dir()), help="Path to GAIM240 dataset")
    parser.add_argument("--csv-out", type=str, default="experiment_results/pwcmp_data.csv", help="Path to output pwcmp CSV")
    parser.add_argument("--mat-out", type=str, default="experiment_results/pwcmp_data.mat", help="Path to output MATLAB .mat")
    args = parser.parse_args()

    export_pwcmp(
        dataset_dir=Path(args.dataset),
        results_dir=Path(args.results_dir),
        output_csv=Path(args.csv_out),
        output_mat=Path(args.mat_out)
    )


if __name__ == "__main__":
    main()
