#!/usr/bin/env python3
"""
training/generate_training_batch.py
===================================
Generates the standardized 18-trial training batch for observer familiarization.

Structure:
- 9 distortion types x 2 scenes = 18 trials total.
- Uses maximum distortion intensity (level2) paired against the uncorrupted reference.
- Every scene in the dataset appears exactly twice.
- 50/50 counterbalanced Left/Right placement (9 left-distorted, 9 right-distorted).
- Interleaved ordering to avoid back-to-back duplicate scenes or distortion categories.
"""

import csv
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from platform_utils import get_dataset_dir

TRAINING_PLAN = [
    # (trial_id, distortion_metric, scene, distorted_side)
    (1, "dlss_rr", "attic", "LEFT"),
    (2, "duration_flicker", "classroom", "RIGHT"),
    (3, "judder", "bistro_exterior", "LEFT"),
    (4, "motion_noise", "marbles", "RIGHT"),
    (5, "motion_resolution", "bistro_interior", "LEFT"),
    (6, "noise_colors", "attic", "RIGHT"),
    (7, "restir", "pink_room", "LEFT"),
    (8, "stutter", "zeroday", "RIGHT"),
    (9, "temporal-resolution-multiplexing", "landscape", "LEFT"),
    (10, "dlss_rr", "pink_room", "RIGHT"),
    (11, "duration_flicker", "subway", "LEFT"),
    (12, "judder", "zeroday", "RIGHT"),
    (13, "motion_noise", "landscape", "LEFT"),
    (14, "motion_resolution", "subway", "RIGHT"),
    (15, "noise_colors", "classroom", "LEFT"),
    (16, "restir", "marbles", "RIGHT"),
    (17, "stutter", "bistro_interior", "LEFT"),
    (18, "temporal-resolution-multiplexing", "bistro_exterior", "RIGHT"),
]


def generate_training_csv(output_path: Path = None, subject_id: str = "TRAINING", dataset_dir: Path = None):
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    if output_path is None:
        output_path = Path(__file__).resolve().parent / "training_batch.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

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

    rows = []
    for trial_num, metric, scene, dist_side in TRAINING_PLAN:
        ref_rel = f"{scene}/reference/video0.rgb"
        ref_path = dataset_dir / scene / "reference" / "video0.rgb"

        dist_rel = f"{scene}/{metric}/level2/video0.rgb"
        dist_path = dataset_dir / scene / metric / "level2" / "video0.rgb"

        # Verify files exist
        if not ref_path.exists():
            raise FileNotFoundError(f"Missing reference video: {ref_path}")
        if not dist_path.exists():
            raise FileNotFoundError(f"Missing distortion video: {dist_path}")

        ref_condition = f"{scene}:reference"
        dist_condition = f"{scene}:{metric}_level2"

        if dist_side == "LEFT":
            left_cond = dist_condition
            left_rel = dist_rel
            left_metric = metric
            left_level = "level2"
            left_p = str(dist_path)

            right_cond = ref_condition
            right_rel = ref_rel
            right_metric = "reference"
            right_level = "reference"
            right_p = str(ref_path)
        else:
            left_cond = ref_condition
            left_rel = ref_rel
            left_metric = "reference"
            left_level = "reference"
            left_p = str(ref_path)

            right_cond = dist_condition
            right_rel = dist_rel
            right_metric = metric
            right_level = "level2"
            right_p = str(dist_path)

        rows.append({
            "TrialNumber": trial_num,
            "SubjectID": subject_id,
            "Scene": scene,
            "RefFilename": "video0.rgb",
            "RefPath": str(ref_path),
            "LeftCondition": left_cond,
            "LeftFilename": left_rel,
            "LeftMetric": left_metric,
            "LeftLevel": left_level,
            "LeftPath": left_p,
            "RightCondition": right_cond,
            "RightFilename": right_rel,
            "RightMetric": right_metric,
            "RightLevel": right_level,
            "RightPath": right_p,
        })

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[Training Batch] Successfully generated {len(rows)} trials -> {output_path}")
    return output_path


if __name__ == "__main__":
    generate_training_csv()
