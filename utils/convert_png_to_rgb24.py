#!/usr/bin/env python3
"""
Convert duration_flicker PNG sequences → raw RGB24 video0.rgb files,
slotted into the existing all_sequences_new_lossless_raw directory tree.

Output structure:
  /mnt/wdblack2tb/all_sequences_new_lossless_raw/{scene}/duration_flicker/{level}/video0.rgb

Frames are sorted by filename and concatenated as raw RGB24 binary
(1280×720×3 bytes per frame). Variable frame counts are fine.
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image

SRC_ROOT = Path("/mnt/wdblack2tb/duration_flicker_dataset_png")
DST_ROOT = Path("/mnt/wdblack2tb/all_sequences_new_lossless_raw")
DISTORTION = "duration_flicker"
WORKERS = 6


LEVEL_MAP = {
    "low": "level0",
    "medium": "level1",
    "high": "level2",
}


def convert_sequence(args):
    scene, level, src_dir, dst_file = args
    tag = f"{scene}/{DISTORTION}/{level}"

    if dst_file.exists():
        return tag, "SKIP", dst_file.stat().st_size // (1280 * 720 * 3), 0

    png_files = sorted(src_dir.glob("*.png"))
    if not png_files:
        return tag, "EMPTY", 0, 0

    dst_file.parent.mkdir(parents=True, exist_ok=True)

    n_frames = 0
    try:
        with open(dst_file, "wb") as out:
            for png_path in png_files:
                img = Image.open(png_path).convert("RGB")
                out.write(img.tobytes())
                n_frames += 1
    except Exception as e:
        dst_file.unlink(missing_ok=True)
        return tag, f"FAIL: {e}", 0, 0

    return tag, "OK", n_frames, dst_file.stat().st_size


def main():
    # Collect all (scene, level) pairs
    jobs = []
    for scene_dir in sorted(SRC_ROOT.iterdir()):
        if not scene_dir.is_dir():
            continue
        scene = scene_dir.name
        for level_dir in sorted(scene_dir.iterdir()):
            if not level_dir.is_dir():
                continue
            src_level = level_dir.name
            dst_level = LEVEL_MAP.get(src_level, src_level)
            dst_file = DST_ROOT / scene / DISTORTION / dst_level / "video0.rgb"
            jobs.append((scene, dst_level, level_dir, dst_file))

    total = len(jobs)
    print(f"=== PNG → RGB24 Conversion: {DISTORTION} ===")
    print(f"Source : {SRC_ROOT}")
    print(f"Dest   : {DST_ROOT}/{{scene}}/{DISTORTION}/{{level}}/video0.rgb")
    print(f"Jobs   : {total}  |  Workers: {WORKERS}")
    print()

    ok = skip = fail = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(convert_sequence, j): j for j in jobs}
        done = 0
        for fut in as_completed(futures):
            tag, status, n_frames, size = fut.result()
            done += 1
            size_gb = size / 1024**3 if size else 0
            if status == "OK":
                ok += 1
                print(f"[{done:2d}/{total}] OK    {tag}  ({n_frames} frames, {size_gb:.2f} GB)")
            elif status == "SKIP":
                skip += 1
                print(f"[{done:2d}/{total}] SKIP  {tag}  (already exists, {n_frames} frames)")
            else:
                fail += 1
                print(f"[{done:2d}/{total}] {status}  {tag}")
            sys.stdout.flush()

    print()
    print(f"=== Done: {ok} OK, {skip} skipped, {fail} failed ===")


if __name__ == "__main__":
    main()
