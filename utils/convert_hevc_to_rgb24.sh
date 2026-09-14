#!/usr/bin/env bash
# Convert all lossless HEVC MP4 files to raw RGB24 (.rgb) using NVDEC
# Source: /mnt/wdblack2tb/all_sequences_new_lossless
# Dest:   /mnt/wdblack2tb/all_sequences_new_lossless_raw
# Mirrors full directory structure; replaces video0.mp4 → video0.rgb
# Decodes via hevc_cuvid (NVDEC) for maximum throughput
# 6 parallel jobs

SRC_ROOT="/mnt/wdblack2tb/all_sequences_new_lossless"
DST_ROOT="/mnt/wdblack2tb/all_sequences_new_lossless_raw"
PARALLEL=6
LOG_FILE="/tmp/convert_rgb24_$(date +%Y%m%d_%H%M%S).log"

echo "=== RGB24 Conversion Started: $(date) ===" | tee "$LOG_FILE"
echo "Source:   $SRC_ROOT" | tee -a "$LOG_FILE"
echo "Dest:     $DST_ROOT" | tee -a "$LOG_FILE"
echo "Parallel: $PARALLEL jobs" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Collect all mp4 files
mapfile -t MP4_FILES < <(find "$SRC_ROOT" -name "*.mp4" | sort)
TOTAL=${#MP4_FILES[@]}
echo "Total files: $TOTAL" | tee -a "$LOG_FILE"

convert_one() {
    local src_mp4="$1"
    local rel_path="${src_mp4#$SRC_ROOT/}"      # e.g. attic/reference/video0.mp4
    local rel_dir="$(dirname "$rel_path")"       # e.g. attic/reference
    local base="$(basename "$rel_path" .mp4)"    # e.g. video0
    local dst_dir="$DST_ROOT/$rel_dir"
    local dst_rgb="$dst_dir/${base}.rgb"

    # Skip if already converted
    if [[ -f "$dst_rgb" ]]; then
        echo "[SKIP] $rel_path (already exists)" | tee -a "$LOG_FILE"
        return 0
    fi

    mkdir -p "$dst_dir"

    local start_ts=$(date +%s%N)

    # Decode: NVDEC → CUDA surface → download to CPU as RGB24 → raw binary output
    ffmpeg -y \
        -hwaccel cuda \
        -hwaccel_output_format cuda \
        -c:v hevc_cuvid \
        -i "$src_mp4" \
        -vf "hwdownload,format=yuv444p" \
        -pix_fmt rgb24 \
        -f rawvideo \
        "$dst_rgb" 2>>"$LOG_FILE"

    local exit_code=$?
    local end_ts=$(date +%s%N)
    local elapsed_ms=$(( (end_ts - start_ts) / 1000000 ))

    if [[ $exit_code -eq 0 ]]; then
        local size_gb=$(du -sh "$dst_rgb" | cut -f1)
        echo "[OK]   $rel_path → $size_gb in ${elapsed_ms}ms" | tee -a "$LOG_FILE"
    else
        echo "[FAIL] $rel_path (exit $exit_code)" | tee -a "$LOG_FILE"
        rm -f "$dst_rgb"   # remove partial output
    fi
    return $exit_code
}

export -f convert_one
export SRC_ROOT DST_ROOT LOG_FILE

# Run with GNU parallel (or xargs fallback)
if command -v parallel &>/dev/null; then
    printf '%s\n' "${MP4_FILES[@]}" | parallel -j "$PARALLEL" convert_one {}
else
    printf '%s\n' "${MP4_FILES[@]}" | xargs -P "$PARALLEL" -I{} bash -c 'convert_one "$@"' _ {}
fi

echo "" | tee -a "$LOG_FILE"
echo "=== Conversion Complete: $(date) ===" | tee -a "$LOG_FILE"

# Summary
OK=$(grep -c '^\[OK\]' "$LOG_FILE" || true)
SKIP=$(grep -c '^\[SKIP\]' "$LOG_FILE" || true)
FAIL=$(grep -c '^\[FAIL\]' "$LOG_FILE" || true)
echo "  OK:      $OK / $TOTAL" | tee -a "$LOG_FILE"
echo "  Skipped: $SKIP" | tee -a "$LOG_FILE"
echo "  Failed:  $FAIL" | tee -a "$LOG_FILE"
echo "  Log:     $LOG_FILE" | tee -a "$LOG_FILE"
