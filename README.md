# GAIM240 Video Quality Comparer

A dual-mode, high-fidelity video comparison tool designed specifically for visual quality inspection and temporal artifact comparison of the GAIM240 video dataset.

Because the videos in this dataset are **240 FPS** with **extremely high bitrates** (e.g. 600 MB - 1 GB for a 5-second video, equivalent to ~1 Gbps), playing them back in real-time can choke standard video decoders. This codebase provides high-performance web and native Rust video players to solve this problem.

---

## Active Sampling Perception Experiment (ASAP + Rust 240Hz)

We incorporate **Active SAmpling for Pairwise comparisons (ASAP)** ([gfxdisp/asap](https://github.com/gfxdisp/asap), Mikhailiuk et al., ICPR 2020) to reduce participant fatigue by **50% to 70%**.

Instead of evaluating all static trials, ASAP dynamically selects the most informative video pair $(i, j)$ maximizing Expected Information Gain (Entropy Reduction).

### Dataset & Distortion Space:
* **9 Scenes**: `attic`, `bistro_exterior`, `bistro_interior`, `classroom`, `landscape`, `marbles`, `pink_room`, `subway`, `zeroday`.
* **9 Distortions (3 levels each: `level0`, `level1`, `level2`)**:
  1. `dlss_rr`
  2. `duration_flicker` *(new)*
  3. `judder`
  4. `motion_noise`
  5. `motion_resolution`
  6. `noise_colors`
  7. `restir`
  8. `stutter`
  9. `temporal-resolution-multiplexing`
* **Total Conditions**: **243 conditions** (9 scenes $\times$ 9 distortions $\times$ 3 intensity levels = 243 distorted video conditions + 9 scene references = 252 videos).

### Candidate Pair Space:
* **`all_trials_bank.csv`**: Contains all **3,159 valid intra-scene candidate pairs** across the 9 scenes ($9 \text{ scenes} \times 351 \text{ pairs/scene} = 3,159$).
  * **Intra-Scene, Intra-Distortion**: 243 pairs ($9 \text{ scenes} \times 9 \text{ distortions} \times 3 \text{ level pairs} = 243$).
  * **Intra-Scene, Inter-Distortion**: 2,916 pairs ($9 \text{ scenes} \times \binom{9}{2} \times (3 \times 3) = 2,916$).
* **Intra-Scene, Inter-Distortion**: Cross-metric comparisons within the same scene (e.g., `pink_room:dlss_rr_level1` vs `pink_room:judder_level2`).
* **Intra-Scene, Intra-Distortion**: Same-metric comparisons within the same scene (e.g., `pink_room:dlss_rr_level1` vs `pink_room:dlss_rr_level2`).

### Modes:
1. **Global Mode (DEFAULT, `--mode=global`)**:
   * Pools all **243 video conditions across all 9 scenes** to construct a single, unified **Global JND Visual Quality Scale**.
2. **Intra-Scene Mode (`--mode=intra --scene=marbles`)**:
   * Restricts sampling to conditions within a single scene.

### How to Run:
```bash
# Run Global Active Sampling (Default, 30 new adaptive trials):
uv run run_asap_experiment.py --subject=P01 --trials=30

# Run Single-Scene Intra Active Sampling (Scene: marbles):
uv run run_asap_experiment.py --subject=P01 --mode=intra --scene=marbles --trials=25
```

### Key Technical Highlights:
* **Instant Fast EIG Selection (< 1ms)**: Analytical Information Gain calculation eliminates inter-trial delay.
* **Zero GPU VRAM Overhead**: Active sampling calculations run entirely on **CPU (NumPy/SciPy)**, keeping **100% of GPU VRAM and CUDA engines dedicated to `rust_player`** for locked 240Hz presentation.
* **Dedicated 240Hz Trial Engine (`asap_trial.rs`)**: Dedicated Rust binary for 2AFC trials, leaving `main.rs` perception player untouched.
* **Outputs**: Saves trial history to `experiment_results/[subject_id]_asap_global_trials.csv` and scale scores to `[subject_id]_asap_global_scores.csv`.

---

## Option 1: Web-Based Player (Python server + Local Browser)

This option serves a modern web application running locally. It decodes videos natively in your browser using hardware acceleration.

### How to Run
From the workspace folder, run:
```bash
uv run app.py
```
This automatically starts the server at `http://127.0.0.1:8000` and opens your default browser.

---

## Option 2: Native Rust 240Hz Player & Benchmark Suite (Recommended for 100% Smooth Playback)

The native Rust visualizer delivers **locked 239.76 FPS** presentation with sub-millisecond GPU draw times ($0.59\text{ ms}$) and instant pre-decoding ($\sim 4.5\text{s}$ per trial).

### Architecture Highlights:
* **CUDA Pipelined YUV420P Decoding**: Decodes HEVC bitstreams into contiguous Planar YUV memory buffers in **~4.5s** (reducing memory transfer payload per video from $3.32\text{ GB}$ to $1.38\text{ GB}$).
* **GPU YUV2RGB Shader**: Performs color matrix conversion in VRAM via GLSL 120 fragment shaders, reaching **>1,300 FPS** uncapped throughput.
* **Nanosecond Frame Pacer**: Drift-compensating high-precision spin loop for exact $240.000\text{ FPS}$ locked presentation (`--pacer`).
* **Zero Quality Loss**: Preserves 100% bit-exact HEVC 4:4:4 raytracing fidelity without chroma degradation or re-encoding.

### How to Run

Navigate to the `rust_player` directory:
```bash
cd rust_player
```

* **Run Interactive Perception Experiment**:
  ```bash
  cargo run --release --bin rust_player -- --subject=P01 --pacer
  ```
* **Run Dedicated ASAP Single Trial Engine**:
  ```bash
  cargo run --release --bin asap_trial -- --left=<left_path> --ref=<ref_path> --right=<right_path> --out=res.csv --pacer
  ```
* **Run Multi-Stage Performance Profiler**:
  ```bash
  cargo run --release --bin profile -- --scene=marbles --pacer
  ```
* **Run Automated 72-Pass Benchmark Suite**:
  ```bash
  cargo run --release --bin benchmark -- --no-vsync
  ```

---

## Codebase File Map

* **[all_trials_bank.csv](all_trials_bank.csv)** – Master CSV database containing all **3,159 valid intra-scene comparison pairs** across the dataset.
* **[run_asap_experiment.py](run_asap_experiment.py)** – Active Sampling experiment controller integrating `gfxdisp/asap` (Global default & Intra modes) with `asap_trial`.
* **[rust_player/src/bin/asap_trial.rs](rust_player/src/bin/asap_trial.rs)** – Dedicated Rust 240Hz single-trial visualizer binary for 2AFC active sampling.
* **[rust_player/src/main.rs](rust_player/src/main.rs)** – Primary Rust 240Hz full perception experiment runner.
* **[rust_player/src/bin/benchmark.rs](rust_player/src/bin/benchmark.rs)** – Automated 72-pass performance benchmark suite.
* **[rust_player/src/bin/profile.rs](rust_player/src/bin/profile.rs)** – Multi-stage timing profiler measuring fetch, GPU upload, and flip durations.
* **[SYSTEM_ARCHITECTURE_240HZ.md](SYSTEM_ARCHITECTURE_240HZ.md)** – Technical architecture specification document.
* **[README.md](README.md)** – This guide.
