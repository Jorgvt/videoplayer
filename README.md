# GAIM240 Video Quality Comparer

A dual-mode, high-fidelity video comparison tool designed specifically for visual quality inspection and temporal artifact comparison of the GAIM240 video dataset.

Because the videos in this dataset are **240 FPS** with **extremely high bitrates** (e.g. 600 MB - 1 GB for a 5-second video, equivalent to ~1 Gbps), playing them back in real-time can choke standard video decoders. This codebase provides high-performance web and native Rust video players to solve this problem.

---

## Active Sampling Perception Experiment (ASAP Multi-Scene Batch Mode + Rust 240Hz)

We incorporate **Active SAmpling for Pairwise comparisons (ASAP)** ([gfxdisp/asap](https://github.com/gfxdisp/asap), Mikhailiuk et al., ICPR 2020) to run efficient, high-information perception experiments.

### Experiment Design & Architecture
* **9 Independent ASAP Models**: Because comparisons are strictly intra-scene, each of the 9 scenes maintains an independent ASAP model and comparison matrix $M_{\text{scene}}$ ($27 \times 27$).
* **Batch Presentation via Minimum Spanning Tree (MST)**: For each observer, ASAP generates an optimal MST of $N - 1 = 26$ pairs per scene ($26 \times 9 = 234$ total presentation trials) maximizing expected information gain while guaranteeing graph connectivity.
* **Stratified Multi-Scene Interleaving**: Trials from all 9 scenes are interleaved pseudo-randomly (max 2 consecutive trials per scene) with 50/50 spatial counterbalancing (Left/Right assignment).
* **Sequential Prior Updates**: When an observer finishes their batch, all results are aggregated, TrueSkill posteriors are re-fitted, and the updated priors guide batch selection for subsequent observers.

For full theoretical background, data schemas, and experimental protocol, see **[docs/ASAP_BATCH_EXPERIMENT.md](docs/ASAP_BATCH_EXPERIMENT.md)**.

### Quick Start:
```bash
# 1. Run full session for Observer P01 (generates batch -> 240Hz presentation -> updates priors):
uv run python run_asap_experiment.py --subject P01

# 2. Run session for Observer P02 (automatically conditions on P01's results):
uv run python run_asap_experiment.py --subject P02

# Modular decoupled commands:
uv run python asap_batch_generator.py --subject P01      # Generate batch CSV
uv run python run_batch_experiment.py --subject P01      # Run 240Hz player with auto-resume
uv run python asap_prior_updater.py                      # Re-fit scores and update history
```

### Key Technical Highlights:
* **Instant Fast EIG Selection (< 1ms)**: Analytical Information Gain calculation eliminates inter-trial delay.
* **Zero GPU VRAM Overhead**: Active sampling calculations run entirely on **CPU (NumPy/SciPy)**, keeping **100% of GPU VRAM and CUDA engines dedicated to `rust_player`** for locked 240Hz presentation.
* **Dedicated 240Hz Trial Engine (`asap_trial.rs`)**: Dedicated Rust binary for 2AFC trials, leaving `main.rs` perception player untouched.
* **Outputs**: Saves trial history to `experiment_results/[subject_id]_asap_global_trials.csv` and scale scores to `[subject_id]_asap_global_scores.csv`.

---

## Native Rust 240Hz Player & Experiment Suite

The native Rust visualizer delivers **locked 239.76 FPS** presentation with sub-millisecond GPU draw times ($0.59\text{ ms}$) and instant pre-decoding ($\sim 4.5\text{s}$ per trial).

### Architecture Highlights:
* **CUDA Pipelined YUV420P/YUV444P Decoding**: Decodes HEVC bitstreams into contiguous Planar YUV memory buffers in **~4.5s** (reducing memory transfer payload per video from $3.32\text{ GB}$ to $1.38\text{ GB}$).
* **GPU YUV2RGB Shader**: Performs color matrix conversion in VRAM via GLSL 120 fragment shaders, reaching **>1,300 FPS** uncapped throughput.
* **Nanosecond Frame Pacer**: Drift-compensating high-precision spin loop for exact $240.000\text{ FPS}$ locked presentation (`--pacer`).
* **Zero Quality Loss**: Preserves 100% bit-exact HEVC raytracing fidelity without chroma degradation or re-encoding.

### How to Run

Navigate to the `rust_player` directory or run with `cargo`:

* **Run Interactive Perception Experiment**:
  ```bash
  cargo run --manifest-path rust_player/Cargo.toml --release --bin rust_player -- --subject=P01 --pacer
  ```
* **Run Dedicated ASAP Single Trial Engine**:
  ```bash
  cargo run --manifest-path rust_player/Cargo.toml --release --bin asap_trial -- --left=<left_path> --ref=<ref_path> --right=<right_path> --out=res.csv --pacer
  ```
* **Run Multi-Stage Performance Profiler**:
  ```bash
  cargo run --manifest-path rust_player/Cargo.toml --release --bin profile -- --scene=marbles --pacer
  ```
* **Run Automated 72-Pass Benchmark Suite**:
  ```bash
  cargo run --manifest-path rust_player/Cargo.toml --release --bin benchmark -- --no-vsync
  ```

---

## Codebase File Map

* **[all_trials_bank.csv](all_trials_bank.csv)** – Master CSV database containing all **3,159 valid intra-scene comparison pairs** across the dataset.
* **[run_asap_experiment.py](run_asap_experiment.py)** – Active Sampling experiment controller integrating `gfxdisp/asap` (Global default & Intra modes) with `asap_trial`.
* **[platform_utils.py](platform_utils.py)** – Cross-platform path resolver and dynamic library environment configuration.
* **[rust_player/src/main.rs](rust_player/src/main.rs)** – Primary Rust 240Hz full perception experiment runner.
* **[rust_player/src/bin/asap_trial.rs](rust_player/src/bin/asap_trial.rs)** – Dedicated Rust 240Hz single-trial visualizer binary for 2AFC active sampling.
* **[rust_player/src/bin/](rust_player/src/bin/)** – Benchmark and profiler binaries (`benchmark.rs`, `profile.rs`, `benchmark_lz4.rs`, etc.).
* **[experiment_results/](experiment_results/)** – Real participant trial logs and estimated JND scale scores.
* **[benchmarks/](benchmarks/)** – Benchmark evaluation scripts, profiler logs, and visualization outputs.
* **[docs/](docs/)** – Architecture specifications ([docs/SYSTEM_ARCHITECTURE_240HZ.md](docs/SYSTEM_ARCHITECTURE_240HZ.md)), design options ([docs/ARCHITECTURAL_OPTIONS.md](docs/ARCHITECTURAL_OPTIONS.md)), and OS-specific performance research.
* **[asap/](asap/)** – Upstream Active Sampling for Pairwise Comparisons repository clone.
* **[README.md](README.md)** – This guide.
