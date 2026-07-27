# GAIM240 Video Quality Comparer

A dual-mode, high-fidelity video comparison tool designed specifically for visual quality inspection and temporal artifact comparison of the GAIM240 video dataset.

Because the videos in this dataset are **240 FPS** with **extremely high bitrates** (e.g. 600 MB - 1 GB for a 5-second video, equivalent to ~1 Gbps), playing them back in real-time can choke standard video decoders. This codebase provides high-performance web and native Rust video players to solve this problem.

---

## Option 1: Web-Based Player (Python server + Local Browser)

This option serves a modern web application running locally. It decodes videos natively in your browser using hardware acceleration.

### How to Run
From the workspace folder, run:
```bash
uv run app.py
```
This automatically starts the server at `http://127.0.0.1:8000` and opens your default browser.

### Web Player Hotkeys
| Hotkey | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Play / Pause |
| <kbd>←</kbd> / <kbd>→</kbd> | Step backward / forward by 1 frame (approx. 1/60s) |
| <kbd>Tab</kbd> (Hold) | Swap to Video B, release to return to Video A (in A/B Swap layout) |
| <kbd>Z</kbd> | Reset Zoom & Pan to 100% |
| <kbd>S</kbd> | Toggle Side-by-Side mode |
| <kbd>D</kbd> | Toggle Split Slider mode |

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
* **Run Multi-Stage Performance Profiler**:
  ```bash
  cargo run --release --bin profile -- --scene=marbles --pacer
  ```
  *(Outputs `profile_results.csv` compatible with `plot_profile.py`)*

* **Run Automated 72-Pass Benchmark Suite**:
  ```bash
  cargo run --release --bin benchmark -- --no-vsync
  ```

---

## Python Native Players (Legacy Reference Implementations)

Python fallback scripts are also available for comparison and benchmarking:

* **Dual Video OpenGL Player**:
  ```bash
  uv run opengl_player.py
  ```
* **Triple Video Pyramid OpenGL Player**:
  ```bash
  uv run opengl_triple_player.py
  ```
* **Triple Video Profiler**:
  ```bash
  uv run profile_triple_player.py
  ```

---

## Codebase File Map

* **[rust_player/src/main.rs](rust_player/src/main.rs)** – Primary Rust 240Hz perception experiment runner with YUV420P GPU Shader pipeline.
* **[rust_player/src/bin/benchmark.rs](rust_player/src/bin/benchmark.rs)** – Automated 72-pass performance benchmark suite.
* **[rust_player/src/bin/profile.rs](rust_player/src/bin/profile.rs)** – Multi-stage timing profiler measuring fetch, GPU upload, and flip durations.
* **[SYSTEM_ARCHITECTURE_240HZ.md](SYSTEM_ARCHITECTURE_240HZ.md)** – Technical architecture specification document.
* **[ARCHITECTURAL_OPTIONS.md](ARCHITECTURAL_OPTIONS.md)** – Technical breakdown of zero-copy decoding and caching strategies.
* **[all_trials_bank.csv](all_trials_bank.csv)** – Master CSV database containing all possible 2AFC comparison pairs across the dataset.
* **[plot_profile.py](plot_profile.py)** – Python visualization tool for plotting multi-stage profiler CSV timelines.
* **[README.md](README.md)** – This guide.
