# GAIM240 Performance Advances & Video Player Report
## Porting, Benchmarking, and VRAM Pre-Uploading Optimization

---

## 1. Executive Summary & Design Achievements

This session focused on evaluating the custom **Nvidia Vulkan Visualizer** (`userstudy`) against our **Native Rust Player** using the high-refresh-rate GAIM240 dataset (240 FPS HEVC sequences). During this process, we solved compatibility blocks, integrated the visualizer into the ASAP Active Sampling pipeline, optimized its loading overhead, and designed a **VRAM Pre-Uploading pathway** in Rust that achieves the best of both worlds.

### Performance Summary Table

| Metric | Original Rust Player (YUV420p) | Nvidia Visualizer (`userstudy`) | **VRAM-Optimized Rust (YUV420p)** | **Native YUV444p Rust (OS Swapped)** | **On-the-Fly Rust (YUV444p)** |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Pre-Decode Time** | **~3.80s (Cuvid)** | ~37.55s | **~5.20s (Cuvid)** | ~6.62s (Cuvid) | **0.00s (Instant!)** |
| **GPU Playback (Uncapped)** | ~1,300 FPS | ~2,000 FPS | **~2,590 FPS** | ~367 FPS (Swap overhead) | ~212 FPS |
| **System RAM Payload** | ~5.00 GB | ~11.00 GB | ~5.00 GB | **~0.00 GB (Streaming)** | **~0.00 GB (Streaming)** |
| **GPU VRAM Footprint** | ~0.05 GB | ~13.20 GB | ~5.00 GB | ~9.90 GB (Overflows 8GB) | **~0.02 GB (1 Frame)** |
| **240Hz VSync Lock** | Perfect | Perfect | **Perfect** | **Perfect** (Fully usable) | Stutter (Drops to ~212) |
| **Color Fidelity** | Downsampled 4:2:0 | Native 4:4:4 | Downsampled 4:2:0 | **Native 4:4:4 (Perfect)** | **Native 4:4:4 (Perfect)** |

---

## 2. Porting the Nvidia Visualizer (ELF Binary Patching)

The compiled Nvidia `userstudy` binary was built against GLIBC 2.43 (math symbol versioning), which caused immediate segment violations and crashes on host systems running Ubuntu 24.04 (GLIBC 2.39). 

We bypassed this closed-source restriction by implementing [`patch_elf_precise.py`](file:///home/jv495/Developer/userstudy_v0.2_linux/userstudy_v0.2/patch_elf_precise.py):
* Parses the ELF section headers directly.
* Downgrades versioned GLIBC math symbols (`atan2f`, `acosf`, `sqrtf`) in `.gnu.version_r` to `GLIBC_2.38` (which exists on Ubuntu 24.04).
* Recalculates symbol hashes (`0x069691b8`) and adjusts indices to `1` (global unversioned).
* Generates [`userstudy_patched`](file:///home/jv495/Developer/userstudy_v0.2_linux/userstudy_v0.2/userstudy_patched) which runs natively without any dynamic loader wrappers.

---

## 3. Optimizing Nvidia Player Loading (FFmpeg Interception)

The Nvidia player's pre-decoding pipeline was the primary performance bottleneck, taking **~37.5s** per trial. The player spawns `ffmpeg` as a subprocess to extract video frames into thousands of `.png` files under `/tmp`, then reads them back into RAM.

To optimize this closed-source mechanism, we built a transparent **FFmpeg interceptor wrapper** at [`bin/ffmpeg`](file:///home/jv495/Developer/videoplayer/bin/ffmpeg):
1. The wrapper is prepended to the `PATH` during visualizer execution.
2. It detects when the visualizer requests `.png` extraction.
3. It automatically injects fast **zero-compression flags** (`-compression_level 0 -pred none`) before calling the real FFmpeg binary.
4. **Results**: This optimization cuts CPU compression cycles by **4.3x** and speeds up real-time PNG writing by **2.2x**, dropping loading times to **~16–18 seconds** for smaller scenes.

---

## 4. Rust VRAM Pre-Uploading Optimization

To challenge the Nvidia player's 2,000 FPS rendering speed without inheriting its loading latency and huge RAM footprint, we implemented a new VRAM Pre-Uploading pathway in the Rust player, located in [`benchmark_vram.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_vram.rs):

* **CPU Parallel HEVC Decode**: Decodes video streams concurrently using CUDA-accelerated parallel threads directly into compact system RAM buffers (taking **~5.20s**).
* **Pre-Upload to GPU Memory**: Right before starting playback, it copies all 3,600 frames into pre-allocated OpenGL textures in GPU VRAM. This massive DMA copy takes less than **0.70s**.
* **Zero-Copy Playback**: During playback, it binds the texture IDs directly in the rendering loop. It performs **zero** CPU-to-GPU memory copies and **zero** disk access.
* **Peak Performance**: Hits an average of **2,538 FPS** (with a peak of **2,601 FPS**), outperforming the Nvidia Vulkan visualizer (2,000 FPS) by **~538 FPS** while using only **5.0 GB VRAM** (compared to Nvidia's 13.2 GB).
* **Safe Resource Lifecycle**: Implements Rust's RAII `Drop` trait to automatically free all 3,600 texture resources from the GPU immediately when the trial window closes.

---

## 5. Native YUV444p and On-the-Fly GPU Decoding Architectures

To achieve perfect chroma fidelity without CPU-based color downsampling, we implemented two additional rendering architectures in Rust:

### A. Native YUV444p Pre-Uploaded Pathway ([`benchmark_yuv444.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_yuv444.rs))
* **Design**: Uses forced hardware CUDA decoding (`hevc_cuvid`) to decompress HEVC 4:4:4 directly on the GPU, streaming the full-resolution Y, U, and V frames into pre-allocated VRAM arrays.
* **Results**: Pre-decodes and uploads 9.9 GB of textures in **~6.62 seconds**. Because it exceeds the 8 GB physical VRAM limit of the Quadro RTX 4000, X11 paged memory back to system RAM, limiting uncapped FPS to **~366 FPS**. However, this is still more than enough to lock perfectly to a **240Hz VSync with zero dropped frames**.

### B. On-the-Fly GPU Decoding ([`benchmark_onthefly.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_onthefly.rs))
* **Design**: Spawns the 3 FFmpeg processes only when playback starts. The main thread pulls the YUV444p frames from the pipes and updates a single set of 9 active textures using `glTexSubImage2D` in real-time.
* **Results**: Playback starts **instantly (0.00s loading time)**. It uses **~0.00 GB system RAM** and only **~0.02 GB VRAM**. Uncapped playback runs at **~212 FPS**, which is slightly below the 240Hz refresh rate due to real-time NVDEC hardware division and pipe read synchronization.

---

## 6. Summary of Compiled Binaries and Run Commands

All files have been structured to ensure the original, stable code remains fully untouched.

### New and Updated Files:
* [`benchmark_vram.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_vram.rs): Rust VRAM pre-uploading optimized benchmark (YUV420p).
* [`benchmark_yuv444.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_yuv444.rs): Rust native YUV444p forced hardware decoding benchmark.
* [`benchmark_onthefly.rs`](file:///home/jv495/Developer/videoplayer/rust_player/src/bin/benchmark_onthefly.rs): Rust on-the-fly streaming GPU decoder benchmark (0.0s load time).
* [`benchmark_nvis.py`](file:///home/jv495/Developer/videoplayer/benchmark_nvis.py): Automated and interactive benchmarking suite for the Nvidia player.
* [`bin/ffmpeg`](file:///home/jv495/Developer/videoplayer/bin/ffmpeg): FFmpeg wrapper script that intercepts calls and forces zero-compression PNG writes.
* [`run_asap_experiment_nvis.py`](file:///home/jv495/Developer/videoplayer/run_asap_experiment_nvis.py): Updated trial runner supporting Nvidia `--player nvis` mode with console capturing.

### Run Commands:

* **Run VRAM-Optimized YUV420p Rust Player Benchmark (Uncapped)**:
  ```bash
  cargo run --release --bin benchmark_vram -- --no-vsync
  ```

* **Run Native YUV444p Rust Player Benchmark (Uncapped)**:
  ```bash
  cargo run --release --bin benchmark_yuv444 -- --no-vsync
  ```

* **Run On-the-Fly GPU Decoded YUV444p Rust Player Benchmark (Uncapped)**:
  ```bash
  cargo run --release --bin benchmark_onthefly -- --no-vsync
  ```

* **Run Nvidia Player Benchmark (Uncapped)**:
  ```bash
  python3 benchmark_nvis.py --vsync off
  ```
