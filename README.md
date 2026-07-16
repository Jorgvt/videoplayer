# GAIM240 Video Quality Comparer

A dual-mode, high-fidelity video comparison tool designed specifically for visual quality inspection and temporal artifact comparison of the GAIM240 video dataset.

Because the videos in this dataset are **240 FPS** with **extremely high bitrates** (e.g. 600 MB - 1 GB for a 5-second video, equivalent to ~1 Gbps), playing them back in real-time can choke standard video decoders. This codebase provides two highly optimized players to solve this problem.

---

## Option 1: Web-Based Player (Python server + Local Browser)

This option serves a gorgeous, modern web application running locally. It decodes videos natively in your browser using hardware acceleration.

### How to Run
From the workspace folder, run:
```bash
uv run app.py
```
This automatically downloads FastAPI/Uvicorn, starts the server at `http://127.0.0.1:8000`, and opens your default browser.

### Why it was choppy (Fixed!)
We resolved a major synchronization bug. Previously, the playback sync engine was performing frame-level corrections on differences larger than `0.05s`. On high-bitrate files, this forced the browser to constantly flush its decoder buffer and perform active seek operations (leading to freezes). The threshold has been relaxed to `0.3s`, making playback **significantly smoother**.

### Web Player Hotkeys
Ensure your browser window is focused (and not currently typing in a dropdown/select menu):

| Hotkey | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Play / Pause |
| <kbd>←</kbd> / <kbd>→</kbd> | Step backward / forward by 1 frame (approx. 1/60s) |
| <kbd>Tab</kbd> (Hold) | Swap to Video B, release to return to Video A (in A/B Swap layout) |
| <kbd>Z</kbd> | Reset Zoom & Pan to 100% |
| <kbd>S</kbd> | Toggle Side-by-Side mode |
| <kbd>D</kbd> | Toggle Split Slider mode |

*Note: For perfect 240 FPS visual inspection, we recommend setting the speed to **0.25x** in the dropdown. This plays back the 240 FPS frames at a smooth, real-time 60 FPS on standard monitors.*

---

## Option 2: Native RAM-Buffered Player (Recommended for 100% Smooth Playback)

If the web browser still struggles with H.264 decoding performance at high speeds, you can use the native Python players: **`native_player.py`**, the new **`triple_player.py`**, or the full PySide6 GUI **`gui_player.py`**.

These players **preload all 1,200 frames directly into your system RAM** (taking ~5 seconds upon selection). Once in RAM, playback is 100% fluid at any speed because it bypasses all disk read and decoding bottlenecks during playback!

### How to Run
From the workspace folder, run:
* **Terminal-based Dual Player**:
  ```bash
  uv run native_player.py
  ```
* **Terminal-based Triple Player (Pyramid View)**:
  ```bash
  uv run triple_player.py
  ```
* **Full Desktop Graphical GUI Player (Recommended)**:
  ```bash
  uv run gui_player.py
  ```
This automatically fetches the required dependencies (`PySide6`, `opencv-python`, `numpy`) and opens the corresponding native player.

### Features
* **RAM Buffering**: Loads frames into memory for perfectly smooth, stutter-free playback.
* **Pixel-Perfect Zoom & Pan**: Scroll the mouse wheel to zoom in/out, and click & drag to pan around. Zooming uses **Nearest Neighbor** scaling to keep pixel boundaries razor-sharp for denoising and rendering quality inspection.
* **Layout Toggles**: Cycle through Side-by-Side, Overlay (A/B Swap), and Single track views.

### Keyboard & Mouse Controls
When the native window is focused:

| Control | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Play / Pause |
| <kbd>←</kbd> / <kbd>→</kbd> | Step backward / forward by 1 frame |
| <kbd>1</kbd>, <kbd>2</kbd>, <kbd>3</kbd>, <kbd>4</kbd> | Load Scene: `marbles` (1), `pink_room` (2), `subway` (3), `zeroday` (4) |
| <kbd>0</kbd>, <kbd>1</kbd>, <kbd>2</kbd> | Load distortion level for Video B (Level 0, 1, or 2) |
| <kbd>M</kbd> / <kbd>N</kbd> | Cycle through quality metrics (Next / Previous) |
| <kbd>L</kbd> | Cycle through Layouts (Side-by-Side, Overlay, Single A, Single B) |
| <kbd>Tab</kbd> | Swap between Video A and B (when in Overlay mode) |
| <kbd>-</kbd> / <kbd>=</kbd> | Decrease / Increase playback speed (`0.1x` to `2.0x`) |
| <kbd>U</kbd> | Toggle UI Overlay (removes text overlay for maximum FPS) |
| <kbd>R</kbd> | Reset Zoom & Pan |
| **Scroll Wheel** | Zoom in / out (synced for both videos) |
| **Left Click + Drag** | Pan around the zoomed view (synced for both videos) |
| <kbd>Esc</kbd> / <kbd>Q</kbd> | Quit player |

---

## Codebase File Map

* **[app.py](app.py)** – FastAPI web server.
* **[index.html](index.html)**, **[style.css](style.css)**, **[script.js](script.js)** – Web player frontend.
* **[native_player.py](native_player.py)** – RAM-buffered OpenCV dual video player.
* **[single_player.py](single_player.py)** – Lightweight single-video diagnostic player for verifying 240Hz monitor output.
* **[triple_player.py](triple_player.py)** – RAM-buffered OpenCV triple video player (Left: Distorted A, Center: Locked Reference, Right: Distorted C).
* **[gui_player.py](gui_player.py)** – Full-featured PySide6 desktop GUI comparison player porting the web layout.
* **[README.md](README.md)** – This guide.
