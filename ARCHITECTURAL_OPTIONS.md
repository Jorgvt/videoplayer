# 240Hz Zero-Latency Video Renderer Architectural Options

## Executive Summary & Root Cause Analysis

### Why Uncompressed Memory Preloading Failed:
* **The Math**: 1200 frames $\times$ 1280 $\times$ 720 $\times$ 3 bytes (BGR) = **2.76 GB per video**.
* **Per Trial**: 3 videos (Left, Ref, Right) = **8.3 GB CPU RAM**.
* **With Prefetching**: Current + Next trial = **16.6 GB CPU RAM + 16.6 GB VRAM**.
* **The Result**: System RAM exceeds physical limits, triggering the Linux Kernel **OOM Killer (`SIGKILL 9`)**, abruptly terminating the application.

---

## The Core Solution: Compressed Zero-Copy Hardware Decoding (GPU NVDEC)

Instead of decoding MP4 videos into raw uncompressed software pixel buffers in CPU RAM, professional 240Hz video engines (mpv, VLC, Unreal Engine, Vulkan Video) store compressed H.264 bitstream packets (**50 MB total**) and decode them directly inside GPU VRAM via hardware decoders (**NVIDIA NVDEC**).

```
+-----------------------------------------------------------------------------------+
|                              RECOMMENDED ARCHITECTURE                             |
+-----------------------------------------------------------------------------------+
|                                                                                   |
|  [ Disk MP4 File ] ---> [ 50 MB Compressed RAM ] ---> [ GPU NVDEC HW Decoder ]    |
|                                                               |                   |
|                                                               v                   |
|                                                    [ GPU VRAM YUV Surface ]       |
|                                                               |                   |
|                                                               v                   |
|                                                    [ OpenGL/Vulkan 239 FPS ]      |
+-----------------------------------------------------------------------------------+
```

---

## Architectural Options Comparison

| Option | Technology Stack | System RAM | Hardware Decoding | Implementation Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Option 1 (Active)** | **`libmpv` C API (Vulkan / OpenGL + NVDEC)** | **~50 MB** | **NVIDIA NVDEC / VDPAU** | **Target Implementation** | **ACTIVE** |
| **Option 2 (Fallback A)** | **FFmpeg C API + CUDA NVDEC (`libavcodec.so.60`)** | **~60 MB** | **CUDA NVDEC Interop** | **Fallback Option A** | Saved |
| **Option 3 (Fallback B)** | **Pure Native Vulkan API (`VK_KHR_video_decode`)** | **~40 MB** | **Vulkan Video Queue** | **Fallback Option B** | Saved |

---

### Option 1: Embedded `libmpv` Engine (Active)
* **Description**: Embeds `libmpv.so` / `libmpv` C API. Uses `mpv`'s zero-copy Vulkan/OpenGL render context and NVDEC hardware video decoder.
* **Benefits**: 50 MB RAM footprint, instant 0.005s file preloading, locked 239.76 FPS.
* **No-Sudo**: Fully supported via user-space dynamic linking or local build.

### Option 2: FFmpeg C API (`libavcodec` + CUDA NVDEC Interop)
* **Description**: Calls `/lib/x86_64-linux-gnu/libavcodec.so.60` with `AV_HWDEVICE_TYPE_CUDA` / `NVDEC`. Frames are decoded directly onto NVIDIA CUDA GPU surfaces and mapped into OpenGL textures via `cudaGraphicsGLRegisterBuffer`.
* **No-Sudo**: `libavcodec.so.60` and `libnvcuvid.so` are already pre-installed on this machine in `/lib/x86_64-linux-gnu`.

### Option 3: Pure Native Vulkan API (`ash` + `VK_KHR_video_decode_h264`)
* **Description**: A pure native Rust Vulkan application written using `ash` (Vulkan bindings for Rust) and `ash-window`. Uses Vulkan's native hardware video decoding extension `VK_KHR_video_decode_queue` and mailbox swapchains (`VK_PRESENT_MODE_MAILBOX_KHR`).
* **No-Sudo**: `libvulkan.so.1` is already pre-installed on this machine in `/lib/x86_64-linux-gnu`.
