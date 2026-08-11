"""
platform_utils.py — Cross-platform helpers for the GAIM240 video player project.

Usage in any script:
    from platform_utils import get_dataset_dir, set_native_lib_env

On Linux these behave identically to the previous hard-coded behaviour.
On Windows they switch to DLL / PATH conventions automatically.
"""

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------
IS_WINDOWS = sys.platform == "win32"
IS_LINUX   = sys.platform.startswith("linux")

# ---------------------------------------------------------------------------
# Dataset directory resolution
# ---------------------------------------------------------------------------
# Priority order:
#   1. GAIM240_DATASET_DIR environment variable (any platform, any path)
#   2. Sibling directory relative to the workspace root  (../../Datasets/GAIM240)
#   3. Linux fallback: /home/jv495/Datasets/GAIM240
#   4. Windows fallback: C:\Datasets\GAIM240
#
# Set GAIM240_DATASET_DIR in your shell (or a .env file) to override
# everything on any machine:
#   Linux:   export GAIM240_DATASET_DIR=/mnt/data/GAIM240
#   Windows: set GAIM240_DATASET_DIR=D:\Datasets\GAIM240
# ---------------------------------------------------------------------------
_WORKSPACE_ROOT = Path(__file__).parent.resolve()

def get_dataset_dir() -> Path:
    """Return the resolved dataset directory, with clear fallback chain."""
    # 1. Explicit environment variable — highest priority
    env_val = os.environ.get("GAIM240_DATASET_DIR")
    if env_val:
        p = Path(env_val).resolve()
        if p.exists():
            return p
        print(f"[platform_utils] WARNING: GAIM240_DATASET_DIR={env_val!r} does not exist. Falling back.")

    # 2. Relative sibling path (works on any machine with the standard layout)
    candidate = (_WORKSPACE_ROOT / ".." / ".." / "Datasets" / "GAIM240").resolve()
    if candidate.exists():
        return candidate

    # 3. Platform-specific absolute fallbacks
    if IS_WINDOWS:
        fallbacks = [
            Path(r"C:\Datasets\GAIM240"),
            Path(r"D:\Datasets\GAIM240"),
            Path(r"D:\GAIM240"),
        ]
    else:
        fallbacks = [
            Path("/home/jv495/Datasets/GAIM240"),
            Path("/home/jv495/Developer/Datasets/GAIM240"),
        ]

    for fb in fallbacks:
        if fb.exists():
            return fb.resolve()

    # Return the relative candidate anyway so callers get a meaningful error
    return candidate


# ---------------------------------------------------------------------------
# Native library / DLL environment setup
# ---------------------------------------------------------------------------
_RUST_PLAYER_DIR = _WORKSPACE_ROOT / "rust_player"
_LIB_ROOT        = _RUST_PLAYER_DIR / "lib"

def set_native_lib_env(env: dict) -> dict:
    """
    Mutate *env* (a copy of os.environ) so that the OS can find the bundled
    shared libraries / DLLs that ship with the Rust player.

    Linux:  prepend to LD_LIBRARY_PATH  (lib/usr/lib/x86_64-linux-gnu)
    Windows: prepend to PATH            (lib/windows/bin  — DLL search path)

    Returns the modified env dict for convenience.
    """
    if IS_WINDOWS:
        # Windows DLLs are resolved via PATH.
        # Place bundled DLLs in rust_player/lib/windows/bin/ on the target machine.
        win_dll_dir = str((_LIB_ROOT / "windows" / "bin").resolve())
        env["PATH"] = win_dll_dir + os.pathsep + env.get("PATH", "")
    else:
        # Linux .so resolution
        linux_lib_dir = str((_LIB_ROOT / "usr" / "lib" / "x86_64-linux-gnu").resolve())
        env["LD_LIBRARY_PATH"] = linux_lib_dir + ":" + env.get("LD_LIBRARY_PATH", "")

    return env


def set_native_bin_env(env: dict) -> dict:
    """
    Prepend the bundled binary directory (ffmpeg, etc.) to PATH.

    Linux:   rust_player/lib/usr/bin  +  bin/
    Windows: rust_player/lib/windows/bin  (same directory as DLLs)
    """
    wrapper_bin_dir = str((_WORKSPACE_ROOT / "bin").resolve())

    if IS_WINDOWS:
        win_bin_dir = str((_LIB_ROOT / "windows" / "bin").resolve())
        env["PATH"] = win_bin_dir + os.pathsep + wrapper_bin_dir + os.pathsep + env.get("PATH", "")
    else:
        linux_bin_dir = str((_LIB_ROOT / "usr" / "bin").resolve())
        env["PATH"] = linux_bin_dir + os.pathsep + wrapper_bin_dir + os.pathsep + env.get("PATH", "")

    return env


# ---------------------------------------------------------------------------
# ffmpeg binary resolution
# ---------------------------------------------------------------------------
def get_ffmpeg_bin() -> str:
    """
    Return the path to the ffmpeg executable to use.
    Checks bundled copies first, then falls back to the system PATH.
    """
    exe = "ffmpeg.exe" if IS_WINDOWS else "ffmpeg"

    candidates = [
        _RUST_PLAYER_DIR / "lib" / ("windows" if IS_WINDOWS else "usr") / ("bin" if IS_WINDOWS else "bin") / exe,
        _WORKSPACE_ROOT / "bin" / exe,
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Fallback: rely on PATH
    return "ffmpeg"
