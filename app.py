# /// script
# dependencies = [
#   "fastapi",
#   "uvicorn",
# ]
# ///

import os
import re
import urllib.parse
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Video Quality Dataset Player")

# Constants
WORKSPACE_DIR = Path(__file__).parent.resolve()
# The dataset path is relative to the workspace, i.e., ../../Datasets/GAIM240/
DATASET_DIR = (WORKSPACE_DIR / ".." / ".." / "Datasets" / "GAIM240").resolve()

# Fallback: if the relative path doesn't exist, search in common locations
if not DATASET_DIR.exists():
    # Try absolute path on the user's system
    DATASET_DIR = Path("/home/jv495/Developer/Datasets/GAIM240").resolve()

print(f"Loading videos from: {DATASET_DIR}")

# Regex to parse the video files
# e.g., marbles_reference.mp4 -> scene: marbles, metric: reference, level: None
# e.g., marbles_judder_level1.mp4 -> scene: marbles, metric: judder, level: level1
# e.g., pink_room_temporal-resolution-multiplexing_level0.mp4 -> scene: pink_room, metric: temporal-resolution-multiplexing, level: level0
# e.g., pink_room_dlss_rr_level2.mp4 -> scene: pink_room, metric: dlss_rr, level: level2

SCENE_PREFIXES = ["marbles", "pink_room", "subway", "zeroday"]

def parse_video_filename(filename: str):
    if not filename.endswith(".mp4"):
        return None
    
    # Match the scene prefix
    scene = None
    for prefix in SCENE_PREFIXES:
        if filename.startswith(prefix + "_"):
            scene = prefix
            break
    
    if not scene:
        return None
    
    # Strip the scene prefix and extension
    rest = filename[len(scene) + 1:-4]
    
    if rest == "reference":
        return {
            "filename": filename,
            "scene": scene,
            "metric": "reference",
            "level": None,
            "label": "Reference"
        }
    
    # Try matching metric and level
    # e.g., judder_level1, dlss_rr_level0, temporal-resolution-multiplexing_level1
    match = re.match(r"^(.*?)(?:_level([0-2]))?$", rest)
    if match:
        metric = match.group(1)
        level_num = match.group(2)
        level = f"level{level_num}" if level_num is not None else None
        
        # Make a pretty label
        metric_label = metric.replace("_", " ").replace("-", " ").title()
        if level:
            label = f"{metric_label} ({level.capitalize()})"
        else:
            label = metric_label
            
        return {
            "filename": filename,
            "scene": scene,
            "metric": metric,
            "level": level,
            "label": label
        }
        
    return {
        "filename": filename,
        "scene": scene,
        "metric": rest,
        "level": None,
        "label": rest.replace("_", " ").replace("-", " ").title()
    }

@app.get("/api/videos")
def get_videos():
    if not DATASET_DIR.exists():
        raise HTTPException(status_code=404, detail=f"Dataset directory not found: {DATASET_DIR}")
    
    scenes = {prefix: [] for prefix in SCENE_PREFIXES}
    
    # Scan files
    for entry in sorted(os.listdir(DATASET_DIR)):
        info = parse_video_filename(entry)
        if info:
            info["path"] = f"/videos/{urllib.parse.quote(entry)}"
            scenes[info["scene"]].append(info)
            
    # Sort reference first, then by metric and level
    for scene in scenes:
        scenes[scene].sort(key=lambda x: (
            0 if x["metric"] == "reference" else 1,
            x["metric"],
            x["level"] or ""
        ))
        
    return {
        "dataset_path": str(DATASET_DIR),
        "scenes": scenes
    }

# Serve the static files of the player
app.mount("/videos", StaticFiles(directory=DATASET_DIR), name="videos")

# Serve index.html, style.css, script.js from workspace
@app.get("/")
def read_root():
    index_path = WORKSPACE_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>Video Player Frontend not found! Put index.html in workspace folder.</h1>")

@app.get("/style.css")
def get_style():
    style_path = WORKSPACE_DIR / "style.css"
    if style_path.exists():
        return FileResponse(style_path, media_type="text/css")
    return HTMLResponse("", status_code=404)

@app.get("/script.js")
def get_script():
    script_path = WORKSPACE_DIR / "script.js"
    if script_path.exists():
        return FileResponse(script_path, media_type="application/javascript")
    return HTMLResponse("", status_code=404)

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading
    import time
    
    port = 8000
    
    def open_browser():
        time.sleep(1.5)
        url = f"http://localhost:{port}"
        print(f"Opening browser at: {url}")
        webbrowser.open(url)
        
    # Start browser in a background thread
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=True)
