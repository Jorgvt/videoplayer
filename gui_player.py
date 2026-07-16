# /// script
# dependencies = [
#   "PySide6",
#   "opencv-python",
#   "numpy",
# ]
# ///

import os
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from PySide6.QtCore import Qt, QTimer, QEvent, QPoint
from PySide6.QtGui import QImage, QPixmap, QKeySequence, QShortcut, QCursor
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QSlider, QCheckBox, QFrame,
    QGridLayout, QSizePolicy, QStyle
)

# Paths
WORKSPACE_DIR = Path(__file__).parent.resolve()
DATASET_DIR = (WORKSPACE_DIR / ".." / ".." / "Datasets" / "GAIM240").resolve()
if not DATASET_DIR.exists():
    # Fallback to system absolute path
    DATASET_DIR = Path("/home/jv495/Datasets/GAIM240").resolve()
SCENES = ["marbles", "pink_room", "subway", "zeroday"]

# High-end Dark QSS Stylesheet
QSS_STYLE = """
QMainWindow {
    background-color: #0b0c10;
}
QWidget {
    color: #e0e0e0;
    font-family: 'Inter', -apple-system, sans-serif;
    font-size: 13px;
}
QFrame#sidebar {
    background-color: #14161d;
    border-right: 1px solid rgba(255, 255, 255, 0.08);
}
QLabel#logo {
    font-size: 20px;
    font-weight: bold;
    color: #9d4edd;
    margin-bottom: 15px;
}
QLabel.section-title {
    font-size: 11px;
    font-weight: bold;
    text-transform: uppercase;
    color: #9ca3af;
    margin-top: 15px;
    margin-bottom: 5px;
}
QPushButton.scene-btn {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 8px;
    font-weight: 500;
}
QPushButton.scene-btn:hover {
    background-color: rgba(255, 255, 255, 0.08);
}
QPushButton.scene-btn[active="true"] {
    background-color: #7b2cbf;
    border-color: #9d4edd;
    color: white;
}
QComboBox {
    background-color: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 6px 10px;
    color: #e0e0e0;
}
QComboBox:hover, QComboBox:focus {
    border-color: #7b2cbf;
    background-color: rgba(255, 255, 255, 0.06);
}
QComboBox::drop-down {
    border: none;
}
QPushButton.level-btn {
    background-color: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    padding: 6px;
}
QPushButton.level-btn:hover {
    background-color: rgba(255, 255, 255, 0.06);
}
QPushButton.level-btn[active="true"] {
    background-color: rgba(123, 44, 191, 0.2);
    border-color: #7b2cbf;
    color: #f3f4f6;
}
QFrame.info-panel {
    background-color: rgba(255, 255, 255, 0.02);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 10px;
    margin-top: 15px;
}
QFrame.hotkey-panel {
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    margin-top: 20px;
    padding-top: 10px;
}
QFrame#viewport-bg {
    background-color: #050608;
    border-radius: 10px;
}
QLabel#video-a-lbl, QLabel#video-b-lbl {
    background-color: black;
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 8px;
}
QFrame#controls-bar {
    background-color: rgba(20, 22, 29, 0.8);
    border-top: 1px solid rgba(255, 255, 255, 0.08);
    padding: 10px 20px;
}
QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: rgba(255, 255, 255, 0.08);
    border-radius: 3px;
}
QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #7b2cbf, stop:1 #9d4edd);
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: white;
    width: 12px;
    height: 12px;
    margin: -3px 0;
    border-radius: 6px;
}
QPushButton#play-btn {
    background-color: #f3f4f6;
    color: #0b0c10;
    border-radius: 18px;
    width: 36px;
    height: 36px;
    font-size: 16px;
}
QPushButton#play-btn:hover {
    background-color: white;
}
QPushButton.ctrl-btn {
    background: none;
    border: none;
    color: #9ca3af;
    font-size: 16px;
}
QPushButton.ctrl-btn:hover {
    color: white;
}
"""

class VideoDisplayLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.setMouseTracking(True)

class NativeGUIPlayer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GAIM240 Video Quality Comparer (Native GUI)")
        self.resize(1280, 800)
        self.setStyleSheet(QSS_STYLE)

        # State Variables
        self.scene_idx = 0
        self.metric_idx = 0
        self.level_idx = 1
        self.videos = {}
        
        self.frames_a = []
        self.frames_b = []
        self.meta_a = None
        self.meta_b = None
        
        self.frame_idx = 0
        self.playing = False
        self.is_looping = True
        self.layout_mode = "side-by-side" # "side-by-side", "single-a", "single-b", "overlay"
        self.show_b_in_overlay = False
        self.playback_speed = 1.0
        
        # Timing anchors
        self.play_start_time = 0.0
        self.play_start_frame_idx = 0
        
        # Zoom & Pan coordinate state
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.drag_start = None
        self.drag_start_pan = None
        self.sync_zoom = True
        
        # FPS status
        self.fps_history = []
        self.current_fps = 0.0
        
        # Scan and setup
        self.scan_dataset()
        self.setup_ui()
        self.setup_hotkeys()
        
        # Load initial video
        self.load_active_videos()
        
        # Playback timer loop
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_playback)
        self.timer.start(0) # Run frame loop as fast as possible

    def scan_dataset(self):
        for scene in SCENES:
            self.videos[scene] = []
            
        if not DATASET_DIR.exists():
            print(f"Dataset directory not found: {DATASET_DIR}")
            return
            
        for file in sorted(os.listdir(DATASET_DIR)):
            if not file.endswith(".mp4"):
                continue
                
            # Find scene
            scene = None
            for s in SCENES:
                if file.startswith(s + "_"):
                    scene = s
                    break
            if not scene:
                continue
                
            # Parse rest
            rest = file[len(scene) + 1:-4]
            if rest == "reference":
                self.videos[scene].append({
                    "filename": file,
                    "path": str(DATASET_DIR / file),
                    "metric": "reference",
                    "level": None,
                    "label": "Reference"
                })
            else:
                match = re.match(r"^(.*?)(?:_level([0-2]))?$", rest)
                if match:
                    metric = match.group(1)
                    lvl_num = match.group(2)
                    level = f"level{lvl_num}" if lvl_num is not None else None
                    metric_label = metric.replace("_", " ").replace("-", " ").title()
                    label = f"{metric_label} ({level.capitalize()})" if level else metric_label
                    
                    self.videos[scene].append({
                        "filename": file,
                        "path": str(DATASET_DIR / file),
                        "metric": metric,
                        "level": level,
                        "label": label
                    })

    def setup_ui(self):
        # Central Widget & Horizontal layout
        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # =========================================================================
        # SIDEBAR (LEFT)
        # =========================================================================
        sidebar = QFrame(self)
        sidebar.setObjectName("sidebar")
        sidebar.setMinimumWidth(300)
        sidebar.setMaximumWidth(320)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(15, 20, 15, 15)
        
        logo = QLabel("GAIM240 Play", sidebar)
        logo.setObjectName("logo")
        sidebar_layout.addWidget(logo)
        
        # Scene Grid
        lbl_scene = QLabel("Select Scene", sidebar)
        lbl_scene.setProperty("class", "section-title")
        sidebar_layout.addWidget(lbl_scene)
        
        self.scene_btns = {}
        scene_grid = QGridLayout()
        scene_grid.setSpacing(8)
        
        for idx, scene in enumerate(SCENES):
            btn = QPushButton(scene.replace("_", " ").title(), sidebar)
            btn.setProperty("class", "scene-btn")
            btn.setCheckable(True)
            if idx == 0:
                btn.setChecked(True)
                btn.setProperty("active", "true")
            btn.clicked.connect(lambda checked=False, s=scene: self.on_scene_clicked(s))
            self.scene_btns[scene] = btn
            scene_grid.addWidget(btn, idx // 2, idx % 2)
            
        sidebar_layout.addLayout(scene_grid)
        
        # Video A Selector
        lbl_select_a = QLabel("Video A (Base / Left)", sidebar)
        lbl_select_a.setProperty("class", "section-title")
        sidebar_layout.addWidget(lbl_select_a)
        
        self.combo_a = QComboBox(sidebar)
        self.combo_a.currentIndexChanged.connect(self.on_combo_a_changed)
        sidebar_layout.addWidget(self.combo_a)
        
        # Video B Selector
        lbl_select_b = QLabel("Video B (Compare / Right)", sidebar)
        lbl_select_b.setProperty("class", "section-title")
        sidebar_layout.addWidget(lbl_select_b)
        
        self.combo_b = QComboBox(sidebar)
        self.combo_b.currentIndexChanged.connect(self.on_combo_b_changed)
        sidebar_layout.addWidget(self.combo_b)
        
        # Compare Level Presets
        lbl_lvl = QLabel("Compare Level", sidebar)
        lbl_lvl.setProperty("class", "section-title")
        sidebar_layout.addWidget(lbl_lvl)
        
        level_layout = QHBoxLayout()
        self.level_btns = []
        for i in range(3):
            btn = QPushButton(f"Lvl {i}", sidebar)
            btn.setProperty("class", "level-btn")
            btn.setCheckable(True)
            if i == 1:
                btn.setChecked(True)
                btn.setProperty("active", "true")
            btn.clicked.connect(lambda checked=False, lvl=i: self.on_level_preset_clicked(lvl))
            self.level_btns.append(btn)
            level_layout.addWidget(btn)
        sidebar_layout.addLayout(level_layout)
        
        # Metadata Panel
        lbl_meta = QLabel("Metadata", sidebar)
        lbl_meta.setProperty("class", "section-title")
        sidebar_layout.addWidget(lbl_meta)
        
        info_panel = QFrame(sidebar)
        info_panel.setProperty("class", "info-panel")
        info_layout = QGridLayout(info_panel)
        info_layout.setContentsMargins(8, 8, 8, 8)
        info_layout.setVerticalSpacing(6)
        
        self.meta_file_a_lbl = QLabel("-", info_panel)
        self.meta_file_b_lbl = QLabel("-", info_panel)
        self.meta_res_lbl = QLabel("-", info_panel)
        self.meta_fps_lbl = QLabel("-", info_panel)
        
        info_layout.addWidget(QLabel("A File:", info_panel), 0, 0)
        info_layout.addWidget(self.meta_file_a_lbl, 0, 1, Qt.AlignRight)
        info_layout.addWidget(QLabel("B File:", info_panel), 1, 0)
        info_layout.addWidget(self.meta_file_b_lbl, 1, 1, Qt.AlignRight)
        info_layout.addWidget(QLabel("Resolution:", info_panel), 2, 0)
        info_layout.addWidget(self.meta_res_lbl, 2, 1, Qt.AlignRight)
        info_layout.addWidget(QLabel("FPS Display:", info_panel), 3, 0)
        info_layout.addWidget(self.meta_fps_lbl, 3, 1, Qt.AlignRight)
        
        sidebar_layout.addWidget(info_panel)
        
        # Hotkeys Help Panel
        hotkeys_panel = QFrame(sidebar)
        hotkeys_panel.setProperty("class", "hotkey-panel")
        hotkeys_layout = QVBoxLayout(hotkeys_panel)
        hotkeys_layout.setContentsMargins(0, 10, 0, 0)
        hotkeys_layout.setSpacing(4)
        
        lbl_hk = QLabel("HOTKEYS", hotkeys_panel)
        lbl_hk.setStyleSheet("font-size: 10px; font-weight: bold; color: #6b7280;")
        hotkeys_layout.addWidget(lbl_hk)
        
        hotkeys_layout.addWidget(QLabel("Space : Play / Pause", hotkeys_panel))
        hotkeys_layout.addWidget(QLabel("Left / Right Arrow : Step Frame", hotkeys_panel))
        hotkeys_layout.addWidget(QLabel("L : Cycle Layouts", hotkeys_panel))
        hotkeys_layout.addWidget(QLabel("Tab : Swap A/B (Overlay)", hotkeys_panel))
        hotkeys_layout.addWidget(QLabel("Z : Reset Zoom / Pan", hotkeys_panel))
        
        sidebar_layout.addWidget(hotkeys_panel)
        sidebar_layout.addStretch()
        main_layout.addWidget(sidebar)
        
        # =========================================================================
        # MAIN VIEW AREA (RIGHT)
        # =========================================================================
        main_view = QFrame(self)
        main_view_layout = QVBoxLayout(main_view)
        main_view_layout.setContentsMargins(0, 0, 0, 0)
        main_view_layout.setSpacing(0)
        
        # Top Header Bar
        header = QFrame(main_view)
        header.setMinimumHeight(55)
        header.setMaximumHeight(55)
        header.setStyleSheet("background-color: #14161d; border-bottom: 1px solid rgba(255, 255, 255, 0.08);")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 0, 20, 0)
        
        # Layout Mode dropdown
        lbl_lay = QLabel("Layout:", header)
        lbl_lay.setStyleSheet("font-weight: bold; color: #9ca3af;")
        header_layout.addWidget(lbl_lay)
        
        self.combo_layout = QComboBox(header)
        self.combo_layout.addItems(["Side-by-Side", "Single A", "Single B", "Overlay Swap"])
        self.combo_layout.currentIndexChanged.connect(self.on_layout_combo_changed)
        header_layout.addWidget(self.combo_layout)
        
        header_layout.addStretch()
        
        self.zoom_chk = QCheckBox("Sync Zoom & Pan", header)
        self.zoom_chk.setChecked(True)
        self.zoom_chk.stateChanged.connect(self.on_zoom_sync_changed)
        header_layout.addWidget(self.zoom_chk)
        
        btn_reset_zoom = QPushButton("Reset View", header)
        btn_reset_zoom.clicked.connect(self.reset_zoom)
        header_layout.addWidget(btn_reset_zoom)
        
        main_view_layout.addWidget(header)
        
        # Viewport Workspace
        viewport = QFrame(main_view)
        viewport.setObjectName("viewport-bg")
        viewport_layout = QHBoxLayout(viewport)
        viewport_layout.setContentsMargins(15, 15, 15, 15)
        viewport_layout.setSpacing(10)
        
        # Displays
        self.lbl_disp_a = VideoDisplayLabel(viewport)
        self.lbl_disp_a.setObjectName("video-a-lbl")
        self.lbl_disp_b = VideoDisplayLabel(viewport)
        self.lbl_disp_b.setObjectName("video-b-lbl")
        
        viewport_layout.addWidget(self.lbl_disp_a)
        viewport_layout.addWidget(self.lbl_disp_b)
        
        # Install event filters on display labels to capture mouse events (for Zoom & Pan)
        self.lbl_disp_a.installEventFilter(self)
        self.lbl_disp_b.installEventFilter(self)
        
        main_view_layout.addWidget(viewport)
        
        # Bottom Controls Footer
        controls = QFrame(main_view)
        controls.setObjectName("controls-bar")
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(20, 10, 20, 10)
        controls_layout.setSpacing(6)
        
        # Seek slider
        slider_layout = QHBoxLayout()
        self.lbl_time_curr = QLabel("00:00.000", controls)
        self.lbl_time_curr.setStyleSheet("font-family: monospace;")
        self.slider = QSlider(Qt.Horizontal, controls)
        self.slider.setRange(0, 100)
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.on_slider_value_changed)
        self.lbl_time_total = QLabel("00:00.000", controls)
        self.lbl_time_total.setStyleSheet("font-family: monospace;")
        
        slider_layout.addWidget(self.lbl_time_curr)
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.lbl_time_total)
        controls_layout.addLayout(slider_layout)
        
        # Button Actions Row
        actions_layout = QHBoxLayout()
        
        btn_prev = QPushButton(controls)
        btn_prev.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        btn_prev.setProperty("class", "ctrl-btn")
        btn_prev.clicked.connect(lambda: self.step_frames(-1))
        actions_layout.addWidget(btn_prev)
        
        self.btn_play = QPushButton(controls)
        self.btn_play.setObjectName("play-btn")
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.btn_play.clicked.connect(self.toggle_play)
        actions_layout.addWidget(self.btn_play)
        
        btn_next = QPushButton(controls)
        btn_next.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipForward))
        btn_next.setProperty("class", "ctrl-btn")
        btn_next.clicked.connect(lambda: self.step_frames(1))
        actions_layout.addWidget(btn_next)
        
        self.btn_loop = QPushButton("Loop", controls)
        self.btn_loop.setCheckable(True)
        self.btn_loop.setChecked(True)
        self.btn_loop.setStyleSheet("QPushButton { font-weight: bold; color: #9ca3af; } QPushButton:checked { color: #9d4edd; }")
        self.btn_loop.clicked.connect(self.on_loop_toggled)
        actions_layout.addWidget(self.btn_loop)
        
        actions_layout.addStretch()
        
        # Speed selector
        actions_layout.addWidget(QLabel("Speed:", controls))
        self.combo_speed = QComboBox(controls)
        self.combo_speed.addItems(["0.1x", "0.25x", "0.5x", "1.0x", "2.0x"])
        self.combo_speed.setCurrentIndex(3) # 1.0x
        self.combo_speed.currentIndexChanged.connect(self.on_speed_changed)
        actions_layout.addWidget(self.combo_speed)
        
        controls_layout.addLayout(actions_layout)
        main_view_layout.addWidget(controls)
        
        main_layout.addWidget(main_view)

    # Event Filters for zoom & pan
    def eventFilter(self, obj, event):
        is_a = obj == self.lbl_disp_a
        is_b = obj == self.lbl_disp_b
        
        if not (is_a or is_b):
            return super().eventFilter(obj, event)
            
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.drag_start = event.globalPosition().toPoint()
                self.drag_start_pan = (self.pan_x, self.pan_y)
                self.setCursor(QCursor(Qt.ClosedHandCursor))
                return True
                
        elif event.type() == QEvent.MouseMove:
            if self.drag_start is not None:
                delta = event.globalPosition().toPoint() - self.drag_start
                self.pan_x = self.drag_start_pan[0] - int(delta.x() / self.zoom_scale)
                self.pan_y = self.drag_start_pan[1] - int(delta.y() / self.zoom_scale)
                self.update_displays()
                return True
                
        elif event.type() == QEvent.MouseButtonRelease:
            if event.button() == Qt.LeftButton:
                self.drag_start = None
                self.drag_start_pan = None
                self.setCursor(QCursor(Qt.ArrowCursor))
                return True
                
        elif event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            old_scale = self.zoom_scale
            if delta > 0:
                self.zoom_scale = min(20.0, self.zoom_scale * 1.15)
            else:
                self.zoom_scale = max(1.0, self.zoom_scale / 1.15)
                if self.zoom_scale == 1.0:
                    self.pan_x = 0
                    self.pan_y = 0
            self.update_displays()
            return True
            
        return super().eventFilter(obj, event)

    # Setup standard QShortcuts
    def setup_hotkeys(self):
        # Space Toggle Play
        QShortcut(QKeySequence(Qt.Key_Space), self, self.toggle_play)
        
        # Right Arrow Frame Step Next
        QShortcut(QKeySequence(Qt.Key_Right), self, lambda: self.step_frames(1))
        
        # Left Arrow Frame Step Prev
        QShortcut(QKeySequence(Qt.Key_Left), self, lambda: self.step_frames(-1))
        
        # Reset Zoom View
        QShortcut(QKeySequence(Qt.Key_Z), self, self.reset_zoom)
        
        # Cycle layout
        QShortcut(QKeySequence(Qt.Key_L), self, self.cycle_layouts)
        
        # Swap Overlay
        tab_sc = QShortcut(QKeySequence(Qt.Key_Tab), self)
        tab_sc.setAutoRepeat(False)
        tab_sc.activated.connect(self.on_tab_down)
        
        # Release swap
        tab_sc_up = QShortcut(QKeySequence("Ctrl+Tab"), self) # Tab keyup fallback in Qt
        
    def on_tab_down(self):
        if self.layout_mode == "overlay":
            self.show_b_in_overlay = not self.show_b_in_overlay
            self.update_displays()

    # Video Setup
    def load_scene(self, scene_key):
        self.scene_idx = SCENES.index(scene_key)
        
        # Update sidebar scene buttons styling
        for key, btn in self.scene_btns.items():
            btn.setChecked(key == scene_key)
            btn.setProperty("active", "true" if key == scene_key else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            
        videos = self.videos[scene_key]
        
        # Block signals during reload
        self.combo_a.blockSignals(True)
        self.combo_b.blockSignals(True)
        
        self.combo_a.clear()
        self.combo_b.clear()
        
        for vid in videos:
            self.combo_a.addItem(vid["label"], vid["path"])
            self.combo_b.addItem(vid["label"], vid["path"])
            
        # Select defaults
        ref_video = next((v for v in videos if v["metric"] == "reference"), None)
        if ref_video:
            ref_idx = self.combo_a.findData(ref_video["path"])
            self.combo_a.setCurrentIndex(ref_idx)
            
        compare_video = next((v for v in videos if v["metric"] != "reference" and (v["level"] == "level1" or v["level"] is None)), None)
        if compare_video:
            comp_idx = self.combo_b.findData(compare_video["path"])
            self.combo_b.setCurrentIndex(comp_idx)
        else:
            self.combo_b.setCurrentIndex(min(1, self.combo_b.count() - 1))
            
        self.combo_a.blockSignals(False)
        self.combo_b.blockSignals(False)
        
        self.on_level_preset_clicked(self.level_idx)
        self.load_active_videos()

    def on_scene_clicked(self, scene_key):
        self.load_scene(scene_key)

    def on_combo_a_changed(self):
        self.load_active_videos()

    def on_combo_b_changed(self):
        # Update level presets check states
        path = self.combo_b.currentData()
        videos = self.videos[SCENES[self.scene_idx]]
        info = next((v for v in videos if v["path"] == path), None)
        
        if info and info["level"] is not None:
            lvl_num = int(info["level"].replace("level", ""))
            self.level_idx = lvl_num
            for i, btn in enumerate(self.level_btns):
                btn.setChecked(i == lvl_num)
                btn.setProperty("active", "true" if i == lvl_num else "false")
                btn.style().unpolish(btn)
                btn.style().polish(btn)
                
        self.load_active_videos()

    def on_level_preset_clicked(self, level):
        self.level_idx = level
        for i, btn in enumerate(self.level_btns):
            btn.setChecked(i == level)
            btn.setProperty("active", "true" if i == level else "false")
            btn.style().unpolish(btn)
            btn.style().polish(btn)
            
        # Try to find B with same metric but new level
        path_b = self.combo_b.currentData()
        videos = self.videos[SCENES[self.scene_idx]]
        info = next((v for v in videos if v["path"] == path_b), None)
        if info and info["metric"] != "reference":
            target = next((v for v in videos if v["metric"] == info["metric"] and v["level"] == f"level{level}"), None)
            if target:
                self.combo_b.blockSignals(True)
                idx = self.combo_b.findData(target["path"])
                self.combo_b.setCurrentIndex(idx)
                self.combo_b.blockSignals(False)
                
        self.load_active_videos()

    def load_active_videos(self):
        path_a = self.combo_a.currentData()
        path_b = self.combo_b.currentData()
        
        if not path_a or not path_b:
            return
            
        self.setCursor(Qt.WaitCursor)
        self.statusBar().showMessage("Buffering frames to RAM...")
        QApplication.processEvents()
        
        self.frames_a = self.preload_video(path_a)
        self.frames_b = self.preload_video(path_b)
        
        # Metadata
        self.meta_file_a_lbl.setText(Path(path_a).name)
        self.meta_file_b_lbl.setText(Path(path_b).name)
        
        if self.frames_a:
            h, w = self.frames_a[0].shape[:2]
            self.meta_res_lbl.setText(f"{w}x{h}")
            
        # Reset frames
        self.frame_idx = 0
        self.update_playback_anchor()
        
        # Reset slider
        num_frames = max(len(self.frames_a), len(self.frames_b))
        self.slider.blockSignals(True)
        self.slider.setRange(0, num_frames - 1)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        
        self.lbl_time_total.setText(self.format_time((num_frames - 1) / 240.0))
        self.lbl_time_curr.setText("00:00.000")
        
        self.update_displays()
        self.setCursor(Qt.ArrowCursor)
        self.statusBar().showMessage("Ready", 2000)

    def preload_video(self, path):
        cap = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    # Display Updates & Zoom Pan logic
    def apply_zoom_pan(self, img):
        if self.zoom_scale == 1.0:
            return img
            
        h, w = img.shape[:2]
        crop_w = int(w / self.zoom_scale)
        crop_h = int(h / self.zoom_scale)
        
        center_x = w // 2 + self.pan_x
        center_y = h // 2 + self.pan_y
        
        x1 = max(0, center_x - crop_w // 2)
        y1 = max(0, center_y - crop_h // 2)
        x2 = min(w, x1 + crop_w)
        y2 = min(h, y1 + crop_h)
        
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - crop_w)
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - crop_h)
            
        cropped = img[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)
        return resized

    def convert_to_qpixmap(self, frame):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
        # Scaled pixmap to QLabel size maintaining aspect ratio
        return QPixmap.fromImage(qimg)

    def update_displays(self):
        if not self.frames_a or not self.frames_b:
            return
            
        idx_a = self.frame_idx % len(self.frames_a)
        idx_b = self.frame_idx % len(self.frames_b)
        
        frame_a = self.frames_a[idx_a]
        frame_b = self.frames_b[idx_b]
        
        # Draw frame
        disp_a = self.apply_zoom_pan(frame_a)
        disp_b = self.apply_zoom_pan(frame_b)
        
        pix_a = self.convert_to_qpixmap(disp_a)
        pix_b = self.convert_to_qpixmap(disp_b)
        
        # Apply layout configurations
        if self.layout_mode == "side-by-side":
            self.lbl_disp_a.setVisible(True)
            self.lbl_disp_b.setVisible(True)
            self.lbl_disp_a.setPixmap(pix_a.scaled(self.lbl_disp_a.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.lbl_disp_b.setPixmap(pix_b.scaled(self.lbl_disp_b.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self.layout_mode == "single-a":
            self.lbl_disp_a.setVisible(True)
            self.lbl_disp_b.setVisible(False)
            self.lbl_disp_a.setPixmap(pix_a.scaled(self.lbl_disp_a.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self.layout_mode == "single-b":
            self.lbl_disp_a.setVisible(False)
            self.lbl_disp_b.setVisible(True)
            self.lbl_disp_b.setPixmap(pix_b.scaled(self.lbl_disp_b.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        elif self.layout_mode == "overlay":
            self.lbl_disp_a.setVisible(True)
            self.lbl_disp_b.setVisible(False)
            active_pix = pix_b if self.show_b_in_overlay else pix_a
            self.lbl_disp_a.setPixmap(active_pix.scaled(self.lbl_disp_a.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # Update display pixmaps on resize to fill labels
        self.update_displays()

    # Playback loop timing
    def update_playback(self):
        if not self.frames_a:
            return
            
        # Update FPS rolling counter
        now = time.perf_counter()
        self.fps_history.append(now)
        self.fps_history = [t for t in self.fps_history if now - t <= 1.0]
        if len(self.fps_history) > 1:
            self.current_fps = len(self.fps_history) / (self.fps_history[-1] - self.fps_history[0])
            self.meta_fps_lbl.setText(f"{self.current_fps:.1f} FPS")
            
        target_fps = 240.0 * self.playback_speed
        
        if self.playing:
            elapsed = now - self.play_start_time
            total_frames = max(len(self.frames_a), len(self.frames_b))
            
            self.frame_idx = self.play_start_frame_idx + int(elapsed * target_fps)
            if self.frame_idx >= total_frames:
                if self.is_looping:
                    self.play_start_time = now
                    self.play_start_frame_idx = 0
                    self.frame_idx = 0
                else:
                    self.playing = False
                    self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
                    self.frame_idx = total_frames - 1
            
            # Sync slider and times
            self.slider.blockSignals(True)
            self.slider.setValue(self.frame_idx)
            self.slider.blockSignals(False)
            self.lbl_time_curr.setText(self.format_time(self.frame_idx / 240.0))
            
            self.update_displays()
            
            # Hybrid sleep pacing to target FPS
            time_elapsed = time.perf_counter() - now
            frame_time = 1.0 / target_fps
            time_remaining = frame_time - time_elapsed
            if time_remaining > 0.001:
                # Let Qt thread rest a bit (yields loop execution)
                time.sleep(time_remaining - 0.0005)

    def toggle_play(self):
        self.playing = not self.playing
        if self.playing:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
            self.update_playback_anchor()
        else:
            self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def step_frames(self, direction):
        self.playing = False
        self.btn_play.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        num_frames = max(len(self.frames_a), len(self.frames_b))
        self.frame_idx = (self.frame_idx + direction) % num_frames
        
        self.slider.blockSignals(True)
        self.slider.setValue(self.frame_idx)
        self.slider.blockSignals(False)
        self.lbl_time_curr.setText(self.format_time(self.frame_idx / 240.0))
        
        self.update_playback_anchor()
        self.update_displays()

    def update_playback_anchor(self):
        self.play_start_time = time.perf_counter()
        self.play_start_frame_idx = self.frame_idx

    # Controls handlers
    def on_speed_changed(self, idx):
        speeds_val = [0.1, 0.25, 0.5, 1.0, 2.0]
        self.playback_speed = speeds_val[idx]
        self.update_playback_anchor()

    def on_loop_toggled(self, checked):
        self.is_looping = checked

    def on_slider_pressed(self):
        self.slider_dragging = True

    def on_slider_released(self):
        self.slider_dragging = False
        self.update_playback_anchor()

    def on_slider_value_changed(self, val):
        self.frame_idx = val
        self.lbl_time_curr.setText(self.format_time(self.frame_idx / 240.0))
        self.update_displays()

    def on_layout_combo_changed(self, idx):
        modes = ["side-by-side", "single-a", "single-b", "overlay"]
        self.layout_mode = modes[idx]
        self.update_displays()

    def cycle_layouts(self):
        idx = (self.combo_layout.currentIndex() + 1) % 4
        self.combo_layout.setCurrentIndex(idx)

    def on_zoom_sync_changed(self, state):
        self.sync_zoom = state == 2 # Qt.Checked

    def reset_zoom(self):
        self.zoom_scale = 1.0
        self.pan_x = 0
        self.pan_y = 0
        self.update_displays()

    def format_time(self, secs):
        m = int(secs // 60)
        s = int(secs % 60)
        ms = int((secs % 1) * 1000)
        return f"{m:02d}:{s:02d}.{ms:03d}"

if __name__ == "__main__":
    app = QApplication(sys.argv)
    # Default layout theme colors
    app.setStyle("Fusion")
    
    player = NativeGUIPlayer()
    player.load_scene("marbles")
    player.show()
    sys.exit(app.exec())
