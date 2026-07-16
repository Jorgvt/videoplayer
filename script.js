// Global State
let videoData = null;
let currentScene = 'marbles';
let currentLayout = 'side-by-side';
let isPlaying = false;
let isLooping = true;
let playbackSpeed = 1.0;
let syncZoom = true;

// Zoom & Pan coordinates
let zoomA = { scale: 1, x: 0, y: 0 };
let zoomB = { scale: 1, x: 0, y: 0 };
let isDragging = false;
let dragStart = { x: 0, y: 0 };
let dragStartCoords = { x: 0, y: 0 };
let activeDragWrapper = null;

// Split Slider State
let isDraggingSlider = false;
let sliderPosPercent = 50; // 0 to 100

// A/B Swap State
let showSwapB = false;

// DOM Elements
const videoSelectA = document.getElementById('video-select-a');
const videoSelectB = document.getElementById('video-select-b');
const videoA = document.getElementById('video-a');
const videoB = document.getElementById('video-b');
const wrapperA = document.getElementById('wrapper-a');
const wrapperB = document.getElementById('wrapper-b');
const playerContainer = document.getElementById('player-container');
const playPauseBtn = document.getElementById('play-pause-btn');
const loopBtn = document.getElementById('loop-btn');
const stepPrevBtn = document.getElementById('step-prev-btn');
const stepNextBtn = document.getElementById('step-next-btn');
const speedSelect = document.getElementById('speed-select');
const timelineWrapper = document.getElementById('timeline-wrapper');
const timelineProgress = document.getElementById('timeline-progress');
const timelineHandle = document.getElementById('timeline-handle');
const currentTimeDisplay = document.getElementById('current-time');
const totalDurationDisplay = document.getElementById('total-duration');
const statusMessage = document.getElementById('status-message');
const syncZoomChk = document.getElementById('sync-zoom-chk');
const resetZoomBtn = document.getElementById('reset-zoom-btn');
const layoutGroup = document.getElementById('layout-group');
const sliderDivider = document.getElementById('slider-divider');
const viewport = document.getElementById('viewport');

// Metadata elements
const metaFileA = document.getElementById('meta-file-a');
const metaFileB = document.getElementById('meta-file-b');
const metaRes = document.getElementById('meta-res');
const metaFrameTime = document.getElementById('meta-frametime');

// Set video elements transform-origin to top-left for zoom math
videoA.style.transformOrigin = '0 0';
videoB.style.transformOrigin = '0 0';

// Frame duration estimate (assumes 60 FPS, i.e. 0.016666 seconds per frame)
const FRAME_DURATION = 1.0 / 60.0;

// Fetch videos and initialize
async function init() {
    try {
        statusMessage.textContent = "Loading database...";
        const response = await fetch('/api/videos');
        if (!response.ok) throw new Error("Failed to load video list");
        videoData = await response.json();
        
        setupSceneButtons();
        loadScene(currentScene);
        setupEventListeners();
        
        statusMessage.textContent = "Ready";
    } catch (err) {
        console.error(err);
        statusMessage.textContent = "Error loading videos!";
        statusMessage.style.color = "#ef4444";
    }
}

// Set up Scene selection buttons
function setupSceneButtons() {
    document.querySelectorAll('.scene-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.scene-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentScene = btn.dataset.scene;
            loadScene(currentScene);
        });
    });
}

// Load files for a particular scene
function loadScene(sceneKey) {
    if (!videoData || !videoData.scenes[sceneKey]) return;
    
    const videos = videoData.scenes[sceneKey];
    
    // Clear Select dropdowns
    videoSelectA.innerHTML = '';
    videoSelectB.innerHTML = '';
    
    videos.forEach(vid => {
        const optionA = document.createElement('option');
        optionA.value = vid.path;
        optionA.textContent = vid.label;
        optionA.dataset.filename = vid.filename;
        videoSelectA.appendChild(optionA);
        
        const optionB = document.createElement('option');
        optionB.value = vid.path;
        optionB.textContent = vid.label;
        optionB.dataset.filename = vid.filename;
        videoSelectB.appendChild(optionB);
    });
    
    // Default selection logic:
    // Video A: Reference
    // Video B: Find level 1 or level 0 of first distortion, otherwise the second file
    const refVideo = videos.find(v => v.metric === 'reference');
    if (refVideo) {
        videoSelectA.value = refVideo.path;
    } else if (videos.length > 0) {
        videoSelectA.selectedIndex = 0;
    }
    
    // Select B default
    const compareVideo = videos.find(v => v.metric !== 'reference' && (v.level === 'level1' || v.level === null));
    if (compareVideo) {
        videoSelectB.value = compareVideo.path;
    } else if (videos.length > 1) {
        videoSelectB.selectedIndex = 1;
    } else if (videos.length > 0) {
        videoSelectB.selectedIndex = 0;
    }

    updateLevelsButtonGroup();
    updateVideos();
    resetZoom();
}

// Update level button selection states based on selected B video
function updateLevelsButtonGroup() {
    const selectedB = videoSelectB.options[videoSelectB.selectedIndex];
    if (!selectedB) return;
    
    const filename = selectedB.dataset.filename;
    const info = videoData.scenes[currentScene].find(v => v.filename === filename);
    
    document.querySelectorAll('.level-btn').forEach(btn => {
        btn.classList.remove('active');
        if (info && info.level === btn.dataset.level) {
            btn.classList.add('active');
        }
    });
}

// Change Video B to a specific level of the current metric
function setCompareLevel(levelKey) {
    const selectedB = videoSelectB.options[videoSelectB.selectedIndex];
    if (!selectedB) return;
    
    const filename = selectedB.dataset.filename;
    const info = videoData.scenes[currentScene].find(v => v.filename === filename);
    if (!info || info.metric === 'reference') return;
    
    // Find video with same metric but the target level
    const targetVideo = videoData.scenes[currentScene].find(
        v => v.metric === info.metric && v.level === levelKey
    );
    
    if (targetVideo) {
        videoSelectB.value = targetVideo.path;
        updateLevelsButtonGroup();
        updateVideos();
    } else {
        statusMessage.textContent = `No ${levelKey} available for this metric.`;
        setTimeout(() => statusMessage.textContent = isPlaying ? "Playing" : "Paused", 2000);
    }
}

// Update the video elements src when selects change
function updateVideos() {
    const pathA = videoSelectA.value;
    const pathB = videoSelectB.value;
    
    if (!pathA || !pathB) return;
    
    const currentA = videoA.src ? new URL(videoA.src).pathname : '';
    const targetA = new URL(pathA, window.location.href).pathname;
    
    const currentB = videoB.src ? new URL(videoB.src).pathname : '';
    const targetB = new URL(pathB, window.location.href).pathname;
    
    // Save current playback state
    const wasPlaying = isPlaying;
    const currentTime = videoA.currentTime;
    
    pause();
    
    if (currentA !== targetA) {
        videoA.src = pathA;
        videoA.load();
    }
    
    if (currentB !== targetB) {
        videoB.src = pathB;
        videoB.load();
    }
    
    // Show filenames in info panel
    const optA = videoSelectA.options[videoSelectA.selectedIndex];
    const optB = videoSelectB.options[videoSelectB.selectedIndex];
    metaFileA.textContent = optA ? optA.dataset.filename : '-';
    metaFileB.textContent = optB ? optB.dataset.filename : '-';
    
    // Sync time when loaded
    const syncTime = () => {
        if (videoA.readyState >= 1 && videoB.readyState >= 1) {
            videoA.currentTime = currentTime || 0;
            videoB.currentTime = currentTime || 0;
            
            // Set speed
            videoA.playbackRate = playbackSpeed;
            videoB.playbackRate = playbackSpeed;
            
            // Set loop
            videoA.loop = isLooping;
            videoB.loop = isLooping;
            
            updateMetadata();
            updateTimeline();
            
            if (wasPlaying) play();
        } else {
            setTimeout(syncTime, 50);
        }
    };
    
    syncTime();
}

// Update metadata in info panel
function updateMetadata() {
    if (videoA.readyState >= 1) {
        metaRes.textContent = `${videoA.videoWidth} x ${videoA.videoHeight}`;
        const fps = 60; // Assume 60 fps standard if we can't probe it.
        metaFrameTime.textContent = `${(1000/fps).toFixed(1)} ms (${fps} Hz)`;
    } else {
        metaRes.textContent = "Loading...";
        metaFrameTime.textContent = "Loading...";
    }
}

// Playback Logic
function play() {
    isPlaying = true;
    videoA.play().catch(e => console.log("Play interrupted A:", e));
    videoB.play().catch(e => console.log("Play interrupted B:", e));
    playPauseBtn.innerHTML = '<i class="fa-solid fa-pause"></i>';
    statusMessage.textContent = "Playing";
}

function pause() {
    isPlaying = false;
    videoA.pause();
    videoB.pause();
    // Re-align frames exactly on pause
    videoB.currentTime = videoA.currentTime;
    playPauseBtn.innerHTML = '<i class="fa-solid fa-play"></i>';
    statusMessage.textContent = "Paused";
}

function togglePlay() {
    if (isPlaying) {
        pause();
    } else {
        play();
    }
}

function setPlaybackSpeed(speed) {
    playbackSpeed = parseFloat(speed);
    videoA.playbackRate = playbackSpeed;
    videoB.playbackRate = playbackSpeed;
    speedSelect.value = speed;
}

function toggleLoop() {
    isLooping = !isLooping;
    videoA.loop = isLooping;
    videoB.loop = isLooping;
    if (isLooping) {
        loopBtn.querySelector('i').classList.add('loop-active');
    } else {
        loopBtn.querySelector('i').classList.remove('loop-active');
    }
}

// Seek both videos to time
function seekTo(time) {
    videoA.currentTime = time;
    videoB.currentTime = time;
    updateTimeline();
}

// Step frame-by-frame
function stepFrames(direction) {
    pause();
    const newTime = videoA.currentTime + (direction * FRAME_DURATION);
    seekTo(Math.max(0, Math.min(videoA.duration || 0, newTime)));
}

// Sync progress bar timeline
function updateTimeline() {
    if (!videoA.duration) return;
    
    const curr = videoA.currentTime;
    const dur = videoA.duration;
    const pct = (curr / dur) * 100;
    
    timelineProgress.style.width = `${pct}%`;
    timelineHandle.style.left = `${pct}%`;
    
    currentTimeDisplay.textContent = formatTime(curr);
    totalDurationDisplay.textContent = formatTime(dur);
}

function formatTime(secs) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const ms = Math.floor((secs % 1) * 1000);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

// Layout Switcher
function setLayout(layoutName) {
    currentLayout = layoutName;
    
    // Clear old classes
    playerContainer.className = 'player-container';
    playerContainer.classList.add(`${layoutName}-layout`);
    
    // Set active button
    document.querySelectorAll('.layout-btn').forEach(btn => {
        btn.classList.remove('active');
        if (btn.dataset.layout === layoutName) {
            btn.classList.add('active');
        }
    });

    // Reset A/B Swap override if exiting swap layout
    if (layoutName !== 'ab-swap') {
        playerContainer.classList.remove('show-b');
        showSwapB = false;
    } else {
        // Default B hidden in A/B swap
        playerContainer.classList.remove('show-b');
        showSwapB = false;
    }
    
    // Re-center zoom since layout size changed
    resetZoom();
}

// Set slider position (Split Slider Layout)
function setSliderPosition(clientX) {
    const rect = viewport.getBoundingClientRect();
    const pos = clientX - rect.left;
    sliderPosPercent = (pos / rect.width) * 100;
    sliderPosPercent = Math.min(Math.max(0, sliderPosPercent), 100);
    
    sliderDivider.style.left = `${sliderPosPercent}%`;
    wrapperB.style.clipPath = `inset(0 ${100 - sliderPosPercent}% 0 0)`; // Clump slider right
    // Wait, let's check: in CSS, B is stacked on top. Clip path was inset(0 0 0 50%) -> shows right half.
    // If slider is at 60% from left:
    // We want B to cover the RIGHT side from 60% to 100%.
    // So B is visible from 60% onwards. That means we clip B's LEFT side up to 60%.
    // Clip path inset(top right bottom left) -> inset(0 0 0 60%)
    // Let's set it:
    wrapperB.style.clipPath = `inset(0 0 0 ${sliderPosPercent}%)`;
}

// Zoom and Pan Engine
function applyZoomTransform(videoEl, zoomState, indicatorEl) {
    // Round to 1 decimal place to prevent subpixel jitter
    videoEl.style.transform = `translate(${zoomState.x}px, ${zoomState.y}px) scale(${zoomState.scale})`;
    indicatorEl.textContent = `${Math.round(zoomState.scale * 100)}%`;
}

function handleZoom(e, isVideoA) {
    e.preventDefault();
    
    const wrapper = isVideoA ? wrapperA : wrapperB;
    const zoom = isVideoA ? zoomA : zoomB;
    
    const rect = wrapper.getBoundingClientRect();
    const mouseX = e.clientX - rect.left;
    const mouseY = e.clientY - rect.top;
    
    // Zoom sensitivity
    const zoomFactor = 1.1;
    const oldScale = zoom.scale;
    let newScale = e.deltaY < 0 ? oldScale * zoomFactor : oldScale / zoomFactor;
    
    // Clamp zoom scale between 0.5x and 30x
    newScale = Math.min(Math.max(0.5, newScale), 30);
    
    if (syncZoom) {
        // Calculate new translations for both to match cursor zoom
        const scaleRatio = newScale / oldScale;
        
        // For video A
        zoomA.x = mouseX - (mouseX - zoomA.x) * scaleRatio;
        zoomA.y = mouseY - (mouseY - zoomA.y) * scaleRatio;
        zoomA.scale = newScale;
        
        // For video B (match translation)
        zoomB.x = zoomA.x;
        zoomB.y = zoomA.y;
        zoomB.scale = newScale;
        
        applyZoomTransform(videoA, zoomA, document.getElementById('zoom-ind-a'));
        applyZoomTransform(videoB, zoomB, document.getElementById('zoom-ind-b'));
    } else {
        // Individual zoom
        const scaleRatio = newScale / oldScale;
        zoom.x = mouseX - (mouseX - zoom.x) * scaleRatio;
        zoom.y = mouseY - (mouseY - zoom.y) * scaleRatio;
        zoom.scale = newScale;
        
        if (isVideoA) {
            applyZoomTransform(videoA, zoomA, document.getElementById('zoom-ind-a'));
        } else {
            applyZoomTransform(videoB, zoomB, document.getElementById('zoom-ind-b'));
        }
    }
}

function resetZoom() {
    zoomA = { scale: 1, x: 0, y: 0 };
    zoomB = { scale: 1, x: 0, y: 0 };
    
    applyZoomTransform(videoA, zoomA, document.getElementById('zoom-ind-a'));
    applyZoomTransform(videoB, zoomB, document.getElementById('zoom-ind-b'));
}

// Drag / Pan Logic
function startDrag(e, isVideoA) {
    if (e.button !== 0) return; // Left click only
    
    isDragging = true;
    activeDragWrapper = isVideoA ? 'A' : 'B';
    dragStart.x = e.clientX;
    dragStart.y = e.clientY;
    
    const activeZoom = isVideoA ? zoomA : zoomB;
    dragStartCoords.x = activeZoom.x;
    dragStartCoords.y = activeZoom.y;
}

function doDrag(e) {
    if (!isDragging) return;
    
    const deltaX = e.clientX - dragStart.x;
    const deltaY = e.clientY - dragStart.y;
    
    if (syncZoom) {
        zoomA.x = dragStartCoords.x + deltaX;
        zoomA.y = dragStartCoords.y + deltaY;
        zoomB.x = zoomA.x;
        zoomB.y = zoomA.y;
        
        applyZoomTransform(videoA, zoomA, document.getElementById('zoom-ind-a'));
        applyZoomTransform(videoB, zoomB, document.getElementById('zoom-ind-b'));
    } else {
        const activeZoom = activeDragWrapper === 'A' ? zoomA : zoomB;
        const activeVideo = activeDragWrapper === 'A' ? videoA : videoB;
        const activeIndicator = document.getElementById(activeDragWrapper === 'A' ? 'zoom-ind-a' : 'zoom-ind-b');
        
        activeZoom.x = dragStartCoords.x + deltaX;
        activeZoom.y = dragStartCoords.y + deltaY;
        
        applyZoomTransform(activeVideo, activeZoom, activeIndicator);
    }
}

function endDrag() {
    isDragging = false;
    activeDragWrapper = null;
}

// A/B Swap triggers
function setABSwapState(showB) {
    if (currentLayout !== 'ab-swap') return;
    showSwapB = showB;
    if (showSwapB) {
        playerContainer.classList.add('show-b');
        statusMessage.textContent = "Comparing: Video B";
    } else {
        playerContainer.classList.remove('show-b');
        statusMessage.textContent = "Comparing: Video A";
    }
}

// Set up all events
function setupEventListeners() {
    // Video Selects
    videoSelectA.addEventListener('change', updateVideos);
    videoSelectB.addEventListener('change', () => {
        updateLevelsButtonGroup();
        updateVideos();
    });

    // Level buttons
    document.querySelectorAll('.level-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setCompareLevel(btn.dataset.level);
        });
    });

    // Controls
    playPauseBtn.addEventListener('click', togglePlay);
    loopBtn.addEventListener('click', toggleLoop);
    stepPrevBtn.addEventListener('click', () => stepFrames(-1));
    stepNextBtn.addEventListener('click', () => stepFrames(1));
    speedSelect.addEventListener('change', (e) => setPlaybackSpeed(e.target.value));
    
    // Zoom Toggle & Reset
    syncZoomChk.addEventListener('change', (e) => {
        syncZoom = e.target.checked;
        if (syncZoom) {
            // Align zoom B to zoom A
            zoomB = { ...zoomA };
            applyZoomTransform(videoB, zoomB, document.getElementById('zoom-ind-b'));
        }
    });
    resetZoomBtn.addEventListener('click', resetZoom);

    // Layout buttons
    document.querySelectorAll('.layout-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            setLayout(btn.dataset.layout);
        });
    });

    // Custom Timeline click/drag seek
    let isDraggingTimeline = false;
    
    const handleTimelineSeek = (clientX) => {
        if (!videoA.duration) return;
        const rect = timelineWrapper.getBoundingClientRect();
        const pct = (clientX - rect.left) / rect.width;
        const targetTime = Math.min(Math.max(0, pct), 1) * videoA.duration;
        seekTo(targetTime);
    };

    timelineWrapper.addEventListener('mousedown', (e) => {
        isDraggingTimeline = true;
        handleTimelineSeek(e.clientX);
    });

    window.addEventListener('mousemove', (e) => {
        if (isDraggingTimeline) {
            handleTimelineSeek(e.clientX);
        }
        if (isDraggingSlider) {
            setSliderPosition(e.clientX);
        }
        if (isDragging) {
            doDrag(e);
        }
    });

    window.addEventListener('mouseup', () => {
        isDraggingTimeline = false;
        isDraggingSlider = false;
        endDrag();
    });

    // Video playback sync loop
    videoA.addEventListener('timeupdate', () => {
        // Only update UI timeline if we are not dragging it
        if (!isDraggingTimeline) {
            updateTimeline();
        }
        
        // Active sync checking on play to prevent drifts.
        // Relaxed threshold to 0.3s to avoid constant seek-stuttering on high-bitrate videos.
        if (isPlaying && Math.abs(videoA.currentTime - videoB.currentTime) > 0.3) {
            videoB.currentTime = videoA.currentTime;
        }
    });
    
    videoA.addEventListener('play', () => {
        if (videoB.paused) videoB.play().catch(e => {});
    });
    
    videoA.addEventListener('pause', () => {
        if (!videoB.paused) videoB.pause();
    });

    // Draggable Divider events
    sliderDivider.addEventListener('mousedown', (e) => {
        e.preventDefault();
        isDraggingSlider = true;
    });

    // Zoom & Pan Wheel events on wrappers
    wrapperA.addEventListener('wheel', (e) => handleZoom(e, true));
    wrapperB.addEventListener('wheel', (e) => handleZoom(e, false));

    // Drag Pan events
    wrapperA.addEventListener('mousedown', (e) => startDrag(e, true));
    wrapperB.addEventListener('mousedown', (e) => startDrag(e, false));

    // A/B Swap Click Trigger
    wrapperA.addEventListener('click', (e) => {
        if (currentLayout === 'ab-swap') {
            setABSwapState(!showSwapB);
        }
    });
    wrapperB.addEventListener('click', (e) => {
        if (currentLayout === 'ab-swap') {
            setABSwapState(!showSwapB);
        }
    });

    // Global Key Down (Hotkeys)
    window.addEventListener('keydown', (e) => {
        // Skip hotkeys if focused on input/select
        if (document.activeElement.tagName === 'SELECT' || document.activeElement.tagName === 'INPUT') {
            return;
        }

        switch (e.code) {
            case 'Space':
                e.preventDefault();
                togglePlay();
                break;
            case 'ArrowLeft':
                e.preventDefault();
                stepFrames(-1);
                break;
            case 'ArrowRight':
                e.preventDefault();
                stepFrames(1);
                break;
            case 'KeyZ':
                e.preventDefault();
                resetZoom();
                break;
            case 'KeyS':
                e.preventDefault();
                setLayout(currentLayout === 'side-by-side' ? 'single-a' : 'side-by-side');
                break;
            case 'KeyD':
                e.preventDefault();
                setLayout(currentLayout === 'slider' ? 'side-by-side' : 'slider');
                break;
            case 'Tab':
                e.preventDefault();
                if (currentLayout === 'ab-swap') {
                    setABSwapState(true);
                } else {
                    // Temporarily force ab-swap layout on Tab hold
                    // Let's just handle it in ab-swap mode for consistency
                }
                break;
        }
    });

    // Global Key Up (Hotkeys)
    window.addEventListener('keyup', (e) => {
        if (e.code === 'Tab') {
            e.preventDefault();
            if (currentLayout === 'ab-swap') {
                setABSwapState(false);
            }
        }
    });
}

// Run initializer
window.addEventListener('DOMContentLoaded', init);
