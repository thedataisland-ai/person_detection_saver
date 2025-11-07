# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a smart person detection recorder that uses YOLO object detection on RTSP streams or video files. It automatically starts recording when a person is detected and stops when the person detection rate drops below a threshold over a sliding time window.

## Environment Setup

```bash
# Create virtual environment with Python 3.11
py -3.11 -m venv .venv

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Fix OpenCV compatibility issues
.venv\scripts\pip uninstall -y opencv-python-headless opencv-contrib-python-headless
.venv\scripts\pip install --upgrade --force-reinstall opencv-python

# Install additional packages
.venv\scripts\pip install python-ffmpeg
.venv\scripts\pip install ultralytics
.venv\scripts\pip install scikit-image
.venv\scripts\pip install fastapi
.venv\scripts\pip install uvicorn
```

## Running the Application

```bash
# Basic usage with multiple streams
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "path/to/video1.mp4" "rtsp://camera1" \
  --yolo person \
  --tail_length 2 \
  --port 7860

# With custom resolution (e.g., 720p)
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "path/to/video.mp4" \
  --resolution 1280x720 \
  --tail_length 2

# With manual recording FPS override (if recordings play at wrong speed)
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "path/to/video.mp4" \
  --fps 25 \
  --verbose

# Reduce web bandwidth without affecting recording quality
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "path/to/video.mp4" \
  --target_fps 12

# Use both independently (recording 30fps, web stream 12fps)
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "path/to/video.mp4" \
  --fps 30 \
  --target_fps 12

# Common options:
# --streams: Space-separated list of RTSP URLs or video file paths
# --model: YOLO model size (default: yolo11n), e.g., yolov8n, yolov8s
# --tail_length: Seconds without person to stop recording (default: 2)
# --confidence: YOLO confidence threshold (default: 0.5)
# --resolution: Output video resolution, e.g., 1920x1080, 1280x720 (default: original video resolution)
#
# FPS options (both independent, both default to source FPS):
# --fps: Recording FPS override (e.g., 30, 25, 24) - use if saved videos play at wrong speed
# --target_fps: Web stream FPS (e.g., 12) - lower this to reduce web bandwidth, doesn't affect recordings
#
# --auto_delete: Delete clips shorter than 3 seconds
# --output_dir: Directory to save clips (default: recordings)
# --port: HTTP server port (default: 7860)
# --verbose: Show detailed detection stats
```

## Architecture

### Core Components

**CameraWorker Class** (`yolo_rtsp_multi.py:111-445`)
- Manages individual camera streams with two threads:
  - `_rx_loop`: Reads frames from video source (RTSP or file) with automatic reconnection for RTSP
  - `_proc_loop`: Processes frames with YOLO detection and manages recording state
- Handles FFmpeg-based recording to MKV files using `python-ffmpeg` library
- Implements consecutive frame counting for smart start/stop based on person detection

**Detection Logic** (`yolo_rtsp_multi.py:257-280`)
- Only detects "person" class (PERSON_CLASS_ID from coco.names)
- Tracks consecutive frames WITHOUT person detection (`consecutive_no_person` counter)
- Max threshold = fps × tail_length (e.g., 30 fps × 2s = 60 frames)
- Starts recording immediately on first person detection
- Stops recording after tail_length seconds (default: 2s) of consecutive frames without person
- Counter resets to 0 whenever a person is detected or recording stops
- Simple and reliable: no complex percentage calculations

**Recording Management** (`yolo_rtsp_multi.py:314-408`)
- Creates temporary files with "_RECORDING.mkv" suffix during active recording
- Renames to final format on completion: `cam{id}_{date}_{start_time}_to_{end_time}.mkv`
- Files organized in date-based folders under output_dir
- For RTSP: uses direct stream copy (vcodec=copy, acodec=copy)
- For files: uses seek with `-ss` to start from current position
- Optional auto-deletion of clips shorter than 3 seconds

**Web Interface** (FastAPI app, `yolo_rtsp_multi.py:451-583`)
- MJPEG streaming endpoint: `/video/{cam_id}` streams annotated frames
- Status endpoint: `/status` returns JSON with recording state for all cameras
- HTML dashboard with auto-refresh showing all camera feeds
- Automatically opens browser on startup

### Dependencies

- **OpenCV (cv2)**: Video capture and frame processing
- **Ultralytics YOLO**: Person detection model
- **python-ffmpeg**: Recording video streams to disk
- **FastAPI + Uvicorn**: Web server for streaming and status
- **scikit-image**: Motion detection using MSE
- **numpy**: Frame processing

### Key Files

- `yolo_rtsp_multi.py`: Main application (single file architecture)
- `coco.names`: COCO class labels (required for PERSON_CLASS_ID lookup)
- `requirements.txt`: Python dependencies
- `recordings/`: Default output directory (organized by date)

## Important Implementation Details

1. **Detection Algorithm**: Uses consecutive frame counting (not sliding window percentage) - stops recording after N consecutive frames without person detection, where N = fps × tail_length
2. **Custom Resolution**: Use `--resolution` to specify output video size (e.g., 1280x720). Frames are automatically resized during recording. Default: uses original video resolution.
3. **FPS Control (Independent Parameters)**:
   - `--fps`: Controls **recording FPS** (default: auto-detect from source). Use if saved videos play at wrong speed.
   - `--target_fps`: Controls **web stream FPS** (default: auto-detect from source). Lower to reduce bandwidth without affecting recordings.
   - Both are completely independent and both default to source FPS for smooth playback.
4. **Thread Safety**: YOLO model predictions use `model_lock` to prevent concurrent inference
5. **RTSP Reconnection**: Automatic reconnection with 5-second retry interval for RTSP streams
6. **Frame Queue**: 5-frame buffer (`maxsize=5`) between capture and processing to handle timing differences
7. **Motion Detection**: Uses Gaussian blur + absolute difference + MSE for motion calculation (though primarily relies on YOLO)
8. **Timestamp Management**: For file sources, uses frame index and FPS to seek to correct position when starting recording
9. **Graceful Shutdown**: FastAPI shutdown event ensures all recordings are properly saved
10. **Countdown Display**: Web overlay shows countdown timer when recording and no person detected

## Testing/Development Notes

- The application requires a YOLO model file (e.g., `yolo11n.pt`) which will be auto-downloaded by Ultralytics on first run
- Test with video files before RTSP streams to verify setup
- Use `--verbose` flag to see detailed detection statistics during development
- The web interface auto-refreshes status every second via JavaScript fetch to `/status`

## Troubleshooting Fast Playback

If saved recordings play back too fast:

1. **Check detected FPS**: Run with `--verbose` and look for `🎬 Detected FPS: X.X`
2. **Override FPS manually**: Use `--fps` to set the correct playback speed

```bash
# If recordings play 2x too fast, halve the FPS
.venv\scripts\python yolo_rtsp_multi.py --streams "video.mp4" --fps 12.5

# Common FPS values to try:
--fps 24   # Cinema standard
--fps 25   # PAL video (Europe)
--fps 30   # NTSC video (USA)
--fps 15   # Half speed of 30fps
--fps 12.5 # Half speed of 25fps
```

3. **Check output message**: After saving, the script shows the FPS used and suggests alternatives if playback is too fast
