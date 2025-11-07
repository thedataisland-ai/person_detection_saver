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

# Common options:
# --streams: Space-separated list of RTSP URLs or video file paths
# --model: YOLO model size (default: yolo11n), e.g., yolov8n, yolov8s
# --tail_length: Seconds without person to stop recording (default: 2)
# --confidence: YOLO confidence threshold (default: 0.5)
# --auto_delete: Delete clips shorter than 3 seconds
# --output_dir: Directory to save clips (default: recordings)
# --port: HTTP server port (default: 7860)
# --target_fps: MJPEG stream fps (default: 12)
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
2. **Thread Safety**: YOLO model predictions use `model_lock` to prevent concurrent inference
3. **RTSP Reconnection**: Automatic reconnection with 5-second retry interval for RTSP streams
4. **Frame Queue**: 5-frame buffer (`maxsize=5`) between capture and processing to handle timing differences
5. **Motion Detection**: Uses Gaussian blur + absolute difference + MSE for motion calculation (though primarily relies on YOLO)
6. **Timestamp Management**: For file sources, uses frame index and FPS to seek to correct position when starting recording
7. **Graceful Shutdown**: FastAPI shutdown event ensures all recordings are properly saved
8. **Countdown Display**: Web overlay shows countdown timer when recording and no person detected

## Testing/Development Notes

- The application requires a YOLO model file (e.g., `yolo11n.pt`) which will be auto-downloaded by Ultralytics on first run
- Test with video files before RTSP streams to verify setup
- Use `--verbose` flag to see detailed detection statistics during development
- The web interface auto-refreshes status every second via JavaScript fetch to `/status`
