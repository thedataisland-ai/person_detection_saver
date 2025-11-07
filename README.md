## Installation and Usage

```bash
# Clone the repository
git clone https://github.com/laurent-rodz/yolo_rtsp.git
cd yolo_rtsp

# Create and activate a virtual environment
py -3.11 -m venv .venv

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Reinstall OpenCV to ensure compatibility
.venv\scripts\pip uninstall -y opencv-python-headless opencv-contrib-python-headless
.venv\scripts\pip install --upgrade --force-reinstall opencv-python

# Install additional packages
.venv\scripts\pip install python-ffmpeg
.venv\scripts\pip install ultralytics
.venv\scripts\pip install scikit-image
.venv\scripts\pip install fastapi
.venv\scripts\pip install uvicorn

# Run YOLO RTSP with multiple video streams
.venv\scripts\python yolo_rtsp_multi.py \
  --streams "C:/Users/ilaur/yolo_rtsp/hikvision.mp4" \
            "C:/Users/ilaur/yolo_rtsp/hikvision.mp4" \
            "C:/Users/ilaur/yolo_rtsp/hikvision.mp4" \
            "C:/Users/ilaur/yolo_rtsp/hikvision.mp4" \
  --yolo person \
  --tail_length 1 \
  --start_frames 1 \
  --port 7860

> **Note:** Replace the sample video paths with your own RTSP URLs.  
> For multiple cameras, specify multiple RTSP URLs separated by spaces.
