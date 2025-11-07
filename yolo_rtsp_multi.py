import os
import sys
import cv2
import time
import queue
import threading
import socket
import webbrowser
import numpy as np
from datetime import datetime
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, BooleanOptionalAction

from ffmpeg import FFmpeg
from ffmpeg.errors import FFmpegError
from ultralytics import YOLO
from skimage.metrics import mean_squared_error as ssim

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
import uvicorn


parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument("--streams", nargs="+", required=False,
                    help="List of RTSP URLs or file paths (space-separated).")
parser.add_argument("--stream", type=str, help="Single RTSP/file stream (deprecated).")
parser.add_argument("--model", default="yolo11n", type=str,
                    help="Ultralytics model size (e.g., yolov8n, yolov8s) without .pt")
parser.add_argument("--tail_length", default=2, type=int, choices=range(1, 30),
                    help="Seconds without person to STOP recording (uses PERCENTAGE method).")
parser.add_argument("--person_threshold", default=0.3, type=float,
                    help="Minimum person detection rate to keep recording (0.0-1.0, default 0.3 = 30%%).")
parser.add_argument("--confidence", default=0.5, type=float,
                    help="YOLO confidence threshold (0.0-1.0).")
parser.add_argument("--auto_delete", default=False, action=BooleanOptionalAction,
                    help="Delete clips shorter than 3 seconds.")
parser.add_argument("--port", default=7860, type=int, help="HTTP port.")
parser.add_argument("--host", default="0.0.0.0", type=str, help="HTTP host.")
parser.add_argument("--target_fps", default=12, type=int, help="MJPEG stream fps to clients.")
parser.add_argument("--output_dir", type=str, default="recordings",
                    help="Directory to save clips.")
parser.add_argument("--verbose", default=False, action=BooleanOptionalAction,
                    help="Show detailed detection stats.")
args = parser.parse_args()

streams = args.streams or ([args.stream] if args.stream else [])
if not streams:
    sys.exit("❌ Provide --streams <s1> <s2> ... (or --stream <s>)")

os.makedirs(args.output_dir, exist_ok=True)

print("="*70)
print("🎥 Smart Person Detection Recorder")
print("="*70)
print(f"📁 Output: {args.output_dir}")
print(f"🎯 Model: {args.model}.pt | Confidence: {args.confidence}")
print(f"⏱️  Stop if <{args.person_threshold*100:.0f}% detection over {args.tail_length}s window")
print(f"🚀 Starts IMMEDIATELY on first person detection")
print(f"👤 Detects ONLY person class (ignores all other objects)")
print("="*70)

labels = open("coco.names").read().strip().split("\n")
model = YOLO(args.model + ".pt")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
model_lock = threading.Lock()
PERSON_CLASS_ID = labels.index("person")

def is_rtsp_url(u: str) -> bool:
    u = str(u).lower()
    return u.startswith("rtsp://") or u.startswith("rtsps://")

def safe_fps(cap) -> float:
    f = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not np.isfinite(f) or f <= 0:
        return 30.0
    return float(f)

def fmt_ts(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h*3600 - m*60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

class suppress_stdout_stderr(object):
    def __enter__(self):
        self.outnull_file = open(os.devnull, 'w')
        self.errnull_file = open(os.devnull, 'w')
        self.old_stdout_fileno_undup = sys.stdout.fileno()
        self.old_stderr_fileno_undup = sys.stderr.fileno()
        self.old_stdout_fileno = os.dup(sys.stdout.fileno())
        self.old_stderr_fileno = os.dup(sys.stderr.fileno())
        self.old_stdout = sys.stdout
        self.old_stderr = sys.stderr
        os.dup2(self.outnull_file.fileno(), self.old_stdout_fileno_undup)
        os.dup2(self.errnull_file.fileno(), self.old_stderr_fileno_undup)
        sys.stdout = self.outnull_file
        sys.stderr = self.errnull_file
        return self
    def __exit__(self, *_):
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        os.dup2(self.old_stdout_fileno, self.old_stdout_fileno_undup)
        os.dup2(self.old_stderr_fileno, self.old_stderr_fileno_undup)
        os.close(self.old_stdout_fileno)
        os.close(self.old_stderr_fileno)
        self.outnull_file.close()
        self.errnull_file.close()

class CameraWorker:
    def __init__(self, cam_id: int, source: str):
        self.cam_id = cam_id
        self.source = source
        self.cap = None
        self.is_rtsp = is_rtsp_url(source)

        self.q = queue.Queue(maxsize=5)
        self.thread_rx = None
        self.thread_proc = None
        self.running = threading.Event()
        self.running.clear()

        self.fps = 30.0
        self.period = 1.0 / self.fps
        self.window_size = 0  # Will be calculated based on tail_length
        
        self.recording = False
        self.ffmpeg_proc = None
        self.ffmpeg_thread = None
        self.filename = None
        self.temp_filename = None
        self.recording_start_time = None

        self.frame_idx = 0
        self.rec_running = False

        # SMART DETECTION: Track recent frames
        self.detection_window = []  # List of booleans: True = person, False = no person
        
        self.res = (256, 144)
        self.blank = np.zeros((self.res[1], self.res[0]), np.uint8)
        self.old_frame = None

        self.last_bgr = None
        self.last_jpeg = None
        self.last_lock = threading.Lock()

    def open(self):
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not self.cap.isOpened():
            print(f"[cam{self.cam_id}] ❌ Unable to open: {self.source}")
            return
            
        self.fps = safe_fps(self.cap)
        self.period = 1.0 / self.fps
        self.window_size = int(self.fps * args.tail_length)  # e.g., 30 fps * 2 sec = 60 frames

        ok, img0 = self.cap.read()
        if ok and img0 is not None:
            h, w = img0.shape[:2]
            self.res = (256, 144) if (w / h) > 1.55 else (216, 162)
            gray = cv2.cvtColor(cv2.resize(img0, self.res), cv2.COLOR_BGR2GRAY)
            self.old_frame = cv2.GaussianBlur(gray, (5, 5), 0)
            self.blank = np.zeros((self.res[1], self.res[0]), np.uint8)
            self.last_bgr = img0
            print(f"[cam{self.cam_id}] ✅ Ready | FPS: {self.fps:.1f} | Window: {self.window_size} frames ({args.tail_length}s)")
            print(f"[cam{self.cam_id}] 📊 Will stop if person detected in <{args.person_threshold*100:.0f}% of last {self.window_size} frames")

    def _video_ts(self) -> str:
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    def _rx_loop(self):
        while self.running.is_set():
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    try:
                        self.q.put(frame, timeout=0.01)
                    except queue.Full:
                        pass
                    if not self.is_rtsp:
                        time.sleep(self.period)
                else:
                    if self.recording:
                        print(f"[cam{self.cam_id}] ⚠️  Stream ended, saving clip...")
                        self._stop_recording()
                    if self.is_rtsp:
                        print(datetime.now().strftime('%H-%M-%S'), f"[cam{self.cam_id}] disconnected. Reconnecting…")
                        while self.running.is_set():
                            with suppress_stdout_stderr():
                                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                            if self.cap.isOpened():
                                self.fps = safe_fps(self.cap)
                                self.period = 1.0 / self.fps
                                self.window_size = int(self.fps * args.tail_length)
                                print(datetime.now().strftime('%H-%M-%S'), f"[cam{self.cam_id}] reconnected.")
                                break
                            time.sleep(5)
                    else:
                        break
            else:
                time.sleep(0.1)

    def _proc_loop(self):
        while self.running.is_set():
            if self.q.empty():
                time.sleep(0.005)
                continue

            img = self.q.get()
            self.frame_idx += 1

            resized = cv2.resize(img, self.res)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            final = cv2.GaussianBlur(gray, (5, 5), 0)
            if self.old_frame is None:
                self.old_frame = final

            diff = cv2.absdiff(final, self.old_frame)
            result = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
            motion_val = int(ssim(result, self.blank))
            self.old_frame = final

            # YOLO: ONLY DETECT PERSON
            person_detected = False
            person_count = 0
            
            with model_lock:
                results = model.predict(img, conf=args.confidence, verbose=False)[0]
            
            if results.boxes is not None:
                for data in results.boxes.data.tolist():
                    xmin, ymin, xmax, ymax, conf, cls_id = data
                    cls_id = int(cls_id)
                    
                    # ONLY PROCESS PERSON CLASS
                    if cls_id == PERSON_CLASS_ID:
                        person_detected = True
                        person_count += 1
                        
                        # Draw person detection
                        color = [int(c) for c in colors[cls_id]]
                        cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)),
                                      color=color, thickness=2)
                        text = f"Person: {conf:.2f}"
                        (tw, th) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                                   fontScale=0.7, thickness=2)[0]
                        tx, ty = int(xmin), int(ymin) - 5
                        overlay = img.copy()
                        cv2.rectangle(overlay, (tx, ty), (tx + tw + 2, ty - th), color=color, thickness=cv2.FILLED)
                        img[:] = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)
                        cv2.putText(img, text, (int(xmin), int(ymin) - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=0.7, color=(0, 0, 0), thickness=2)

            # UPDATE DETECTION WINDOW
            self.detection_window.append(person_detected)
            if len(self.detection_window) > self.window_size:
                self.detection_window.pop(0)

            # CALCULATE DETECTION RATE over window
            if len(self.detection_window) >= self.window_size:
                person_frames = sum(self.detection_window)
                detection_rate = person_frames / len(self.detection_window)
            else:
                detection_rate = 1.0 if person_detected else 0.0

            # SMART RECORDING LOGIC
            if not self.recording:
                # NOT RECORDING - start immediately on first person
                if person_detected:
                    self._start_recording()
                    self.recording = True
                    if args.verbose:
                        print(f"[cam{self.cam_id}] 🔴 Person detected! Starting recording immediately")
            else:
                # RECORDING - check if person rate dropped below threshold
                if len(self.detection_window) >= self.window_size:
                    if detection_rate < args.person_threshold:
                        person_frames = sum(self.detection_window)
                        print(f"[cam{self.cam_id}] 🛑 Person rate dropped to {detection_rate*100:.1f}% ({person_frames}/{len(self.detection_window)} frames)")
                        print(f"[cam{self.cam_id}] Below threshold of {args.person_threshold*100:.0f}%, stopping recording...")
                        self._stop_recording()
                        self.recording = False
                        self.detection_window = []  # Clear window
                    elif args.verbose and self.frame_idx % 30 == 0:
                        person_frames = sum(self.detection_window)
                        print(f"[cam{self.cam_id}] 📊 Rate: {detection_rate*100:.1f}% ({person_frames}/{len(self.detection_window)}) | Threshold: {args.person_threshold*100:.0f}%")

            # OVERLAY INFO
            status = '🔴 REC' if self.recording else '⚪ IDLE'
            person_status = f"Person: {'YES' if person_detected else 'NO'}"
            if person_count > 0:
                person_status += f" ({person_count})"
            
            if self.recording and len(self.detection_window) >= 10:
                recent_detections = sum(self.detection_window[-30:]) if len(self.detection_window) >= 30 else sum(self.detection_window)
                recent_total = min(30, len(self.detection_window))
                recent_rate = (recent_detections / recent_total) * 100
                person_status += f" | Rate: {recent_rate:.0f}%"
                
            header = f"Cam {self.cam_id} | {status} | {person_status}"
            cv2.putText(img, header, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            ts_text = self._video_ts()
            cv2.putText(img, f"TS: {ts_text}", (10, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            with self.last_lock:
                self.last_bgr = img
                ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                self.last_jpeg = buf.tobytes() if ok else None

    def _start_recording(self):
        if self.recording:
            return False
            
        self.recording_start_time = datetime.now()
        start_datestr = self.recording_start_time.strftime('%Y-%m-%d')
        start_timestr = self.recording_start_time.strftime('%H-%M-%S')
        
        folder_path = os.path.join(args.output_dir, start_datestr)
        os.makedirs(folder_path, exist_ok=True)
        
        self.temp_filename = os.path.join(folder_path, f"cam{self.cam_id}_{start_timestr}_RECORDING.mkv")

        ff = FFmpeg().option("y")

        if self.is_rtsp:
            ff = ff.input(self.source, rtsp_transport="tcp")
        else:
            now_sec = self.frame_idx / max(1e-6, self.fps)
            ff = ff.option("re").input(self.source, ss=fmt_ts(now_sec))

        self.ffmpeg_proc = ff.output(self.temp_filename, vcodec="copy", acodec="copy")

        def _run():
            self.rec_running = True
            try:
                self.ffmpeg_proc.execute()
            except Exception as e:
                if args.verbose:
                    print(f"[cam{self.cam_id}] FFmpeg: {e}")
            finally:
                self.rec_running = False

        self.ffmpeg_thread = threading.Thread(target=_run, daemon=True)
        self.ffmpeg_thread.start()
        print(f"{start_timestr} [cam{self.cam_id}] 🔴 REC START → {self.temp_filename}")
        return True

    def _stop_recording(self):
        if not self.recording and not self.rec_running:
            return
            
        recording_end_time = datetime.now()
        end_timestr = recording_end_time.strftime('%H-%M-%S')

        proc = self.ffmpeg_proc
        th = self.ffmpeg_thread
        old_temp_filename = self.temp_filename

        self.ffmpeg_proc = None
        self.ffmpeg_thread = None
        self.temp_filename = None

        if proc and self.rec_running:
            try:
                proc.terminate()
            except Exception as e:
                if "not executed" not in str(e).lower() and args.verbose:
                    print(f"[cam{self.cam_id}] terminate error: {e}")

        if th and th.is_alive():
            th.join(timeout=3.0)

        self.rec_running = False
        
        if old_temp_filename and os.path.isfile(old_temp_filename):
            start_datestr = self.recording_start_time.strftime('%Y-%m-%d')
            start_timestr = self.recording_start_time.strftime('%H-%M-%S')
            
            folder_path = os.path.join(args.output_dir, start_datestr)
            final_filename = os.path.join(folder_path, 
                                        f"cam{self.cam_id}_{start_datestr}_{start_timestr}_to_{end_timestr}.mkv")
            
            file_size = os.path.getsize(old_temp_filename)
            duration = (recording_end_time - self.recording_start_time).total_seconds()
            
            should_delete = False
            if args.auto_delete and duration < 3.0:
                should_delete = True
            
            if should_delete:
                os.remove(old_temp_filename)
                print(f"{end_timestr} [cam{self.cam_id}] 🗑️  Deleted (too short: {duration:.1f}s)")
            else:
                try:
                    os.rename(old_temp_filename, final_filename)
                    print(f"{end_timestr} [cam{self.cam_id}] ✅ SAVED: {duration:.1f}s, {file_size/1024/1024:.1f}MB → {final_filename}")
                    self.filename = final_filename
                except Exception as e:
                    print(f"[cam{self.cam_id}] ❌ Rename error: {e}")
        else:
            print(f"{end_timestr} [cam{self.cam_id}] ⚠️  File not found")
        
        self.filename = None
        self.recording_start_time = None

    def start(self):
        self.open()
        self.running.set()
        self.thread_rx = threading.Thread(target=self._rx_loop, daemon=True)
        self.thread_proc = threading.Thread(target=self._proc_loop, daemon=True)
        self.thread_rx.start()
        self.thread_proc.start()

    def stop(self):
        self.running.clear()
        if self.recording or self.rec_running:
            self._stop_recording()
        if self.thread_rx:
            self.thread_rx.join(timeout=1.0)
        if self.thread_proc:
            self.thread_proc.join(timeout=1.0)
        if self.cap:
            self.cap.release()

    def mjpeg_generator(self, target_fps: int = 12):
        boundary = b"--frame"
        interval = 1.0 / max(1, target_fps)
        next_ts = time.perf_counter()
        while True:
            with self.last_lock:
                data = self.last_jpeg
            if data is not None:
                yield boundary + b"\r\n"
                yield b"Content-Type: image/jpeg\r\n"
                yield b"Content-Length: " + str(len(data)).encode() + b"\r\n\r\n"
                yield data + b"\r\n"
            now = time.perf_counter()
            if now < next_ts:
                time.sleep(next_ts - now)
            next_ts += interval


workers = [CameraWorker(i + 1, s) for i, s in enumerate(streams)]
for w in workers:
    w.start()

app = FastAPI(title="Smart Person Detection")

HTML_INDEX = """<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>Smart Person Detection</title>
<style>
body { font-family: system-ui, sans-serif; margin: 0; background:#0b1020; color:#eaeaea;}
.header { background: #1a2332; padding: 20px; text-align: center; box-shadow: 0 2px 10px rgba(0,0,0,.3); }
.header h1 { margin: 0 0 10px 0; font-size: 28px; }
.header p { margin: 0; opacity: 0.8; font-size: 13px; }
.grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)); gap:12px; padding:12px;}
.card { position: relative; background:#141b34; border-radius:12px; padding:8px; box-shadow:0 6px 20px rgba(0,0,0,.25); }
h3 { margin:6px 0 8px 6px; font-weight:600; }
img { width:100%; border-radius:8px; background:#000; }
.badge {
  position:absolute; top:10px; right:10px;
  background: rgba(0,0,0,.6); color:#fff; padding:6px 10px;
  border-radius: 8px; font-size: 14px; font-weight: 600;
}
.status { margin-top: 8px; padding: 8px; background: rgba(0,0,0,0.3); border-radius: 6px; font-size: 13px; }
.status-item { display: flex; justify-content: space-between; padding: 4px 0; }
.rec-active { background: rgba(255,0,0,0.15) !important; }
</style>
</head>
<body>
  <div class="header">
    <h1>🎥 Smart Person Detection</h1>
    <p>Starts immediately • Only detects person • Stops when detection rate drops</p>
  </div>
  <div class="grid">
  %CARDS%
  </div>

  <script>
    async function refresh() {
      try {
        const res = await fetch('/status', { cache: 'no-store' });
        const data = await res.json();
        for (const cam of data) {
          const tsEl = document.getElementById('ts-' + cam.camera);
          const statusEl = document.getElementById('status-' + cam.camera);
          const cardEl = document.getElementById('card-' + cam.camera);
          
          if (tsEl) tsEl.textContent = cam.timestamp || '--:--:--';
          if (cardEl) {
            if (cam.recording) {
              cardEl.classList.add('rec-active');
            } else {
              cardEl.classList.remove('rec-active');
            }
          }
          if (statusEl) {
            const recStatus = cam.recording ? '🔴 Recording' : '⚪ Standby';
            const fileName = cam.filename ? cam.filename.split('/').pop() : 'Waiting for person...';
            statusEl.innerHTML = `
              <div class="status-item"><span>Status:</span><span>${recStatus}</span></div>
              <div class="status-item"><span>FPS:</span><span>${cam.fps.toFixed(1)}</span></div>
              ${cam.recording ? `<div class="status-item"><span>File:</span><span style="font-size:11px">${fileName}</span></div>` : ''}
            `;
          }
        }
      } catch (e) {}
    }
    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"""

def _card(i):
    return f"""
    <div class="card" id="card-{i}">
      <div class="badge" id="ts-{i}">--:--:--</div>
      <h3>Camera {i}</h3>
      <img src="/video/{i}" />
      <div class="status" id="status-{i}">
        <div class="status-item"><span>Status:</span><span>Initializing...</span></div>
      </div>
      <small>Person-only detection • Rate shown on overlay</small>
    </div>
    """

@app.get("/", response_class=HTMLResponse)
def index():
    cards = "".join(_card(i+1) for i in range(len(workers)))
    return HTML_INDEX.replace("%CARDS%", cards)

@app.get("/video/{cam_id}")
def video(cam_id: int):
    if cam_id < 1 or cam_id > len(workers):
        raise HTTPException(404, "No such camera")
    gen = workers[cam_id - 1].mjpeg_generator(target_fps=args.target_fps)
    return StreamingResponse(gen, media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/status", response_class=JSONResponse)
def status():
    out = []
    for w in workers:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        out.append({
            "camera": w.cam_id,
            "source": w.source,
            "fps": w.fps,
            "recording": w.recording,
            "filename": w.filename,
            "timestamp": ts,
        })
    return out

@app.on_event("shutdown")
def _shutdown():
    for w in workers:
        w.stop()

def open_browser_when_ready(host, port, path="/"):
    def _worker():
        target_host = "127.0.0.1" if host in ("0.0.0.0", "::") else host
        url = f"http://{target_host}:{port}{path}"
        for _ in range(60):
            try:
                with socket.create_connection((target_host, port), timeout=0.5):
                    webbrowser.open(url)
                    return
            except OSError:
                time.sleep(0.2)
        webbrowser.open(url)
    threading.Thread(target=_worker, daemon=True).start()

if __name__ == "__main__":
    open_browser_when_ready(args.host, args.port, "/")
    uvicorn.run(app, host=args.host, port=args.port, reload=False)