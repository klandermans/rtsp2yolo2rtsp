import cv2
import numpy as np
import subprocess
import threading
import time
from ultralytics import YOLO
import sys

# Insert your own YouTube Stream Key here
YOUTUBE_STREAM_KEY = ""
YOUTUBE_URL = f"rtmp://a.rtmp.youtube.com/live2/{YOUTUBE_STREAM_KEY}"

# RTSP camera sources
CAMERA_URLS = {
    "cam1": "rtsp://u:p@10.82.16.217//onvif-media/media.amp",
    "cam2": "rtsp://u:p@10.82.16.218//onvif-media/media.amp",
    "cam4": "rtsp://u:p@10.82.16.221//onvif-media/media.amp",
    "cam7": "rtsp://u:p@10.82.16.224//onvif-media/media.amp",
}

# Load the YOLOv8 model
model = YOLO("/home/bert/site/models/yolov8/best.pt")

# Variables for thread safety and frame storage
frame_lock = threading.Lock()
latest_frames = {cam_id: None for cam_id in CAMERA_URLS}
COLORS = [(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for _ in range(50)]

def draw_img_results(img, boxes, keypoints, ids=None):
    if keypoints is None or len(keypoints) == 0:
        return img

    for i, kps in enumerate(keypoints):
        if len(kps) < 4:
            continue

        color = COLORS[int(ids[i]) % len(COLORS)] if ids is not None else (0, 255, 0)
        for x, y in kps[:4]:
            if (x, y) != (0, 0):
                cv2.circle(img, (int(x), int(y)), 15, color, -1)

        # Draw lines between specific keypoints
        connections = [(0, 2), (1, 2), (2, 3)]
        for p1, p2 in connections:
            x1, y1 = map(int, kps[p1])
            x2, y2 = map(int, kps[p2])
            if (x1, y1) != (0, 0) and (x2, y2) != (0, 0):
                cv2.line(img, (x1, y1), (x2, y2), color, 8)

    return img

def process_camera(cam_id, url):
    cap = cv2.VideoCapture(url)
    while True:
        success, frame = cap.read()
        if not success:
            print(f"[{cam_id}] No frame received, waiting 1s...")
            time.sleep(1)
            continue

        try:
            # Run YOLO tracking on the frame
            results = model.track(source=frame, conf=0.5, iou=0.5, stream=False, tracker="bytetrack.yaml")[0]
            boxes = results.boxes.xyxy.cpu().numpy() if results.boxes else []
            ids = results.boxes.id.cpu().numpy() if results.boxes and results.boxes.id is not None else []
            keypoints = results.keypoints.xy.cpu().numpy() if results.keypoints and results.keypoints.xy is not None else []
            frame = draw_img_results(frame, boxes, keypoints=keypoints, ids=ids)
        except Exception as e:
            print(f"[{cam_id}] YOLO error: {e}")

        # Resize the frame to 960x540 for 1920x1080 grid
        frame = cv2.resize(frame, (960, 540))
        with frame_lock:
            latest_frames[cam_id] = frame

def start_ffmpeg_stream():
    command = [
        'ffmpeg',
        '-f', 'lavfi',
        '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',  # Add silent audio
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '1920x1080',
        '-r', '25',
        '-i', '-',  # Read input from stdin
        '-c:v', 'libx264',
        '-preset', 'veryfast',
        '-b:v', '6000k',
        '-maxrate', '6000k',
        '-bufsize', '12000k',
        '-g', '50',
        '-c:a', 'aac',
        '-ar', '44100',
        '-b:a', '128k',
        '-pix_fmt', 'yuv420p',
        '-f', 'flv',
        YOUTUBE_URL
    ]

    try:
        return subprocess.Popen(command, stdin=subprocess.PIPE, stderr=sys.stderr)
    except Exception as e:
        print(f"❌ Failed to start FFmpeg: {e}")
        return None

def combine_and_stream():
    ffmpeg = start_ffmpeg_stream()
    if ffmpeg is None:
        print("❌ Could not start FFmpeg. Check installation or stream key.")
        return

    print("✅ Streaming started to YouTube in 1920x1080...")

    while True:
        with frame_lock:
            frames = [latest_frames.get(k) for k in CAMERA_URLS]

        if all(f is not None for f in frames):
            # Combine four frames into a 2x2 grid
            top = np.hstack((frames[0], frames[1]))
            bottom = np.hstack((frames[2], frames[3]))
            grid = np.vstack((top, bottom))  # 1920x1080
        else:
            # Show black frames as fallback
            black = np.zeros((540, 960, 3), dtype=np.uint8)
            grid = np.vstack((
                np.hstack((black, black)),
                np.hstack((black, black))
            ))

        try:
            ffmpeg.stdin.write(grid.tobytes())
        except Exception as e:
            print(f"❌ Error writing to FFmpeg: {e}")
            break

        time.sleep(1 / 25)  # 25 FPS

if __name__ == '__main__':
    print("🚀 Starting camera threads...")
    for cam_id, url in CAMERA_URLS.items():
        threading.Thread(target=process_camera, args=(cam_id, url), daemon=True).start()

    time.sleep(5)  # Give threads some time to start
    combine_and_stream()
