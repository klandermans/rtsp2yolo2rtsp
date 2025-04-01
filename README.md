# Multi-Camera YOLOv8 Livestream to YouTube

This demo was developed as part of the Next Level Animal Science knowledge-based program at Wageningen University & Research. The goal of this project is to study dairy cow behavior using keypoint-based video analysis. While the YOLOv8-based keypoint tracking model was trained by fellow researchers, this implementation serves as a practical example of how to combine multiple IP camera streams, apply real-time detection and visualization, and broadcast the enriched video — even from behind a university firewall — directly to YouTube.

![image](https://github.com/user-attachments/assets/528af1a4-8e89-45ae-ad84-19a1795af8a0)


This example enables real-time object detection using YOLOv8 on multiple RTSP camera streams, overlays keypoints and IDs, and streams the combined output directly to YouTube in a 2x2 video grid.

## Features

- Connects to multiple RTSP camera feeds (tested with ONVIF IP cams)
- Runs YOLOv8 tracking (`.track()`) using [Ultralytics](https://github.com/ultralytics/ultralytics)
- Draws keypoints and tracked object IDs with color-coded markers
- Combines 4 live camera feeds into a single 1920x1080 video grid
- Streams directly to YouTube Live using FFmpeg
- Multi-threaded for real-time performance

## Example Setup

| Camera ID | RTSP URL                                      |
|-----------|-----------------------------------------------|
| `cam1`    | `rtsp://username:pasword@ip`                  |
| `cam2`    | `rtsp://...`                                  |
| `cam4`    | `rtsp://...`                                  |
| `cam7`    | `rtsp://...`                                  |

> You can customize the camera sources in the `CAMERA_URLS` dictionary in `app.py`.

## Requirements

- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- FFmpeg (must be installed and available in your system path)
- NumPy

### Installation

```bash
pip install ultralytics opencv-python numpy
```
