# Multi-Camera YOLOv8 Livestream to YouTube

This open-source project enables real-time object detection using YOLOv8 on multiple RTSP camera streams, overlays keypoints and IDs, and streams the combined output directly to YouTube in a 2x2 video grid.

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
| `cam1`    | `rtsp://...`                                  |
| `cam2`    | `rtsp://...`                                  |
| `cam4`    | `rtsp://...`                                  |
| `cam7`    | `rtsp://...`                                  |

> You can customize the camera sources in the `CAMERA_URLS` dictionary in `app.py`.

## Requirements

- Python 3.8+
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- OpenCV (`opencv-python`)
- FFmpeg (must be installed and available in your system path)
- NumPy

### Installation

```bash
pip install ultralytics opencv-python numpy
