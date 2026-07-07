# Truck Camera Counter Backend

This backend uses `package_label_best.pt` with OpenCV and Ultralytics YOLO to count packages from a webcam or MJPEG/IP camera stream.

## Settings file

Edit:

```text
backend/settings.json
```

Main settings:

```json
{
  "camera_source": "0",
  "confidence": 0.20,
  "image_size": 640,
  "min_area": 0.003,
  "max_aspect": 4.0,
  "line_y": 0.65,
  "direction": "any",
  "jpeg_quality": 82
}
```

For phone camera apps, set `camera_source` to the phone MJPEG URL:

```json
"camera_source": "http://PHONE_IP:8080/video"
```

Lower `image_size` for more FPS. Use `480` for speed, `640` for balance, and `960` for accuracy.

## Install

```powershell
cd "C:\Users\ILYESS\Desktop\react learning\my-app\backend"
python -m pip install -r requirements.txt
```

## Run

```powershell
python camera_counter_service.py
```

Open:

```text
http://localhost:8000/video
http://localhost:8000/snapshot
http://localhost:8000/metrics
http://localhost:8000/settings
```

Environment variables still override the settings file for temporary changes:

```powershell
$env:IMGSZ="480"
$env:CONF="0.25"
python camera_counter_service.py
```
## GPU mode

The backend reads these settings from `settings.json`:

```json
"device": "auto",
"half": true
```

`device: "auto"` uses CUDA when PyTorch can see your NVIDIA GPU. Check with:

```powershell
cd "C:\Users\ILYESS\Desktop\react learning\my-app\backend"
python check_gpu.py
```

If it says `cuda available: False`, your current PyTorch is CPU-only. Install a CUDA-enabled PyTorch build from the official PyTorch selector, then restart the backend.

For CUDA 12.8 wheels, the command is usually:

```powershell
python -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

After install, `python check_gpu.py` should show `cuda available: True` and the backend `/settings` endpoint should show `device: 0`.
## Run on your phone browser

Start the backend on the PC, then open this on your phone browser:

```text
http://YOUR_PC_IP:8000/mobile
```

Example:

```text
http://192.168.1.13:8000/mobile
```

Your phone and PC must be on the same Wi-Fi, and Windows Firewall must allow Python on port 8000.