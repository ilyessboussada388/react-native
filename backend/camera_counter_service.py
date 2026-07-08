import argparse
import asyncio
import base64
import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response, StreamingResponse
from ultralytics import YOLO

torch.set_num_threads(max(1, int(os.getenv("TORCH_NUM_THREADS", "1"))))

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription
except Exception:
    RTCPeerConnection = None
    RTCSessionDescription = None


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    xyxy: BBox
    confidence: float
    class_name: str
    track_id: Optional[int] = None

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bottom_center(self) -> Point:
        x1, _, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, y2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def aspect(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        h = max(1.0, y2 - y1)
        return max(0.0, x2 - x1) / h


class SimpleIOUTracker:
    def __init__(self, iou_threshold: float = 0.18, max_missed: int = 20, max_center_distance: float = 0.18) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.max_center_distance = max_center_distance
        self.next_id = 1
        self.tracks: Dict[int, Tuple[BBox, int]] = {}

    def update(self, detections: List[Detection], frame_shape=None) -> List[Detection]:
        unmatched = set(self.tracks.keys())
        for det in detections:
            best_id = None
            best_score = 0.0
            for track_id in list(unmatched):
                bbox, _ = self.tracks[track_id]
                score = self._match_score(det.xyxy, bbox, frame_shape)
                if score > best_score:
                    best_score = score
                    best_id = track_id
            if best_id is not None and best_score >= self.iou_threshold:
                det.track_id = best_id
                self.tracks[best_id] = (det.xyxy, 0)
                unmatched.remove(best_id)
            else:
                det.track_id = self.next_id
                self.tracks[self.next_id] = (det.xyxy, 0)
                self.next_id += 1

        for track_id in unmatched:
            bbox, missed = self.tracks[track_id]
            missed += 1
            if missed > self.max_missed:
                del self.tracks[track_id]
            else:
                self.tracks[track_id] = (bbox, missed)
        return detections

    def _match_score(self, current: BBox, previous: BBox, frame_shape) -> float:
        iou = self._iou(current, previous)
        if frame_shape is None:
            return iou
        h, w = frame_shape[:2]
        cx1, cy1 = self._center(current)
        cx2, cy2 = self._center(previous)
        diagonal = max(1.0, (w * w + h * h) ** 0.5)
        distance_ratio = (((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5) / diagonal
        if distance_ratio > self.max_center_distance:
            return iou
        # Let nearby centers keep the same ID even when uploads skip enough frames that IoU drops.
        return max(iou, self.iou_threshold + (self.max_center_distance - distance_ratio))

    @staticmethod
    def _center(box: BBox) -> Point:
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        x1, y1 = max(ax1, bx1), max(ay1, by1)
        x2, y2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0


@dataclass
class LineCounter:
    line_y_ratio: float = 0.65
    direction: str = "any"
    min_track_frames: int = 1
    count_mode: str = "zone"
    duplicate_distance_ratio: float = 0.12
    duplicate_ttl_seconds: float = 2.5
    counted_ids: Set[int] = field(default_factory=set)
    previous_points: Dict[int, Point] = field(default_factory=dict)
    track_frames: Dict[int, int] = field(default_factory=dict)
    counted_points: List[Tuple[Point, float]] = field(default_factory=list)

    def update(self, detections: List[Detection], frame_shape) -> int:
        h, w = frame_shape[:2]
        start = (int(0.10 * w), int(self.line_y_ratio * h))
        end = (int(0.90 * w), int(self.line_y_ratio * h))
        now = time.time()
        self._expire_counted_points(now)
        for det in detections:
            if det.track_id is None:
                continue
            point = det.bottom_center
            self.track_frames[det.track_id] = self.track_frames.get(det.track_id, 0) + 1
            previous = self.previous_points.get(det.track_id)
            if previous and det.track_id not in self.counted_ids:
                if self.track_frames[det.track_id] >= self.min_track_frames and self._should_count(previous, point, start, end, frame_shape):
                    self.counted_ids.add(det.track_id)
                    self.counted_points.append((point, now))
            elif previous is None and det.track_id not in self.counted_ids:
                if self.track_frames[det.track_id] >= self.min_track_frames and self._zone_hit(point, start, end, frame_shape):
                    self.counted_ids.add(det.track_id)
                    self.counted_points.append((point, now))
            self.previous_points[det.track_id] = point
        return len(self.counted_ids)

    def draw(self, frame) -> None:
        h, w = frame.shape[:2]
        y = int(self.line_y_ratio * h)
        start, end = (int(0.10 * w), y), (int(0.90 * w), y)
        cv2.line(frame, start, end, (0, 255, 255), 3)
        cv2.putText(frame, f"Count line ({self.direction})", (start[0], max(25, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    def reset(self) -> None:
        self.counted_ids.clear()
        self.previous_points.clear()
        self.track_frames.clear()
        self.counted_points.clear()

    def _should_count(self, previous: Point, current: Point, start: Point, end: Point, frame_shape) -> bool:
        if self._is_recent_duplicate(current, frame_shape):
            return False
        if self.count_mode == "crossing":
            return self._crossed(previous, current, start, end)
        return self._crossed(previous, current, start, end) or self._zone_hit(current, start, end, frame_shape)

    def _zone_hit(self, current: Point, start: Point, end: Point, frame_shape) -> bool:
        if self._is_recent_duplicate(current, frame_shape):
            return False
        line_y = start[1]
        if self.direction == "up":
            return current[1] <= line_y
        return current[1] >= line_y

    def _is_recent_duplicate(self, current: Point, frame_shape) -> bool:
        h, w = frame_shape[:2]
        max_distance = self.duplicate_distance_ratio * max(1.0, (w * w + h * h) ** 0.5)
        for point, _ in self.counted_points:
            distance = ((current[0] - point[0]) ** 2 + (current[1] - point[1]) ** 2) ** 0.5
            if distance <= max_distance:
                return True
        return False

    def _expire_counted_points(self, now: float) -> None:
        self.counted_points = [(point, ts) for point, ts in self.counted_points if now - ts <= self.duplicate_ttl_seconds]

    def _crossed(self, previous: Point, current: Point, start: Point, end: Point) -> bool:
        dy = current[1] - previous[1]
        if abs(dy) < 2:
            return False
        prev_side = self._line_side(previous, start, end)
        curr_side = self._line_side(current, start, end)
        crossed = prev_side * curr_side < 0 or self._segments_intersect(previous, current, start, end)
        if not crossed:
            return False
        if self.direction == "down":
            return dy > 0
        if self.direction == "up":
            return dy < 0
        return True

    @staticmethod
    def _line_side(point: Point, start: Point, end: Point) -> int:
        value = (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])
        if abs(value) < 1e-6:
            return 0
        return 1 if value > 0 else -1

    @staticmethod
    def _segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
        def orient(p: Point, q: Point, r: Point) -> float:
            return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
        return orient(a, b, c) * orient(a, b, d) <= 0 and orient(c, d, a) * orient(c, d, b) <= 0


def load_settings(base_dir: Path) -> dict:
    defaults = {
        "model_path": "models/package_label_best.pt",
        "camera_source": "upload",
        "confidence": 0.20,
        "image_size": 640,
        "min_area": 0.003,
        "max_aspect": 4.0,
        "line_y": 0.65,
        "direction": "any",
        "count_mode": "zone",
        "tracker_iou": 0.18,
        "tracker_max_missed": 20,
        "tracker_max_center_distance": 0.18,
        "duplicate_distance": 0.12,
        "duplicate_ttl_seconds": 2.5,
        "max_upload_width": 640,
        "jpeg_quality": 82,
        "host": "0.0.0.0",
        "port": 8000,
        "device": "auto",
        "half": True,
    }
    settings_path = Path(os.getenv("SETTINGS_PATH", base_dir / "settings.json"))
    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8-sig") as f:
            defaults.update(json.load(f))
    return defaults


def setting_value(settings: dict, key: str, env_key: str, cast):
    raw = os.getenv(env_key, settings.get(key))
    return cast(raw)


class BoxCounterService:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent
        self.settings = load_settings(base_dir)
        raw_model_path = os.getenv("MODEL_PATH", self.settings.get("model_path", "models/package_label_best.pt"))
        self.model_path = Path(raw_model_path)
        if not self.model_path.is_absolute():
            self.model_path = base_dir / self.model_path
        self.source = str(os.getenv("CAMERA_SOURCE", self.settings.get("camera_source", "0")))
        self.conf = setting_value(self.settings, "confidence", "CONF", float)
        self.imgsz = setting_value(self.settings, "image_size", "IMGSZ", int)
        self.min_area = setting_value(self.settings, "min_area", "MIN_AREA", float)
        self.max_aspect = setting_value(self.settings, "max_aspect", "MAX_ASPECT", float)
        self.line_y = setting_value(self.settings, "line_y", "LINE_Y", float)
        self.direction = str(os.getenv("DIRECTION", self.settings.get("direction", "any")))
        self.count_mode = str(os.getenv("COUNT_MODE", self.settings.get("count_mode", "zone")))
        self.tracker_iou = setting_value(self.settings, "tracker_iou", "TRACKER_IOU", float)
        self.tracker_max_missed = setting_value(self.settings, "tracker_max_missed", "TRACKER_MAX_MISSED", int)
        self.tracker_max_center_distance = setting_value(self.settings, "tracker_max_center_distance", "TRACKER_MAX_CENTER_DISTANCE", float)
        self.duplicate_distance = setting_value(self.settings, "duplicate_distance", "DUPLICATE_DISTANCE", float)
        self.duplicate_ttl_seconds = setting_value(self.settings, "duplicate_ttl_seconds", "DUPLICATE_TTL_SECONDS", float)
        self.max_upload_width = setting_value(self.settings, "max_upload_width", "MAX_UPLOAD_WIDTH", int)
        self.jpeg_quality = setting_value(self.settings, "jpeg_quality", "JPEG_QUALITY", int)
        self.device = self._resolve_device(str(os.getenv("DEVICE", self.settings.get("device", "auto"))))
        self.half = setting_value(self.settings, "half", "HALF", lambda value: str(value).lower() in {"1", "true", "yes", "on"})

        self.model = YOLO(str(self.model_path))
        print(f"[counter] model={self.model_path}")
        print(f"[counter] device={self.device}, torch_cuda={torch.cuda.is_available()}")
        self.names = self.model.names
        self.tracker = SimpleIOUTracker(
            iou_threshold=self.tracker_iou,
            max_missed=self.tracker_max_missed,
            max_center_distance=self.tracker_max_center_distance,
        )
        self.counter = LineCounter(
            line_y_ratio=self.line_y,
            direction=self.direction,
            count_mode=self.count_mode,
            duplicate_distance_ratio=self.duplicate_distance,
            duplicate_ttl_seconds=self.duplicate_ttl_seconds,
        )
        self.lock = threading.Lock()
        self.latest_jpeg: Optional[bytes] = None
        self.metrics = {"count": 0, "visible": 0, "fps": 0.0, "model": self.model_path.name, "source": self.source}
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.upload_fps = 0.0
        self.last_upload_time: Optional[float] = None
        self.webrtc_fps = 0.0
        self.last_webrtc_time: Optional[float] = None

    def start(self) -> None:
        if self.source.lower() == "upload":
            return
        if self.running:
            return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def reset(self) -> None:
        self.counter.reset()
        self.tracker = SimpleIOUTracker(
            iou_threshold=self.tracker_iou,
            max_missed=self.tracker_max_missed,
            max_center_distance=self.tracker_max_center_distance,
        )
        with self.lock:
            self.metrics["count"] = 0

    def _resolve_device(self, requested: str):
        requested = requested.lower().strip()
        if requested in {"auto", "cuda", "0"}:
            if torch.cuda.is_available():
                return 0
            if requested in {"cuda", "0"}:
                print("[counter] CUDA was requested but this Python environment has CPU-only PyTorch. Falling back to CPU.")
        return "cpu"

    def _open_capture(self):
        source = int(self.source) if str(self.source).isdigit() else self.source
        return cv2.VideoCapture(source)

    def _detect(self, frame) -> Tuple[List[Detection], float]:
        start = time.perf_counter()
        results = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, half=self.half and self.device != "cpu", verbose=False)
        infer_ms = (time.perf_counter() - start) * 1000.0
        boxes = results[0].boxes
        detections: List[Detection] = []
        if boxes is None:
            return detections, infer_ms
        frame_area = frame.shape[0] * frame.shape[1]
        for box in boxes:
            cls_id = int(box.cls[0])
            name = str(self.names.get(cls_id, cls_id)).lower()
            if name != "package":
                continue
            det = Detection(
                xyxy=tuple(float(v) for v in box.xyxy[0].cpu().numpy()),
                confidence=float(box.conf[0]),
                class_name=name,
            )
            if det.area / frame_area < self.min_area:
                continue
            if det.aspect > self.max_aspect:
                continue
            detections.append(det)
        return detections, infer_ms

    def _draw(self, frame, detections: List[Detection], fps: float, count: int) -> None:
        for det in detections:
            x1, y1, x2, y2 = [int(v) for v in det.xyxy]
            label = f"ID {det.track_id} package {det.confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (40, 220, 70), 2)
            cv2.circle(frame, (int(det.bottom_center[0]), int(det.bottom_center[1])), 4, (40, 220, 70), -1)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 70), 2)
        self.counter.draw(frame)
        cv2.rectangle(frame, (10, 10), (310, 112), (0, 0, 0), -1)
        cv2.putText(frame, f"Boxes counted: {count}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, f"Visible packages: {len(detections)}", (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    def _process_frame(self, frame, fps: Optional[float] = None, return_image: bool = True) -> Tuple[Optional[bytes], dict]:
        process_start = time.perf_counter()
        detections, infer_ms = self._detect(frame)
        detections = self.tracker.update(detections, frame.shape)
        count = self.counter.update(detections, frame.shape)
        display_fps = fps if fps is not None else float(self.metrics.get("fps", 0.0))
        payload = None
        if return_image:
            self._draw(frame, detections, display_fps, count)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            if not ok:
                raise RuntimeError("Could not encode frame")
            payload = encoded.tobytes()
        detection_payload = []
        h, w = frame.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det.xyxy
            detection_payload.append({
                "id": det.track_id,
                "class_name": det.class_name,
                "confidence": round(det.confidence, 4),
                "x": max(0.0, min(1.0, x1 / max(1, w))),
                "y": max(0.0, min(1.0, y1 / max(1, h))),
                "width": max(0.0, min(1.0, (x2 - x1) / max(1, w))),
                "height": max(0.0, min(1.0, (y2 - y1) / max(1, h))),
            })
        metrics = {
            "count": count,
            "visible": len(detections),
            "fps": round(display_fps, 2),
            "model": self.model_path.name,
            "source": self.source,
            "confidence": self.conf,
            "image_size": self.imgsz,
            "line_y": self.line_y,
            "count_mode": self.count_mode,
            "device": str(self.device),
            "torch_cuda": torch.cuda.is_available(),
            "frame_width": w,
            "frame_height": h,
            "infer_ms": round(infer_ms, 1),
            "process_ms": round((time.perf_counter() - process_start) * 1000.0, 1),
            "detections": detection_payload,
        }
        with self.lock:
            if payload is not None:
                self.latest_jpeg = payload
            self.metrics = metrics
        return payload, metrics

    def process_uploaded_frame(self, image_b64: str, return_image: bool = True) -> Tuple[Optional[bytes], dict]:
        if image_b64.startswith("data:image"):
            image_b64 = image_b64.split(",", 1)[1]
        raw = base64.b64decode(image_b64)
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Invalid image")
        frame = self._resize_for_upload(frame)
        now = time.time()
        if self.last_upload_time is None:
            fps = 0.0
        else:
            dt = max(1e-6, now - self.last_upload_time)
            instant = 1.0 / dt
            fps = 0.85 * self.upload_fps + 0.15 * instant if self.upload_fps else instant
        self.last_upload_time = now
        self.upload_fps = fps
        return self._process_frame(frame, fps, return_image=return_image)

    def _resize_for_upload(self, frame):
        if self.max_upload_width <= 0:
            return frame
        h, w = frame.shape[:2]
        if w <= self.max_upload_width:
            return frame
        scale = self.max_upload_width / float(w)
        return cv2.resize(frame, (self.max_upload_width, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)


    def process_webrtc_frame(self, frame) -> Tuple[bytes, dict]:
        now = time.time()
        if self.last_webrtc_time is None:
            fps = 0.0
        else:
            dt = max(1e-6, now - self.last_webrtc_time)
            instant = 1.0 / dt
            fps = 0.9 * self.webrtc_fps + 0.1 * instant if self.webrtc_fps else instant
        self.last_webrtc_time = now
        self.webrtc_fps = fps
        return self._process_frame(frame, fps)

    def _loop(self) -> None:
        cap = self._open_capture()
        last = time.time()
        fps = 0.0
        while self.running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.3)
                cap.release()
                cap = self._open_capture()
                continue
            now = time.time()
            dt = max(1e-6, now - last)
            fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else 1.0 / dt
            last = now
            self._process_frame(frame, fps)
        cap.release()

    def frames(self):
        self.start()
        while True:
            with self.lock:
                frame = self.latest_jpeg
            if frame is None:
                time.sleep(0.05)
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            time.sleep(0.01)


service = BoxCounterService()
app = FastAPI(title="Truck package counter")
peer_connections = set()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.on_event("startup")
def startup_event():
    if service.source.lower() != "upload":
        service.start()


@app.post("/webrtc/offer")
async def webrtc_offer(request: Request):
    if RTCPeerConnection is None or RTCSessionDescription is None:
        return {
            "ok": False,
            "error": "aiortc is not installed. Run: python -m pip install aiortc",
        }

    params = await request.json()
    pc = RTCPeerConnection()
    peer_connections.add(pc)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        print(f"[webrtc] connection={pc.connectionState}")
        if pc.connectionState in {"failed", "closed", "disconnected"}:
            await pc.close()
            peer_connections.discard(pc)

    @pc.on("track")
    def on_track(track):
        print(f"[webrtc] track={track.kind}")
        if track.kind != "video":
            return

        async def consume_video():
            while True:
                try:
                    frame = await track.recv()
                    image = frame.to_ndarray(format="bgr24")
                    service.process_webrtc_frame(image)
                    await asyncio.sleep(0)
                except Exception as exc:
                    print(f"[webrtc] video stopped: {exc}")
                    break

        asyncio.create_task(consume_video())

    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)
    return {"sdp": pc.localDescription.sdp, "type": pc.localDescription.type}


@app.post("/upload-frame")
async def upload_frame(request: Request):
    try:
        payload = await request.json()
        image_b64 = payload.get("image", "")
        return_image = bool(payload.get("return_image", True))
        if not image_b64:
            return {"ok": False, "error": "Missing image"}
        frame, metrics = service.process_uploaded_frame(image_b64, return_image=return_image)
        response = {"ok": True, "metrics": metrics}
        if return_image and frame is not None:
            response["image"] = base64.b64encode(frame).decode("ascii")
        return response
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

@app.get("/metrics")
def metrics():
    with service.lock:
        return dict(service.metrics)


@app.post("/reset")
def reset():
    service.reset()
    return {"ok": True, "count": 0}




@app.get("/mobile", response_class=HTMLResponse)
def mobile_page():
    return """
<!doctype html>
<html>
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1" />
  <title>Truck Package Counter</title>
  <style>
    * { box-sizing: border-box; }
    body { margin: 0; font-family: Arial, sans-serif; background: #0b0f14; color: white; }
    header { padding: 14px 16px; background: #111827; position: sticky; top: 0; z-index: 2; }
    h1 { margin: 0; font-size: 20px; }
    .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px; padding: 10px; }
    .metric { background: #1f2937; border: 1px solid #374151; border-radius: 8px; padding: 10px; text-align: center; }
    .label { color: #9ca3af; font-size: 12px; }
    .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .video { width: 100%; background: #000; min-height: 260px; display: flex; align-items: center; justify-content: center; }
    img { width: 100%; height: auto; display: block; background: #000; }
    .controls { padding: 12px; display: grid; gap: 8px; }
    button { border: 0; border-radius: 8px; padding: 12px; font-size: 16px; font-weight: 700; background: #fbbf24; color: #111827; }
    .small { color: #9ca3af; font-size: 12px; word-break: break-word; }
  </style>
</head>
<body>
  <header><h1>Truck Package Counter</h1></header>
  <section class="metrics">
    <div class="metric"><div class="label">Count</div><div id="count" class="value">0</div></div>
    <div class="metric"><div class="label">Visible</div><div id="visible" class="value">0</div></div>
    <div class="metric"><div class="label">FPS</div><div id="fps" class="value">0.0</div></div>
  </section>
  <section class="video">
    <img id="frame" src="/snapshot" alt="Live camera" />
  </section>
  <section class="controls">
    <button onclick="resetCount()">Reset count</button>
    <div id="info" class="small">Connecting...</div>
  </section>
<script>
  const frame = document.getElementById('frame');
  const count = document.getElementById('count');
  const visible = document.getElementById('visible');
  const fps = document.getElementById('fps');
  const info = document.getElementById('info');

  function refreshFrame() {
    frame.src = '/snapshot?t=' + Date.now();
  }

  async function refreshMetrics() {
    try {
      const res = await fetch('/metrics?t=' + Date.now());
      const data = await res.json();
      count.textContent = data.count ?? 0;
      visible.textContent = data.visible ?? 0;
      fps.textContent = Number(data.fps ?? 0).toFixed(1);
      info.textContent = `Model: ${data.model} | Source: ${data.source} | Device: ${data.device ?? 'unknown'}`;
    } catch (err) {
      info.textContent = 'Waiting for backend...';
    }
  }

  async function resetCount() {
    await fetch('/reset', { method: 'POST' });
    await refreshMetrics();
  }

  setInterval(refreshFrame, 250);
  setInterval(refreshMetrics, 1000);
  refreshMetrics();
</script>
</body>
</html>
"""
@app.get("/snapshot")
def snapshot():
    service.start()
    with service.lock:
        frame = service.latest_jpeg
    if frame is None:
        return Response(status_code=503, content=b"No frame ready yet")
    return Response(content=frame, media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/video")
def video():
    return StreamingResponse(service.frames(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/settings")
def settings():
    return dict(service.settings, model_path=str(service.model_path), camera_source=service.source, confidence=service.conf, image_size=service.imgsz, line_y=service.line_y, count_mode=service.count_mode, device=str(service.device), torch_cuda=torch.cuda.is_available())


@app.get("/health")
def health():
    return {"ok": True, "model": service.model_path.name, "device": str(service.device)}


@app.get("/")
def root():
    return {"webrtc_offer": "/webrtc/offer", "upload_frame": "/upload-frame", "mobile": "/mobile", "video": "/video", "snapshot": "/snapshot", "metrics": "/metrics", "settings": "/settings", "reset": "/reset"}


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    settings = load_settings(base_dir)
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=str(settings.get("host", "0.0.0.0")))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", settings.get("port", 8000))))
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)
