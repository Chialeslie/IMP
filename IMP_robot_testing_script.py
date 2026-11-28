#!/usr/bin/env python3
import os
import sys

if os.getenv("INVOCATION_ID") or not os.getenv("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# --- Suppress Log Noise ---
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("LIBCAMERA_LOG_LEVELS", "3")
os.environ.setdefault("QT_LOGGING_RULES", "*.debug=false;qt.qpa.*=false")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# --- Cap Thread Contention (Must precede numpy / cv2 / ML runtime imports) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import time
import json
import cv2

# --- Cap OpenCV Internal Threads & Disable OpenCL ---
cv2.setNumThreads(1)
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass

import queue
import serial
import signal
import sqlite3
import numpy as np
import threading
import pyaudio
import pyttsx3
import subprocess
import pickle
import librosa
import math
import mediapipe as mp
from datetime import datetime
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from collections import deque, defaultdict, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vosk import Model, KaldiRecognizer
from typing import Any, Callable, Optional
from contextlib import contextmanager
from dataclasses import dataclass

# Hailo RT Imports
from hailo_platform import HEF, VDevice, HailoStreamInterface, ConfigureParams, InputVStreamParams, OutputVStreamParams, InputVStreams, OutputVStreams, FormatType

# Raspberry Pi AI Camera (IMX500) & Picamera2 Imports
from picamera2 import Picamera2
from picamera2.devices import IMX500

# --- Suppress ALSA / aplay Error Output ---
from ctypes import CFUNCTYPE, c_char_p, c_int, cdll

def py_error_handler(filename, line, function, err, fmt):
    pass

c_error_handler = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)(py_error_handler)
try:
    asound = cdll.LoadLibrary('libasound.so.2')
    asound.snd_lib_error_set_handler(c_error_handler)
except Exception:
    pass

# --- UNSTOPPABLE KERNEL EXIT ---
def force_emergency_exit(sig=None, frame=None):
    sys.stderr.write("\n[EMERGENCY STOP] Instantly terminating process...\n")
#    try:
#        ser = serial.Serial("/dev/serial/by-id/usb-Arduino_Giga_003400403233511135333535-if00", 115200, timeout=0.05)
#        ser.write(bytes([0x03]) + b"STOP\n")
#        ser.close()
#    except Exception:
#        pass
    
    # Force hard exit immediately, bypassing any hanging C-audio buffers
    os._exit(0)

signal.signal(signal.SIGINT, force_emergency_exit)
signal.signal(signal.SIGQUIT, force_emergency_exit)
signal.signal(signal.SIGTERM, force_emergency_exit)

# -----------------------------------------------------------------------------
# Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = "/home/chiagekliang/chiagekliangIMP"
MODEL_PATH = os.path.join(BASE_DIR, "model/model-en")
HAND_TASK_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
DB_PATH = os.path.join(BASE_DIR, "robot_faces.db")
HAILO10_FACE_HEF = os.path.join(BASE_DIR, "arcface_mobilefacenet_hailo10.hef")
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

IMX500_MODEL_PATH = "/usr/share/imx500-models/imx500_network_ssd_mobilenetv2_fpnlite_320x320_pp.rpk"
SERIAL_PORT = "/dev/serial/by-id/usb-Arduino_Giga_004B00313533511530363930-if00"
BAUD_RATE = 115200
ESP_AUDIO_PORT = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_A4:CB:8F:20:E0:3C-if00"
ESP_BAUD_RATE = 921600 # High baud rate required for uncompressed 16kHz audio streaming
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
CENTER_THRESHOLD = 60
CONFIDENCE_THRESHOLD = 0.2
TARGET_CLASS_ID = 0

HAND_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),
    (0,5), (5,6), (6,7), (7,8),
    (5,9), (9,10), (10,11), (11,12),
    (9,13), (13,14), (14,15), (15,16),
    (13,17), (17,18), (18,19), (19,20), (0,17)
]

# -----------------------------------------------------------------------------
# IMX500 AI Camera Thread
# -----------------------------------------------------------------------------
class IMX500Camera:
    def __init__(self, model_path, target_fps=30, fps_window_size=30, print_fps_interval=5.0):
        print("[TRACE] Initializing native Picamera2 + IMX500 context...")
        self.imx500 = IMX500(model_path)
        self.picam2 = Picamera2(self.imx500.camera_num)

        default_w = globals().get("FRAME_WIDTH", 640)
        default_h = globals().get("FRAME_HEIGHT", 480)
        self.frame_width = int(default_w)
        self.frame_height = int(default_h)

        config = self.picam2.create_video_configuration(
            main={"size": (self.frame_width, self.frame_height), "format": "RGB888"},
            controls={"FrameRate": target_fps},
            queue=True,
            buffer_count=4,
        )
        self.picam2.configure(config)
        self.picam2.start()

        # Sync actual configured dimensions if available
        main_size = config.get("main", {}).get("size")
        if main_size and len(main_size) == 2:
            self.frame_width, self.frame_height = int(main_size[0]), int(main_size[1])

        # FPS tracking state
        self._timestamps = deque(maxlen=max(2, fps_window_size))
        self._current_fps = 0.0
        self._print_fps_interval = print_fps_interval
        self._last_fps_log = time.monotonic()
        self.last_capture_ts = time.monotonic()

        print(
            f"[TRACE] IMX500 camera started ({self.frame_width}x{self.frame_height}, "
            f"target {target_fps} FPS)."
        )

    @property
    def fps(self):
        """Return the latest measured rolling FPS."""
        return self._current_fps

    def _update_fps(self, now):
        self._timestamps.append(now)
        if len(self._timestamps) >= 2:
            elapsed = self._timestamps[-1] - self._timestamps[0]
            if elapsed > 0:
                self._current_fps = (len(self._timestamps) - 1) / elapsed

        if self._print_fps_interval and (now - self._last_fps_log) >= self._print_fps_interval:
            print(f"[FPS] Measured IMX500 capture rate: {self._current_fps:.1f} FPS")
            self._last_fps_log = now

    def read(self):
        request = None
        try:
            request = self.picam2.capture_request()
            captured_at = time.monotonic()
            frame = request.make_array("main")
            metadata = request.get_metadata() or {}

            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
                # Picamera2 "RGB888" buffers are stored in BGR byte order; normalize to RGB
                if frame.ndim == 3 and frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                self.frame_height, self.frame_width = int(frame.shape[0]), int(frame.shape[1])

            self.last_capture_ts = captured_at
            self._update_fps(captured_at)
            return frame, metadata
        except Exception as e:
            print(f"[READ ERROR]: {e}")
            return None, {}
        finally:
            if request is not None:
                request.release()

    def parse_detections(self, metadata, confidence_threshold=0.5, target_class_id=0):
        detections = []
        w, h = self.frame_width, self.frame_height
        try:
            outputs = self.imx500.get_outputs(metadata)
            if outputs is not None:
                for detection in outputs:
                    flat_det = detection.flatten() if hasattr(detection, "flatten") else detection

                    if len(flat_det) >= 6:
                        ymin, xmin, ymax, xmax = (
                            float(flat_det[0]),
                            float(flat_det[1]),
                            float(flat_det[2]),
                            float(flat_det[3]),
                        )
                        conf = float(flat_det[4])
                        class_id = int(flat_det[5])

                        if not (0.0 <= conf <= 1.0):
                            continue

                        if conf >= confidence_threshold and (
                            target_class_id is None or class_id == target_class_id
                        ):
                            x1 = int(xmin * w)
                            y1 = int(ymin * h)
                            x2 = int(xmax * w)
                            y2 = int(ymax * h)

                            if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
                                detections.append({
                                    "bbox": [x1, y1, x2, y2],
                                    "confidence": conf,
                                })
        except Exception as e:
            print(f"[PARSER ERROR]: {e}")
        return detections

    def close(self):
        print("[TRACE] Stopping Picamera2 and releasing IMX500 resources...")
        try:
            self.picam2.stop()
            self.picam2.close()
            print("[TRACE] IMX500 camera successfully released.")
        except Exception as e:
            print(f"[ERROR] Error shutting down IMX500 camera: {e}")


# -----------------------------------------------------------------------------
# Robot Voice Synthesizer (TTS Helper)
# -----------------------------------------------------------------------------
class RobotSpeaker:
    def __init__(self, max_queue_size=2):
        self.speech_queue = queue.Queue(maxsize=max(1, max_queue_size))
        self.running = True
        self.is_speaking_now = False
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._speak_loop, name="robot-speaker", daemon=True)
        self.thread.start()

    def _speak_loop(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self.running:
                self.speech_queue.task_done()
                break

            with self._lock:
                self.is_speaking_now = True
            print(f"[ROBOT SPEAKS]: {text}")
            try:
                subprocess.run(
                    ["espeak-ng", str(text)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15.0,
                )
            except subprocess.TimeoutExpired:
                print("[SPEAKER WARN] espeak-ng timed out.")
            except Exception as e:
                print(f"[SPEAKER ERROR] Could not play audio: {e}")
            finally:
                with self._lock:
                    self.is_speaking_now = False
                self.speech_queue.task_done()

    def say(self, text):
        if not text or not self.running:
            return
        msg = str(text)
        try:
            self.speech_queue.put_nowait(msg)
        except queue.Full:
            # Drop oldest pending phrase so newer status/command speech is prioritized
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                pass
            try:
                self.speech_queue.put_nowait(msg)
            except queue.Full:
                pass

    def wait_until_done(self, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                speaking = self.is_speaking_now
            if self.speech_queue.unfinished_tasks == 0 and not speaking:
                return
            time.sleep(0.02)

    def stop(self):
        self.running = False
        # Drain any queued items so task_done() won't block anything waiting
        while True:
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)

# -----------------------------------------------------------------------------
# Terrain & Pavement Classifier
# -----------------------------------------------------------------------------
class TerrainClassifier:
    def __init__(
        self,
        brightness_thresh: float = 110.0,
        edge_density_thresh: float = 0.08,
        stair_edge_thresh: float = 0.18,
        max_roi_width: int = 320,
    ):
        self.current_terrain = "ROAD"
        self.brightness_thresh = float(brightness_thresh)
        self.edge_density_thresh = float(edge_density_thresh)
        self.stair_edge_thresh = float(stair_edge_thresh)
        self.max_roi_width = int(max_roi_width)

    def analyze_surface(
        self,
        frame: np.ndarray,
        display_frame: Optional[np.ndarray] = None,
        *,
        draw_overlay: bool = True,
        frame_is_rgb: bool = True,
    ) -> str:
        if frame is None or frame.size == 0 or len(frame.shape) < 2:
            return self.current_terrain

        h, w = int(frame.shape[0]), int(frame.shape[1])
        roi_y0 = int(h * 0.6)
        roi = frame[roi_y0:h, 0:w]
        if roi.size == 0:
            return self.current_terrain

        # Downscale wide ROIs before Canny to reduce Pi 5 CPU load
        roi_h, roi_w = int(roi.shape[0]), int(roi.shape[1])
        if self.max_roi_width > 0 and roi_w > self.max_roi_width:
            scale = self.max_roi_width / float(roi_w)
            target_h = max(1, int(round(roi_h * scale)))
            roi_proc = cv2.resize(roi, (self.max_roi_width, target_h), interpolation=cv2.INTER_AREA)
        else:
            roi_proc = roi

        if len(roi_proc.shape) == 3:
            gray_code = cv2.COLOR_RGB2GRAY if frame_is_rgb else cv2.COLOR_BGR2GRAY
            gray_roi = cv2.cvtColor(roi_proc, gray_code)
        else:
            gray_roi = roi_proc

        avg_brightness = float(np.mean(gray_roi))
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)

        # --- Enhanced Heuristic Classification ---
        if edge_density > self.stair_edge_thresh:
            self.current_terrain = "STAIRCASE"
        elif edge_density > self.edge_density_thresh:
            self.current_terrain = "CURB_EDGE"
        elif avg_brightness > self.brightness_thresh:
            self.current_terrain = "PAVEMENT"
        else:
            self.current_terrain = "ROAD"

        if draw_overlay and display_frame is not None:
            cv2.rectangle(display_frame, (0, roi_y0), (w, h), (255, 100, 0), 2)
            cv2.putText(
                display_frame,
                f"TERRAIN: {self.current_terrain}",
                (20, 190),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                display_frame,
                f"Bright: {int(avg_brightness)} | Edge: {edge_density:.2f}",
                (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

        return self.current_terrain

# -----------------------------------------------------------------------------
# Hailo-10H Face Embedder
# -----------------------------------------------------------------------------
def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


class Hailo10FaceEmbedder:
    """Computes L2-normalized face embeddings on Hailo-10H using UINT8 [0..255] RGB input."""

    def __init__(self, hef_path: str, infer_lock: Optional[threading.Lock] = None):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"Hailo HEF model not found: {hef_path}")

        self._infer_lock = infer_lock if infer_lock is not None else threading.Lock()
        self._closed = False

        params = VDevice.create_params()
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.UINT8)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        self.configured_infer_model = self.infer_model.configure()

        in_shape = self.infer_model.input().shape
        self.input_height = int(in_shape[0]) if len(in_shape) >= 2 else 112
        self.input_width = int(in_shape[1]) if len(in_shape) >= 2 else 112

        out_shape = self.infer_model.output().shape
        self._output_buffer = np.empty(out_shape, dtype=np.float32)

    def get_embedding(
        self, face_crop: np.ndarray, *, input_is_rgb: bool = True
    ) -> Optional[np.ndarray]:
        if self._closed:
            return None
        try:
            if face_crop is None or face_crop.size == 0:
                return None

            src_h, src_w = int(face_crop.shape[0]), int(face_crop.shape[1])
            interp = (
                cv2.INTER_AREA
                if (src_w > self.input_width or src_h > self.input_height)
                else cv2.INTER_LINEAR
            )
            resized = cv2.resize(
                face_crop,
                (self.input_width, self.input_height),
                interpolation=interp,
            )
            rgb_frame = (
                resized if input_is_rgb else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            )
            input_data = np.expand_dims(
                np.ascontiguousarray(rgb_frame, dtype=np.uint8), axis=0
            )

            with self._infer_lock:
                if self._closed or self.configured_infer_model is None:
                    return None
                bindings = self.configured_infer_model.create_bindings()
                bindings.input().set_buffer(input_data)
                bindings.output().set_buffer(self._output_buffer)
                self.configured_infer_model.run([bindings], timeout=5000)
                emb = l2_normalize(self._output_buffer)

            return emb
        except Exception as e:
            print(f"[LIVE EMBED ERROR]: {e}")
            return None

    def close(self) -> None:
        with self._infer_lock:
            if self._closed:
                return
            self._closed = True
            self.configured_infer_model = None
            self.infer_model = None
            if getattr(self, "target", None) is not None:
                release_fn = getattr(self.target, "release", None)
                if callable(release_fn):
                    try:
                        release_fn()
                    except Exception:
                        pass
                self.target = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

# -----------------------------------------------------------------------------
# SQLite Face Database Manager
# -----------------------------------------------------------------------------
def _normalize_l2(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1).copy()
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 0.0 else arr


class FaceDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self.known_embeddings: list[np.ndarray] = []
        self.known_names: list[str] = []
        self.known_roles: list[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._matrix_indices: list[int] = []
        self.reload_database()

    def _rebuild_index(self) -> None:
        if not self.known_embeddings:
            self._matrix = None
            self._matrix_indices = []
            return

        first_shape = self.known_embeddings[0].shape
        valid_vecs = []
        valid_indices = []
        for idx, vec in enumerate(self.known_embeddings):
            if vec.shape == first_shape and vec.size > 0:
                valid_vecs.append(vec)
                valid_indices.append(idx)

        if valid_vecs:
            self._matrix = np.vstack(valid_vecs).astype(np.float32, copy=False)
            self._matrix_indices = valid_indices
        else:
            self._matrix = None
            self._matrix_indices = []

    def reload_database(self) -> None:
        self.known_embeddings.clear()
        self.known_names.clear()
        self.known_roles.clear()
        self._matrix = None
        self._matrix_indices = []

        if not os.path.exists(self.db_path):
            print(
                f"[WARNING] Face database missing at {self.db_path}. "
                "Operating in generic person mode."
            )
            return

        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=5.0)
            cursor = conn.cursor()
            cursor.execute("SELECT name, role, embedding FROM users")
            rows = cursor.fetchall()

            loaded = 0
            for name, role, emb_bytes in rows:
                if not emb_bytes:
                    continue
                vec = _normalize_l2(np.frombuffer(emb_bytes, dtype=np.float32))
                if vec.size == 0:
                    continue
                self.known_embeddings.append(vec)
                self.known_names.append(str(name))
                self.known_roles.append(str(role))
                loaded += 1

            self._rebuild_index()
            print(f"[INFO] Loaded {loaded} profiles from face database.")
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
        finally:
            if conn is not None:
                conn.close()

    def match_face(
        self, query_vector: np.ndarray, max_distance: float = 0.65
    ) -> tuple[str, str, float]:
        if self._matrix is None or not self._matrix_indices or query_vector is None:
            return "Unknown", "none", 1.0

        q = _normalize_l2(query_vector)
        if q.size == 0 or q.shape[0] != self._matrix.shape[1]:
            return "Unknown", "none", 1.0

        similarities = self._matrix @ q
        distances = np.maximum(0.0, 1.0 - similarities)
        best_pos = int(np.argmin(distances))
        min_dist = float(distances[best_pos])
        best_idx = self._matrix_indices[best_pos]

        if min_dist <= max_distance:
            return (
                self.known_names[best_idx],
                self.known_roles[best_idx],
                min_dist,
            )

        return "Unknown", "none", min_dist
    
# -----------------------------------------------------------------------------
# Fall Detection using MediaPipe Pose Landmarker using key body joints like shoulder
# and hip combine with Multi-Criteria Geometric & Dynamic Rules
# -----------------------------------------------------------------------------
class PoseFallDetector:
    def __init__(
        self,
        frame_height=480,
        frame_width=640,
        horizontal_angle_deg=36.0,
        fall_drop_norm_per_sec=0.45,   # Normalized vertical drop/sec (relative to ROI height)
        impact_window_sec=1.2,          # How long after a rapid drop we still treat horizontal pose as a fall
        lying_duration_sec=1.5,         # Sustained horizontal duration to confirm fall
        cooldown_sec=4.0,
        min_visibility=0.45,
        max_crop_width=320,
    ):
        self.frame_height = int(frame_height)
        self.frame_width = int(frame_width)
        self.horizontal_angle_deg = float(horizontal_angle_deg)
        self.fall_drop_norm_per_sec = float(fall_drop_norm_per_sec)
        self.impact_window_sec = float(impact_window_sec)
        self.lying_duration_sec = float(lying_duration_sec)
        self.cooldown_sec = float(cooldown_sec)
        self.min_visibility = float(min_visibility)
        self.max_crop_width = int(max_crop_width)

        self._lock = threading.Lock()
        self.prev_norm_hip_y = None
        self.prev_time = None
        self.last_drop_time = None
        self.lying_start_time = None
        self.fall_detected = False
        self.fall_timestamp = 0.0
        self._last_err_log_time = 0.0
        self._closed = False

        pose_task_path = (
            os.path.join(BASE_DIR, "pose_landmarker.task")
            if "BASE_DIR" in globals()
            else "pose_landmarker.task"
        )
        if os.path.exists(pose_task_path):
            base_options = python.BaseOptions(model_asset_path=pose_task_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.4,
                min_pose_presence_confidence=0.4,
                min_tracking_confidence=0.4,
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        else:
            self.pose_detector = None
            print(
                f"[WARNING] Pose landmark task missing at {pose_task_path}. Fall detection disabled.",
                flush=True,
            )

    @staticmethod
    def _pick_joint(lm_left, lm_right, min_vis):
        left_ok = getattr(lm_left, "visibility", 1.0) >= min_vis
        right_ok = getattr(lm_right, "visibility", 1.0) >= min_vis
        if left_ok and right_ok:
            return (lm_left.x + lm_right.x) * 0.5, (lm_left.y + lm_right.y) * 0.5, True
        if left_ok:
            return lm_left.x, lm_left.y, True
        if right_ok:
            return lm_right.x, lm_right.y, True
        return None, None, False

    @staticmethod
    def _expand_and_clamp_bbox(bbox, w_img, h_img, pad_x=0.12, pad_y=0.15, min_size=32):
        """Expands a person bbox with margin and clamps to image bounds; returns None if invalid."""
        if bbox is None or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            return None
        if not (
            math.isfinite(x1)
            and math.isfinite(y1)
            and math.isfinite(x2)
            and math.isfinite(y2)
        ):
            return None

        bw = x2 - x1
        bh = y2 - y1
        if bw < min_size or bh < min_size:
            return None

        px = bw * pad_x
        py = bh * pad_y
        xmin = max(0, min(w_img - 1, int(round(x1 - px))))
        ymin = max(0, min(h_img - 1, int(round(y1 - py))))
        xmax = max(xmin + 1, min(w_img, int(round(x2 + px))))
        ymax = max(ymin + 1, min(h_img, int(round(y2 + py))))

        if (xmax - xmin) < min_size or (ymax - ymin) < min_size:
            return None
        return xmin, ymin, xmax, ymax

    @staticmethod
    def _is_lying_from_hip_ankle(
        hx_norm, hy_norm, landmarks, min_vis=0.45, max_norm_dy=0.14
    ):
        """
        Returns:
        - True: ankles are visible and nearly level vertically with hips (lying posture)
        - False: ankles are clearly below hips (standing/crouching posture)
        - None: ankles not reliably visible (fall back to torso/motion rules)
        """
        if len(landmarks) <= 28:
            return None

        ax_norm, ay_norm, a_ok = PoseFallDetector._pick_joint(
            landmarks[27], landmarks[28], min_vis
        )
        if not a_ok:
            return None

        # In image coordinates, y increases downward.
        vertical_drop_to_ankle = ay_norm - hy_norm
        horizontal_span = abs(ax_norm - hx_norm)

        # If ankles are well below hips, person is upright/crouching rather than flat
        if vertical_drop_to_ankle > max_norm_dy and vertical_drop_to_ankle > horizontal_span * 0.6:
            return False

        # If hips and ankles are nearly at the same vertical height (or horizontal span dominates)
        if abs(vertical_drop_to_ankle) <= max_norm_dy or horizontal_span > vertical_drop_to_ankle * 1.5:
            return True

        return None

    def _check_cooldown_unlocked(self, now):
        if self.fall_detected:
            if (now - self.fall_timestamp) < self.cooldown_sec:
                return True, "FALL DETECTED!"
            self.fall_detected = False
        return False, "NORMAL"

    def _log_error_unlocked(self, now, msg):
        if (now - self._last_err_log_time) > 2.0:
            print(f"[FALL ERROR] {msg}", flush=True)
            self._last_err_log_time = now

    def process_crop_or_frame(
        self, frame, display_frame, bbox=None, is_rgb=True, *, draw_overlay=True
    ):
        now = time.monotonic()

        with self._lock:
            if self._closed or not self.pose_detector or frame is None or getattr(frame, "size", 0) == 0:
                return self._check_cooldown_unlocked(now)

            h_img, w_img = frame.shape[:2]
            expanded = (
                self._expand_and_clamp_bbox(bbox, w_img, h_img)
                if bbox is not None
                else None
            )
            if expanded is not None:
                xmin, ymin, xmax, ymax = expanded
                person_roi = frame[ymin:ymax, xmin:xmax]
            else:
                xmin, ymin, xmax, ymax = 0, 0, w_img, h_img
                person_roi = frame

            if person_roi is None or person_roi.size == 0:
                return self._check_cooldown_unlocked(now)

            roi_h, roi_w = person_roi.shape[:2]
            if self.max_crop_width > 0 and roi_w > self.max_crop_width:
                scale = self.max_crop_width / float(roi_w)
                det_w = self.max_crop_width
                det_h = max(1, int(round(roi_h * scale)))
                det_roi = cv2.resize(person_roi, (det_w, det_h), interpolation=cv2.INTER_AREA)
            else:
                det_roi = person_roi

            status_text = "NORMAL"

            try:
                rgb_roi = det_roi if is_rgb else cv2.cvtColor(det_roi, cv2.COLOR_BGR2RGB)
                rgb_contig = np.ascontiguousarray(rgb_roi, dtype=np.uint8)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)
                results = self.pose_detector.detect(mp_image)

                pose_list = getattr(results, "pose_landmarks", None) or []
                if pose_list:
                    landmarks = pose_list[0]

                    sx_norm, sy_norm, s_ok = self._pick_joint(
                        landmarks[11], landmarks[12], self.min_visibility
                    )
                    hx_norm, hy_norm, h_ok = self._pick_joint(
                        landmarks[23], landmarks[24], self.min_visibility
                    )

                    if s_ok and h_ok:
                        shoulder_x = xmin + int(sx_norm * roi_w)
                        shoulder_y = ymin + int(sy_norm * roi_h)
                        hip_x = xmin + int(hx_norm * roi_w)
                        hip_y = ymin + int(hy_norm * roi_h)

                        if draw_overlay and display_frame is not None:
                            # RGB colors (converted to BGR before cv2.imshow / cv2.imwrite)
                            cv2.circle(display_frame, (shoulder_x, shoulder_y), 4, (255, 0, 0), -1)
                            cv2.circle(display_frame, (hip_x, hip_y), 4, (0, 0, 255), -1)
                            cv2.line(
                                display_frame,
                                (shoulder_x, shoulder_y),
                                (hip_x, hip_y),
                                (255, 255, 0),
                                2,
                            )

                        # 1. Torso angle relative to horizontal (0 deg = horizontal, 90 deg = vertical)
                        dx = hip_x - shoulder_x
                        dy = hip_y - shoulder_y
                        torso_angle = math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))
                        is_horizontal = torso_angle < self.horizontal_angle_deg

                        # 2. Scale-normalized downward hip velocity (in ROI-heights per second)
                        rapid_descent = False
                        if self.prev_norm_hip_y is not None and self.prev_time is not None:
                            dt = now - self.prev_time
                            if 1e-3 < dt < 1.0:
                                drop_rate = (hy_norm - self.prev_norm_hip_y) / dt
                                if drop_rate > self.fall_drop_norm_per_sec:
                                    rapid_descent = True
                                    self.last_drop_time = now

                        self.prev_norm_hip_y = hy_norm
                        self.prev_time = now

                        recent_drop = (
                            self.last_drop_time is not None
                            and (now - self.last_drop_time) <= self.impact_window_sec
                        )

                        # 3. Aspect ratio check (guard against tiny/degenerate boxes)
                        box_flat = False
                        if bbox is not None:
                            box_w = xmax - xmin
                            box_h = ymax - ymin
                            if box_h > 20 and (box_w / box_h) >= 1.15:
                                box_flat = True

                        # 4. Ankle-to-hip posture check
                        ankle_lying = self._is_lying_from_hip_ankle(
                            hx_norm, hy_norm, landmarks, min_vis=self.min_visibility
                        )

                        # Suppress false triggers when ankles clearly show upright/crouching posture
                        # without a fast downward drop.
                        upright_ankles_blocking = (ankle_lying is False) and not rapid_descent

                        # Require horizontal torso plus lying ankles, recent drop, or wide/flat box
                        posture_is_lying = is_horizontal and (
                            (ankle_lying is True) or rapid_descent or recent_drop or box_flat
                        )

                        is_fall_candidate = posture_is_lying and not upright_ankles_blocking

                        if is_fall_candidate:
                            if self.lying_start_time is None:
                                self.lying_start_time = now
                            lying_duration = now - self.lying_start_time
                            if lying_duration >= self.lying_duration_sec:
                                self.fall_detected = True
                                self.fall_timestamp = now
                                self.lying_start_time = None
                                self.last_drop_time = None
                                return True, "FALL DETECTED!"
                            status_text = f"CHECKING FALL ({lying_duration:.1f}s)..."
                        else:
                            self.lying_start_time = None
                    else:
                        self.lying_start_time = None
                        self.prev_norm_hip_y = None
                        self.prev_time = None
                else:
                    self.lying_start_time = None
                    self.prev_norm_hip_y = None
                    self.prev_time = None

            except Exception as e:
                self._log_error_unlocked(now, str(e))

            in_cooldown, cooldown_status = self._check_cooldown_unlocked(now)
            if in_cooldown:
                return True, cooldown_status
            return False, status_text

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.pose_detector is not None:
                try:
                    self.pose_detector.close()
                except Exception:
                    pass
                self.pose_detector = None
                
# -----------------------------------------------------------------------------
# MediaPipe Hand Gesture Recognizer
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(
        self,
        task_path=HAND_TASK_PATH,
        max_proc_width: int = 320,
        debug_log: bool = False,
    ):
        if not os.path.exists(task_path):
            raise FileNotFoundError(f"[GESTURE ERROR] Task missing: {task_path}")
        base_options = python.BaseOptions(model_asset_path=task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=1,
            min_hand_detection_confidence=0.60,
            min_tracking_confidence=0.60,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.max_proc_width = int(max_proc_width)
        self.debug_log = bool(debug_log)

        self._lock = threading.Lock()
        self._closed = False
        self.last_wrist_y = None
        self.last_landmarks = None

    def process_frame(
        self,
        frame,
        display_frame=None,
        *,
        is_rgb: bool = True,
        draw_overlay: bool = True,
    ):
        if frame is None or frame.size == 0:
            return None, "Unknown"

        with self._lock:
            if self._closed or self.detector is None:
                return None, "Unknown"

            src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
            if self.max_proc_width > 0 and src_w > self.max_proc_width:
                scale = self.max_proc_width / float(src_w)
                proc_w = self.max_proc_width
                proc_h = max(1, int(round(src_h * scale)))
                interp = cv2.INTER_AREA
                proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=interp)
            else:
                proc_frame = frame

            rgb_frame = (
                proc_frame if is_rgb else cv2.cvtColor(proc_frame, cv2.COLOR_BGR2RGB)
            )
            rgb_contig = np.ascontiguousarray(rgb_frame, dtype=np.uint8)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_contig)
            results = self.detector.detect(mp_image)

            gesture_cmd = None
            hand_side = "Unknown"
            self.last_wrist_y = None
            self.last_landmarks = None

            if results.hand_landmarks:
                for idx, hand_landmarks in enumerate(results.hand_landmarks):
                    if results.handedness and len(results.handedness) > idx:
                        hand_side = results.handedness[idx][0].category_name

                    lm = hand_landmarks
                    wrist_y = lm[0].y
                    self.last_wrist_y = wrist_y
                    self.last_landmarks = lm

                    if draw_overlay and display_frame is not None:
                        disp_h = (
                            int(display_frame.shape[0])
                            if len(display_frame.shape) >= 2
                            else globals().get("FRAME_HEIGHT", 480)
                        )
                        disp_w = (
                            int(display_frame.shape[1])
                            if len(display_frame.shape) >= 2
                            else globals().get("FRAME_WIDTH", 640)
                        )
                        for connection in HAND_CONNECTIONS:
                            p1 = (
                                int(lm[connection[0]].x * disp_w),
                                int(lm[connection[0]].y * disp_h),
                            )
                            p2 = (
                                int(lm[connection[1]].x * disp_w),
                                int(lm[connection[1]].y * disp_h),
                            )
                            cv2.line(display_frame, p1, p2, (0, 255, 255), 2)

                    index_open = lm[8].y < lm[6].y
                    middle_open = lm[12].y < lm[10].y
                    ring_open = lm[16].y < lm[14].y
                    pinky_open = lm[20].y < lm[18].y

                    palm_scale = max(
                        1e-5,
                        ((lm[9].x - lm[0].x) ** 2 + (lm[9].y - lm[0].y) ** 2) ** 0.5,
                    )
                    pinch_ratio = (
                        ((lm[4].x - lm[8].x) ** 2 + (lm[4].y - lm[8].y) ** 2) ** 0.5
                    ) / palm_scale

                    if self.debug_log:
                        print(
                            f"[GESTURE DEBUG] Hand: {hand_side} | Wrist Y: {wrist_y:.2f} | "
                            f"Index Up: {index_open}"
                        )
                        print(
                            f"[DEBUG GESTURE] index_open: {index_open} | "
                            f"middle_open: {middle_open} | ring_open: {ring_open} | "
                            f"pinky_open: {pinky_open}"
                        )

                    # ONLY evaluate gestures if the hand is raised above 0.50 height gate
                    if wrist_y < 0.50:
                        # 1. FORWARD: ONLY Index finger open
                        if index_open and not middle_open and not ring_open and not pinky_open:
                            gesture_cmd = "FORWARD"
                            break

                        # 2. BACKWARD: Index AND Middle open, others closed
                        elif index_open and middle_open and not ring_open and not pinky_open:
                            gesture_cmd = "BACKWARD"
                            break

                        # 3. OK Sign (Take Photo): Thumb & index pinched relative to palm size, other 3 fingers open
                        elif pinch_ratio < 0.35 and middle_open and ring_open and pinky_open:
                            gesture_cmd = "TAKE_PHOTO"
                            break

                        # 4. Thumb Left / Right: Four fingers closed, thumb extended horizontally
                        elif not index_open and not middle_open and not ring_open and not pinky_open:
                            thumb_dx = lm[4].x - lm[0].x
                            if abs(thumb_dx) > 0.12:
                                gesture_cmd = "LEFT" if thumb_dx < 0 else "RIGHT"
                                break

                        # 5. STOP: All fingers open
                        elif index_open and middle_open and ring_open and pinky_open:
                            gesture_cmd = "STOP"
                            break

                    break

            return gesture_cmd, hand_side

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self.detector is not None:
                try:
                    self.detector.close()
                except Exception:
                    pass
                self.detector = None
    
# -----------------------------------------------------------------------------
# Offline Voice Recognizer (Vosk + PyAudio / ESP32 Serial)
# -----------------------------------------------------------------------------
class OfflineVoiceRecognizer:

    def __init__(
        self,
        command_queue,
        model_path,
        speaker=None,
        vip_ref_paths=None,
        speaker_threshold: float = 0.48,
        short_word_threshold: float = 0.45,
        post_speech_hold: float = 1.4,
    ):
        self.command_queue = command_queue
        self.speaker = speaker
        self.is_running = True

        self.speaker_threshold = float(speaker_threshold)
        self.short_word_threshold = float(short_word_threshold)
        self.post_speech_hold = float(post_speech_hold)

        # Audio configuration (Matches ESP32-C3 firmware sample rate of 16kHz)
        self.target_rate = 16000
        self.audio_buffer = deque(maxlen=250)

        self._lock = threading.Lock()
        self._speaking_active = False
        self._mute_until = 0.0
        self._needs_flush = False
        self._last_reject_tts_time = 0.0
        self.speaker_embeddings = {}
        self.voice_encoder = None

        print("[VOICE] Loading offline Vosk model...")
        self.model = Model(model_path)
        self.grammar_json = json.dumps([
            "forward", "come", "ahead",
            "backward", "back", "reverse", "retreat", "go",
            "stop", "halt", "wait",
            "left", "right",
            "auto", "track", "follow",
            "chia", "eloy",
            "[unk]",
        ])
        self.recognizer = KaldiRecognizer(self.model, self.target_rate, self.grammar_json)

        self.command_map = {
            "forward": ("FORWARD", "Moving forward."),
            "come": ("FORWARD", "Moving forward."),
            "ahead": ("FORWARD", "Moving forward."),
            "backward": ("BACKWARD", "Reversing."),
            "back": ("BACKWARD", "Reversing."),
            "reverse": ("BACKWARD", "Reversing."),
            "retreat": ("BACKWARD", "Reversing."),
            "go": ("FORWARD", "Moving forward."),
            "stop": ("STOP", "Stopping robot."),
            "halt": ("STOP", "Stopping robot."),
            "wait": ("STOP", "Stopping robot."),
            "left": ("LEFT", "Turning left."),
            "right": ("RIGHT", "Turning right."),
            "auto": ("AUTO", "Resuming auto mode."),
            "track": ("AUTO", "Resuming auto mode."),
            "follow": ("AUTO", "Resuming auto mode."),
            "chia": ("CHIA", "Switching target to Chia."),
            "eloy": ("ELOY", "Switching target to Eloy."),
        }

        # --- ESP32-C3 SERIAL STREAM SETUP ---
        self.esp_audio_port = globals().get(
            "ESP_AUDIO_PORT",
            "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_A4:CB:8F:20:E0:3C-if00",
        )
        self.esp_baud_rate = int(globals().get("ESP_BAUD_RATE", 921600))

        try:
            self.esp_audio_serial = serial.Serial(
                self.esp_audio_port, self.esp_baud_rate, timeout=0.01
            )
            print(f"[VOICE] Connected to ESP32-C3 Audio Node on {self.esp_audio_port}")
        except Exception as e:
            print(f"[ERROR] Failed to open ESP32 audio stream port: {e}")
            self.esp_audio_serial = None

        # Voice encoder setup for verification
        self._init_voice_encoder(vip_ref_paths)

        # Start the background listening thread
        self.voice_thread = threading.Thread(
            target=self._listen_loop, name="voice-recognizer", daemon=True
        )
        self.voice_thread.start()
        print("[INIT] ESP32 Voice listener background thread explicitly triggered.")

    def send_text_command_to_arduino(self, source, command):
        """
        Sends a text command packet to the Arduino Giga using header 0x03.
        Format: [0x03] + "SOURCE:COMMAND\n"
        Example: [0x03] + "VOICE:STOP\n"
        """
        try:
            ser = getattr(self, "esp_audio_serial", None)
            if ser and ser.is_open:
                payload = bytes([0x03]) + f"{source}:{command}\n".encode("utf-8")
                with self._lock:
                    ser.write(payload)
        except Exception as e:
            print(f"[SERIAL TEXT ERROR]: {e}")

    def set_speaking(self, speaking: bool, cooldown: float | None = None):
        """Non-blocking TTS mute gate with configurable post-speech reverberation hold."""
        default_hold = min(self.post_speech_hold, 0.45)
        hold = default_hold if cooldown is None else max(0.0, float(cooldown))
        now = time.monotonic()
        with self._lock:
            if speaking:
                self._speaking_active = True
                self._mute_until = max(self._mute_until, now + hold)
                self.audio_buffer.clear()
            else:
                self._speaking_active = False
                # Only extend _mute_until if no longer actively speaking
                self._mute_until = max(self._mute_until, now + hold)
                self._needs_flush = True
                self.audio_buffer.clear()

    def _is_muted_and_should_flush(self, now: float) -> tuple[bool, bool]:
        with self._lock:
            speaker_busy = bool(getattr(self.speaker, "is_speaking_now", False))
            if speaker_busy:
                self._speaking_active = True
                # Keep a short floor while actively speaking; release promptly once done
                self._mute_until = max(self._mute_until, now + 0.15)
            elif self._speaking_active:
                self._speaking_active = False

            muted = speaker_busy or (now < self._mute_until)
            flush = False
            if not muted and self._needs_flush:
                flush = True
                self._needs_flush = False
                self.audio_buffer.clear()
            return muted, flush

    def _track_async_speech_end(self, hold: float):
        """Waits for RobotSpeaker playback to finish before starting short cooldown."""
        try:
            if hasattr(self.speaker, "wait_until_done"):
                self.speaker.wait_until_done(timeout=6.0)
        finally:
            self.set_speaking(False, cooldown=hold)

    def say_muted(self, text: str, extra_hold: float = 0.0, cooldown: float | None = None):
        """Speaks via self.speaker while keeping Vosk muted only during playback + short tail."""
        if not self.speaker or not text:
            return
        base_hold = min(self.post_speech_hold, 0.45) if cooldown is None else max(0.0, float(cooldown))
        total_hold = max(0.0, base_hold + float(extra_hold))
        self.set_speaking(True, cooldown=total_hold)
        try:
            self.speaker.say(text)
            if hasattr(self.speaker, "wait_until_done") and hasattr(self.speaker, "speech_queue"):
                watcher = threading.Thread(
                    target=self._track_async_speech_end,
                    args=(total_hold,),
                    name="tts-mute-watcher",
                    daemon=True,
                )
                watcher.start()
                return
        except Exception:
            self.set_speaking(False, cooldown=total_hold)
            raise
        self.set_speaking(False, cooldown=total_hold)

    def _init_voice_encoder(self, vip_ref_paths=None):
        """Loads multi-speaker Resemblyzer embeddings from pickle."""
        model_path = (
            os.path.join(BASE_DIR, "authorized_speakers.pkl")
            if "BASE_DIR" in globals() and not os.path.exists("authorized_speakers.pkl")
            else "authorized_speakers.pkl"
        )
        try:
            with open(model_path, "rb") as f:
                loaded = pickle.load(f)

            if isinstance(loaded, dict) and loaded:
                self.speaker_embeddings = loaded
                print(
                    f"[VOICE] Loaded authorized speaker embeddings successfully: "
                    f"{list(self.speaker_embeddings.keys())}"
                )
            else:
                print(f"[VOICE WARNING] '{model_path}' loaded, but it is empty.")
                self.speaker_embeddings = {}

        except Exception as e:
            print(f"[VOICE WARNING] Could not load authorized speaker embeddings: {e}")
            self.speaker_embeddings = {}

        # Initialize Resemblyzer Voice Encoder
        try:
            try:
                import torch
                torch.set_num_threads(1)
                torch.set_num_interop_threads(1)
            except Exception:
                pass
            from resemblyzer import VoiceEncoder
            self.voice_encoder = VoiceEncoder()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Resemblyzer VoiceEncoder: {e}")
            self.voice_encoder = None

    def _verify_speaker(self, raw_audio, threshold=None, min_samples=4800):
        """Verifies incoming 16kHz 16-bit mono PCM audio against enrolled speaker embeddings."""
        # 16-bit PCM -> float32 normalized to [-1.0, 1.0]
        audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

        # Trim leading/trailing silence
        audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
        
        eff_threshold = self.speaker_threshold if threshold is None else float(threshold)
        if not self.speaker_embeddings or not self.voice_encoder:
            print("[VOICE WARNING] Speaker verification bypassed (no models or encoder loaded).")
            return True, "unknown"

        try:
            if not raw_audio or len(raw_audio) < 2:
                return False, "UNKNOWN"

            # 16-bit PCM -> float32 normalized to [-1.0, 1.0]
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0

            # Trim leading/trailing silence
            audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)

            if len(audio_trimmed) < min_samples:
                if len(audio_np) < min_samples:
                    print(
                        f"[VOICE DEBUG] Audio too short ({len(audio_trimmed)}/{min_samples} samples) "
                        f"for reliable speaker embedding."
                    )
                    return False, "UNKNOWN"
                audio_trimmed = audio_np

            test_embedding = self.voice_encoder.embed_utterance(audio_trimmed)
            test_norm = float(np.linalg.norm(test_embedding))
            if test_norm == 0.0:
                return False, "UNKNOWN"

            best_score = -1.0
            matched_speaker = "UNKNOWN"

            for name, ref_embedding in self.speaker_embeddings.items():
                ref_arr = np.asarray(ref_embedding, dtype=np.float32).reshape(-1)
                ref_norm = float(np.linalg.norm(ref_arr))
                if ref_norm == 0.0 or ref_arr.shape != test_embedding.shape:
                    continue
                similarity = float(np.dot(test_embedding, ref_arr) / (test_norm * ref_norm))
                if similarity > best_score:
                    best_score = similarity
                    matched_speaker = name

            print(
                f"[VOICE DEBUG] Best Match: '{matched_speaker}' | "
                f"Cosine Similarity: {best_score:.4f} (Threshold: {eff_threshold:.2f})"
            )

            if best_score >= eff_threshold:
                return True, matched_speaker

            return False, matched_speaker

        except Exception as e:
            print(f"[VOICE ERROR] Error during speaker verification: {e}")
            return False, "UNKNOWN"

    def _select_command(self, words: list[str]):
        # Prefer explicit target/mode switches if present in a multi-word phrase
        for priority_word in ("chia", "eloy", "auto", "track", "follow", "stop", "halt", "wait"):
            if priority_word in words and priority_word in self.command_map:
                return self.command_map[priority_word]
        return next((self.command_map[w] for w in words if w in self.command_map), None)

    def _listen_loop(self):
        print("[VOICE DEBUG] _listen_loop thread started...", flush=True)
        try:
            if not self.esp_audio_serial:
                print("[VOICE ERROR] ESP32 serial stream is not active. Exiting loop.", flush=True)
                return

            print("[VOICE] Streaming raw PCM bytes from ESP32-C3 at 16kHz...", flush=True)

            def register_cmd(cmd_string, speech_reply):
                print(f"[VOICE BROADCAST]: {speech_reply}", flush=True)
                self.command_queue.put(cmd_string)
                # Short tail after speech finishes so follow-up commands aren't blocked
                reply_cooldown = min(self.post_speech_hold, 0.35)
                self.say_muted(speech_reply, cooldown=reply_cooldown)

            chunk_counter = 0
            rx_bytes_window = 0
            rx_rms_sum = 0.0
            rx_rms_samples = 0
            last_rx_log_mono = time.monotonic()

            while self.is_running:
                try:
                    data = self.esp_audio_serial.read(512)
                except Exception as read_err:
                    if not self.is_running:
                        break
                    print(f"[VOICE SERIAL READ ERROR]: {read_err}", flush=True)
                    time.sleep(0.02)
                    continue

                if not data or len(data) < 2:
                    now_idle = time.monotonic()
                    if (now_idle - last_rx_log_mono) >= 2.0:
                        print(
                            f"[VOICE RX] bytes_2s={rx_bytes_window} | avg_rms=0.0 | muted=n/a",
                            flush=True,
                        )
                        rx_bytes_window = 0
                        rx_rms_sum = 0.0
                        rx_rms_samples = 0
                        last_rx_log_mono = now_idle
                    continue

                if len(data) % 2 != 0:
                    data = data[:-1]
                if not data:
                    continue

                audio_array = np.frombuffer(data, dtype=np.int16)
                rms = (
                    float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2)))
                    if audio_array.size > 0
                    else 0.0
                )

                rx_bytes_window += len(data)
                rx_rms_sum += rms
                rx_rms_samples += 1
                chunk_counter += 1

                now_mono = time.monotonic()
                muted, should_flush = self._is_muted_and_should_flush(now_mono)

                if (now_mono - last_rx_log_mono) >= 2.0:
                    avg_rms = (rx_rms_sum / rx_rms_samples) if rx_rms_samples > 0 else 0.0
                    print(
                        f"[VOICE RX] bytes_2s={rx_bytes_window} | avg_rms={avg_rms:.1f} | muted={muted}",
                        flush=True,
                    )
                    rx_bytes_window = 0
                    rx_rms_sum = 0.0
                    rx_rms_samples = 0
                    last_rx_log_mono = now_mono

                partial_str = ""
                is_final = False
                result = None
                utterance_pcm = b""

                with self._lock:
                    if should_flush:
                        self.audio_buffer.clear()

                    if muted:
                        continue

                    self.audio_buffer.append(data)
                    is_final = bool(self.recognizer.AcceptWaveform(data))
                    if is_final:
                        result = json.loads(self.recognizer.Result())
                        utterance_pcm = b"".join(self.audio_buffer)
                        self.audio_buffer.clear()
                    elif chunk_counter % 8 == 0:
                        partial_str = (
                            json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
                        )

                if is_final and isinstance(result, dict):
                    text = str(result.get("text", "")).strip().lower()
                    print(f"\n[VOSK RAW RESULT JSON]: {result}", flush=True)

                    if text:
                        print(f"\n[VOICE RECOGNIZED RAW TEXT]: '{text}'", flush=True)
                        words = text.split()
                        matched_entry = self._select_command(words)

                        if not matched_entry:
                            print(f"[VOICE IGNORED] '{text}' is not in command allowlist.", flush=True)
                            with self._lock:
                                self.recognizer.Reset()
                            continue

                        req_thresh = (
                            self.short_word_threshold if len(words) == 1 else self.speaker_threshold
                        )
                        is_verified, speaker_name = self._verify_speaker(
                            utterance_pcm, threshold=req_thresh
                        )
                        print(
                            f"[VOICE DEBUG] Speaker Verification Result: {is_verified} "
                            f"({speaker_name}) for '{text}' (RMS: {rms:.1f})",
                            flush=True,
                        )

                        if not is_verified:
                            print(
                                f"[VOICE REJECTED] Command '{text}' ignored by speaker verification.",
                                flush=True,
                            )
                            if len(words) >= 2 and (now_mono - self._last_reject_tts_time) >= 4.0:
                                self._last_reject_tts_time = now_mono
                                self.say_muted(
                                    "Command ignored. Unrecognized voice.", cooldown=0.6
                                )
                            with self._lock:
                                self.recognizer.Reset()
                            continue

                        print(
                            f"[VOICE PASSED VERIFICATION]: '{text}' (User: {speaker_name})",
                            flush=True,
                        )
                        cmd_string, speech_reply = matched_entry
                        register_cmd(cmd_string, speech_reply)
                    else:
                        with self._lock:
                            self.recognizer.Reset()
                elif partial_str:
                    print(f"[VOICE PARTIAL]: '{partial_str}'          ", end="\r", flush=True)

        except Exception as e:
            if self.is_running:
                print(f"[VOICE FATAL ERROR CRASH] {e}", flush=True)
                import traceback
                traceback.print_exc()

    def stop(self):
        self.is_running = False
        try:
            if self.esp_audio_serial and self.esp_audio_serial.is_open:
                self.esp_audio_serial.close()
        except Exception:
            pass
        if getattr(self, "voice_thread", None) and self.voice_thread.is_alive():
            self.voice_thread.join(timeout=0.5)

# -----------------------------------------------------------------------------
# Motor Controller (with Auto-Reconnect & Thread-Safe Serial Framing)
# -----------------------------------------------------------------------------
class MotorController:
    def __init__(self, port, baud, debug=False, reconnect_interval=2.0):
        self.port = port
        self.baud = int(baud)
        self.debug = bool(debug)
        self.ser = None
        self._io_lock = threading.Lock()
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval = float(reconnect_interval)
        self.connect()

    def connect(self):
        with self._io_lock:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass

            self._last_reconnect_attempt = time.monotonic()
            try:
                self.ser = serial.Serial(
                    self.port,
                    self.baud,
                    timeout=0.05,
                    write_timeout=0.05,
                )
                print(f"[INFO] Serial connected to {self.port}")
            except Exception as e:
                self.ser = None
                print(f"[WARNING] Serial connection failed: {e}")

    def _ensure_connection(self):
        """Non-blocking rate-limited reconnect check (must be called with _io_lock held)."""
        if self.ser is not None and self.ser.is_open:
            return True

        now = time.monotonic()
        if (now - self._last_reconnect_attempt) < self._reconnect_interval:
            return False

        self._last_reconnect_attempt = now
        try:
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=0.05,
                write_timeout=0.05,
            )
            print(f"[INFO] Serial reconnected to {self.port}")
            return True
        except Exception as e:
            self.ser = None
            if self.debug:
                print(f"[WARNING] Serial reconnect failed: {e}")
            return False

    def _write_packet(self, packet: bytes, label: str):
        with self._io_lock:
            if not self._ensure_connection():
                return
            try:
                self.ser.write(packet)
                if self.debug:
                    print(f"[SERIAL OUT] {label} | Hex: {packet.hex()}")
            except serial.SerialTimeoutException:
                # Drop packet if write buffer is busy so camera/control loop stays fluid
                if self.debug:
                    print(f"[SERIAL DROP] Write timeout for {label}")
            except Exception as e:
                print(f"[WARNING] Serial write error ({label}): {e}")
                try:
                    self.ser.close()
                except Exception:
                    pass
                self.ser = None

    def send_command(self, source, cmd):
        """Sends a framed text command packet: [0x03] + 'SOURCE:CMD\\n'."""
        payload = f"{source}:{cmd}\n"
        packet = bytes([0x03]) + payload.encode("utf-8")
        self._write_packet(packet, f"Source={source}, Cmd='{cmd}'")

    def send_hybrid_tracking_command(self, pan_angle, tilt_angle, steering_error, base_speed):
        """Sends a 5-byte hybrid tracking packet: [0x02, pan, tilt, encoded_error, speed]."""
        p_angle = pan_angle if pan_angle is not None else 135.0
        t_angle = tilt_angle if tilt_angle is not None else 85.0

        pan_scaled = max(30, min(240, min(255, int(float(p_angle)))))
        tilt_byte = max(0, min(180, int(float(t_angle))))

        err = steering_error if steering_error is not None else 0
        clamped_err = max(-100, min(100, int(round(float(err)))))
        encoded_error = max(0, min(255, clamped_err + 100))
        speed_byte = max(0, min(255, int(round(float(base_speed)))))

        packet = bytes([0x02, pan_scaled, tilt_byte, encoded_error, speed_byte])
        self._write_packet(
            packet,
            f"Hybrid Pan={pan_scaled} Tilt={tilt_byte} Err={encoded_error} Speed={speed_byte}",
        )

    def send_buzzer_command(self, state=True):
        """Sends a 1-byte buzzer state packet: 0x04 (ON) or 0x05 (OFF)."""
        header_byte = 0x04 if state else 0x05
        packet = bytes([header_byte])
        self._write_packet(packet, f"Buzzer={'ON' if state else 'OFF'}")

    def close(self):
        with self._io_lock:
            if self.ser and self.ser.is_open:
                try:
                    stop_pkt = bytes([0x03]) + b"HOLD:STOP\n"
                    self.ser.write(stop_pkt)
                except Exception:
                    pass
                try:
                    self.ser.close()
                except Exception:
                    pass
            self.ser = None

# -----------------------------------------------------------------------------
# Tracking and Following
# -----------------------------------------------------------------------------
class DistanceTrackingController:
    def __init__(
        self,
        ideal_box_height: float = 300.0,
        height_tolerance: float = 65.0,
        exit_margin: float = 15.0,
        box_height_alpha: float = 0.18,
        dist_cmd_min_dwell: float = 0.75,
    ):
        self.ideal_box_height = float(ideal_box_height)
        self.height_tolerance = float(height_tolerance)
        self.exit_margin = float(exit_margin)
        self.alpha = float(box_height_alpha)
        self.min_dwell = float(dist_cmd_min_dwell)

        self._lock = threading.Lock()
        self.smoothed_height: float | None = None
        self.active_cmd: str = "STOP"
        self._candidate_cmd: str = "STOP"
        self._candidate_since: float = time.monotonic()

    def reset(self, initial_cmd: str = "STOP"):
        with self._lock:
            self.smoothed_height = None
            self.active_cmd = initial_cmd
            self._candidate_cmd = initial_cmd
            self._candidate_since = time.monotonic()

    def _classify_with_hysteresis(self, h: float) -> str:
        low_enter = self.ideal_box_height - self.height_tolerance
        high_enter = self.ideal_box_height + self.height_tolerance

        low_exit = self.ideal_box_height - (self.height_tolerance + self.exit_margin)
        high_exit = self.ideal_box_height + (self.height_tolerance + self.exit_margin)

        if self.active_cmd == "STOP":
            # Wider band required to leave STOP than to enter it
            if h < low_exit:
                return "FORWARD"   # Person is farther away -> move forward
            if h > high_exit:
                return "BACKWARD"  # Person is closer -> move backward
            return "STOP"

        if self.active_cmd == "FORWARD":
            # Stop once within standard tolerance; only reverse if clearly too close
            if h >= low_enter:
                return "STOP" if h <= high_exit else "BACKWARD"
            return "FORWARD"

        if self.active_cmd == "BACKWARD":
            # Stop once within standard tolerance; only move forward if clearly too far
            if h <= high_enter:
                return "STOP" if h >= low_exit else "FORWARD"
            return "BACKWARD"

        return "STOP"

    def update(self, raw_box_height: float | None, now: float | None = None) -> tuple[str, float | None, bool]:
        """
        Updates smoothed box height and returns (command, smoothed_height, changed).
        If raw_box_height is None or <= 0 (target lost), immediately returns STOP.
        """
        ts = time.monotonic() if now is None else float(now)

        with self._lock:
            if raw_box_height is None or raw_box_height <= 0:
                changed = self.active_cmd != "STOP"
                self.active_cmd = "STOP"
                self._candidate_cmd = "STOP"
                self._candidate_since = ts
                return self.active_cmd, self.smoothed_height, changed

            h_raw = float(raw_box_height)
            if self.smoothed_height is None:
                self.smoothed_height = h_raw
            else:
                self.smoothed_height = (self.alpha * h_raw) + ((1.0 - self.alpha) * self.smoothed_height)
            desired_cmd = self._classify_with_hysteresis(self.smoothed_height)

            if desired_cmd != self._candidate_cmd:
                self._candidate_cmd = desired_cmd
                self._candidate_since = ts

            changed = False
            if self._candidate_cmd != self.active_cmd:
                if (ts - self._candidate_since) >= self.min_dwell:
                    self.active_cmd = self._candidate_cmd
                    changed = True

            return self.active_cmd, self.smoothed_height, changed

# -----------------------------------------------------------------------------
# Measure each loop iteration
# -----------------------------------------------------------------------------
class LoopProfiler:
    def __init__(self, interval=5.0):
        self.interval = float(interval)
        self._lock = threading.Lock()
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.last_report = time.monotonic()

    def record(self, stage, elapsed_sec):
        with self._lock:
            self.totals[stage] += float(elapsed_sec)
            self.counts[stage] += 1

    @contextmanager
    def measure(self, stage):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, time.perf_counter() - t0)

    def maybe_report(self):
        now = time.monotonic()
        with self._lock:
            if (now - self.last_report) < self.interval:
                return
            parts = []
            for stage, total in self.totals.items():
                cnt = self.counts[stage]
                avg_ms = (total / cnt * 1000.0) if cnt else 0.0
                parts.append(f"{stage}={avg_ms:.1f}ms")
            if parts:
                print("[PROFILE] " + " | ".join(parts), flush=True)
            self.totals.clear()
            self.counts.clear()
            self.last_report = now


# -----------------------------------------------------------------------------
# Decouple Heavy Vision Models with "Latest-Frame" Worker Threads
# -----------------------------------------------------------------------------
class LatestFrameWorker:
    """Runs a slow vision callable asynchronously on the freshest frame."""

    def __init__(
        self,
        name,
        infer_fn,
        log_every_sec=5.0,
        min_interval_sec=0.0,
        max_age_sec=0.30,
    ):
        self.name = name
        self.infer_fn = infer_fn
        self.log_every_sec = float(log_every_sec)
        self.min_interval_sec = max(0.0, float(min_interval_sec))
        self.max_age_sec = float(max_age_sec)

        self._in_q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_result = None
        self._stop_event = threading.Event()
        self._last_infer_start = 0.0

        self._infer_sum_ms = 0.0
        self._age_sum_ms = 0.0
        self._count = 0
        self._last_log = time.monotonic()

        self._thread = threading.Thread(
            target=self._run, name=f"worker-{name}", daemon=True
        )
        self._thread.start()

    def submit(self, frame, capture_ts=None):
        if frame is None or getattr(frame, "size", 0) == 0 or self._stop_event.is_set():
            return
        if capture_ts is None:
            capture_ts = time.monotonic()

        payload = (frame.copy(), float(capture_ts))
        if self._in_q.full():
            try:
                self._in_q.get_nowait()
            except queue.Empty:
                pass
        try:
            self._in_q.put_nowait(payload)
        except queue.Full:
            pass

    def get_latest(self):
        with self._lock:
            return self._latest_result

    def _run(self):
        while not self._stop_event.is_set():
            try:
                frame, capture_ts = self._in_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self._stop_event.is_set():
                break

            now_mono = time.monotonic()
            if self.max_age_sec > 0.0 and (now_mono - capture_ts) > self.max_age_sec:
                continue

            if self.min_interval_sec > 0.0:
                since_last = now_mono - self._last_infer_start
                if since_last < self.min_interval_sec:
                    time.sleep(min(self.min_interval_sec - since_last, 0.02))
                    # Swap to an even fresher queued frame if one arrived while pacing
                    try:
                        newer_frame, newer_ts = self._in_q.get_nowait()
                        frame, capture_ts = newer_frame, newer_ts
                    except queue.Empty:
                        pass
                    now_mono = time.monotonic()
                    if self.max_age_sec > 0.0 and (now_mono - capture_ts) > self.max_age_sec:
                        continue

            self._last_infer_start = time.monotonic()
            t0 = time.perf_counter()
            try:
                res = self.infer_fn(frame)
            except Exception as e:
                print(f"[{self.name} WORKER ERROR]: {e}", flush=True)
                continue

            done_mono = time.monotonic()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            age_ms = (done_mono - capture_ts) * 1000.0

            if isinstance(res, dict):
                res.setdefault("capture_ts", capture_ts)
                res["done_ts"] = done_mono
                res["infer_ms"] = infer_ms
                res["age_ms"] = age_ms

            with self._lock:
                self._latest_result = res
                self._infer_sum_ms += infer_ms
                self._age_sum_ms += age_ms
                self._count += 1

                if (done_mono - self._last_log) >= self.log_every_sec and self._count > 0:
                    avg_infer = self._infer_sum_ms / self._count
                    avg_age = self._age_sum_ms / self._count
                    print(
                        f"[WORKER:{self.name}] runs={self._count} | "
                        f"avg_infer={avg_infer:.1f}ms | avg_age={avg_age:.1f}ms",
                        flush=True,
                    )
                    self._infer_sum_ms = 0.0
                    self._age_sum_ms = 0.0
                    self._count = 0
                    self._last_log = done_mono

    def stop(self):
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

# ----------------------------------------------------------
#
# ----------------------------------------------------------
@dataclass
class FramePacket:
    seq: int
    frame: np.ndarray
    captured_at: float  # time.monotonic()


class LatestFrameMailbox:
    """Thread-safe single-slot buffer that always keeps only the newest frame."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._seq = 0
        self._latest: Optional[FramePacket] = None
        self._closed = False

    def publish(self, frame: np.ndarray, captured_at: Optional[float] = None) -> int:
        ts = time.monotonic() if captured_at is None else captured_at
        with self._cond:
            self._seq += 1
            seq = self._seq
            # Store reference or copy depending on whether caller mutates buffer in-place
            self._latest = FramePacket(seq=seq, frame=frame, captured_at=ts)
            self._cond.notify_all()
            return seq

    def wait_for_new(self, after_seq: int, timeout: float = 0.05) -> Optional[FramePacket]:
        with self._cond:
            if self._closed:
                return None
            if self._latest is None or self._latest.seq <= after_seq:
                self._cond.wait_for(
                    lambda: self._closed or (self._latest is not None and self._latest.seq > after_seq),
                    timeout=timeout,
                )
            if self._closed or self._latest is None or self._latest.seq <= after_seq:
                return None
            return self._latest

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()


class RateLimitedVisionWorker:
    """Runs inference at a capped rate on the newest frame without backlog."""

    def __init__(
        self,
        name: str,
        mailbox: LatestFrameMailbox,
        infer_fn: Callable[[np.ndarray, float], Any],
        on_result: Optional[Callable[[Any, float, float], None]] = None,
        min_interval_s: float = 0.10,   # e.g., 0.10s = 10 Hz max rate
        max_age_s: float = 0.25,        # drop frames older than 250 ms
        copy_frame: bool = False,
        npu_lock: Optional[threading.Lock] = None,
    ) -> None:
        self.name = name
        self.mailbox = mailbox
        self.infer_fn = infer_fn
        self.on_result = on_result
        self.min_interval_s = min_interval_s
        self.max_age_s = max_age_s
        self.copy_frame = copy_frame
        self.npu_lock = npu_lock

        self._stop_event = threading.Event()
        self._last_seen_seq = 0
        self._last_infer_start = 0.0
        self._thread = threading.Thread(target=self._run, name=f"worker-{name}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, join_timeout: float = 1.0) -> None:
        self._stop_event.set()
        self._thread.join(timeout=join_timeout)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            pkt = self.mailbox.wait_for_new(self._last_seen_seq, timeout=0.05)
            if pkt is None:
                continue

            # Mark latest sequence as seen so we don't re-read the same frame
            self._last_seen_seq = pkt.seq
            now = time.monotonic()

            # 1. Drop stale frames that waited too long before a worker slot opened
            age_s = now - pkt.captured_at
            if age_s > self.max_age_s:
                continue

            # 2. Enforce cadence cap (e.g., 10 Hz -> at least 100 ms between starts)
            since_last = now - self._last_infer_start
            if since_last < self.min_interval_s:
                time.sleep(min(self.min_interval_s - since_last, 0.02))
                # Re-fetch newest frame after pacing sleep in case a fresher one arrived
                pkt = self.mailbox.wait_for_new(self._last_seen_seq - 1, timeout=0.001) or pkt
                self._last_seen_seq = pkt.seq
                now = time.monotonic()
                if (now - pkt.captured_at) > self.max_age_s:
                    continue

            frame_input = pkt.frame.copy() if self.copy_frame else pkt.frame
            self._last_infer_start = now

            t0 = time.monotonic()
            try:
                if self.npu_lock is not None:
                    with self.npu_lock:
                        result = self.infer_fn(frame_input, pkt.captured_at)
                else:
                    result = self.infer_fn(frame_input, pkt.captured_at)
            except Exception as exc:
                print(f"[WORKER:{self.name} ERROR] {exc}")
                continue

            infer_ms = (time.monotonic() - t0) * 1000.0
            age_ms = (time.monotonic() - pkt.captured_at) * 1000.0

            if self.on_result is not None:
                self.on_result(result, infer_ms, age_ms)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def extract_face_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """Extract a clamped face bounding-box crop from frame with optional padding."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = int(frame.shape[0]), int(frame.shape[1])
    x, y, bw, bh = bbox
    if bw <= 1.0 or bh <= 1.0 or w <= 0 or h <= 0:
        return np.empty((0, 0, 3), dtype=np.uint8)

    pad_x = bw * float(pad_ratio)
    pad_y = bh * float(pad_ratio)

    x1 = max(0, min(w, int(round(x - pad_x))))
    y1 = max(0, min(h, int(round(y - pad_y))))
    x2 = max(0, min(w, int(round(x + bw + pad_x))))
    y2 = max(0, min(h, int(round(y + bh + pad_y))))

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)

    return np.ascontiguousarray(frame[y1:y2, x1:x2])

# =====================================================
def make_face_worker_fn(
    detector: Any,
    embedder: Any,
    db: Any,
    get_override_name: Optional[Callable[[], Optional[str]]] = None,
    *,
    max_distance: float = 0.72,
    required_streak: int = 2,
    frame_is_rgb: bool = True,
    default_confidence: float = 0.9,
    max_det_width: int = 640,
) -> Callable[[np.ndarray], dict[str, Any]]:
    """Create a per-frame face detection, embedding match, and authorization worker."""
    streak_limit = max(1, int(required_streak))
    state_lock = threading.Lock()
    log_state: dict[str, Any] = {"last_time": 0.0, "last_name": None}
    streak_state: dict[str, Any] = {"name": None, "count": 0}

    def _empty_result(ts: float) -> dict[str, Any]:
        return {"matches": [], "timestamp": ts}

    def _run_face(frame: np.ndarray) -> dict[str, Any]:
        ts = time.time()
        if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
            return _empty_result(ts)

        if detector is None or embedder is None or db is None:
            return _empty_result(ts)

        h, w = int(frame.shape[0]), int(frame.shape[1])
        if h <= 0 or w <= 0:
            return _empty_result(ts)

        if max_det_width > 0 and w > max_det_width:
            scale = max_det_width / float(w)
            det_w = max_det_width
            det_h = max(1, int(round(h * scale)))
            det_src = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_AREA)
            inv_scale_x = w / float(det_w)
            inv_scale_y = h / float(det_h)
        else:
            det_src = frame
            det_w, det_h = w, h
            inv_scale_x = 1.0
            inv_scale_y = 1.0

        det_input = (
            cv2.cvtColor(det_src, cv2.COLOR_RGB2BGR) if frame_is_rgb else det_src
        )

        with state_lock:
            detector.setInputSize((det_w, det_h))
            _, faces = detector.detect(det_input)

        if faces is None or len(faces) == 0:
            with state_lock:
                streak_state["name"] = None
                streak_state["count"] = 0
            return _empty_result(ts)

        override_name: Optional[str] = None
        if callable(get_override_name):
            try:
                raw_override = get_override_name()
                if raw_override is not None:
                    override_name = str(raw_override).strip().lower() or None
            except Exception:
                override_name = None

        matches: list[dict[str, Any]] = []

        for face in faces:
            if len(face) < 4:
                continue

            fx = float(face[0]) * inv_scale_x
            fy = float(face[1]) * inv_scale_y
            fw = float(face[2]) * inv_scale_x
            fh = float(face[3]) * inv_scale_y

            crop = extract_face_crop(frame, (fx, fy, fw, fh), pad_ratio=0.0)
            if crop.size == 0:
                continue

            query_vector = embedder.get_embedding(crop, input_is_rgb=frame_is_rgb)
            if query_vector is None:
                continue

            matched_name, matched_role, dist = db.match_face(
                query_vector, max_distance=max_distance
            )
            name = str(matched_name) if matched_name else "Unknown"
            role = str(matched_role) if matched_role else "none"
            is_known = name != "Unknown"

            now_mono = time.monotonic()
            with state_lock:
                if not is_known:
                    streak_state["name"] = None
                    streak_state["count"] = 0
                elif streak_state["name"] == name:
                    streak_state["count"] += 1
                else:
                    streak_state["name"] = name
                    streak_state["count"] = 1

                streak_count = int(streak_state["count"]) if is_known else 0
                confirmed_lock = is_known and (streak_count >= streak_limit)

                should_log = (
                    name != log_state["last_name"]
                    or (now_mono - log_state["last_time"]) >= 1.0
                )
                if should_log:
                    log_state["last_time"] = now_mono
                    log_state["last_name"] = name

            if should_log:
                status = "MATCH" if is_known else "REJECT"
                lock_tag = (
                    "LOCKED"
                    if confirmed_lock
                    else f"STREAK {streak_count}/{streak_limit}"
                )
                dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
                print(
                    f"[FACE DEBUG] {status} ({lock_tag}) | name={name} ({role}) | "
                    f"dist={dist_str} | max_distance={max_distance:.2f}",
                    flush=True,
                )

            fx1 = max(0, min(int(round(fx)), w))
            fy1 = max(0, min(int(round(fy)), h))
            fx2 = max(0, min(int(round(fx + fw)), w))
            fy2 = max(0, min(int(round(fy + fh)), h))

            role_allowed = (
                role.upper() == "VIP_FOLLOW"
                or (override_name is not None and name.lower() == override_name)
            )
            is_authorized = bool(confirmed_lock and role_allowed)

            conf = float(face[14]) if len(face) > 14 else default_confidence

            matches.append(
                {
                    "name": name,
                    "role": role,
                    "dist": dist,
                    "bbox": [fx1, fy1, fx2, fy2],
                    "confidence": conf,
                    "streak": streak_count,
                    "confirmed": confirmed_lock,
                    "authorized": is_authorized,
                }
            )

        return {"matches": matches, "timestamp": ts}

    return _run_face

# ==================================================================
def make_gesture_worker_fn(recognizer, *, frame_is_rgb: bool = True):
    def _run_gesture(frame):
        ts = time.time()
        if recognizer is None:
            return {
                "raw_cmd": None,
                "hand_side": "Unknown",
                "wrist_y": None,
                "landmarks": None,
                "timestamp": ts,
            }

        try:
            raw_result = recognizer.process_frame(
                frame, None, is_rgb=frame_is_rgb, draw_overlay=False
            )
        except TypeError:
            raw_result = recognizer.process_frame(frame, None)

        wrist_y = getattr(recognizer, "last_wrist_y", None)
        landmarks = (
            getattr(recognizer, "last_landmarks", None)
            or getattr(recognizer, "hand_landmarks", None)
            or getattr(recognizer, "landmarks", None)
        )

        if isinstance(raw_result, tuple):
            if len(raw_result) == 4:
                raw_cmd, hand_side, wrist_y, landmarks = raw_result
            elif len(raw_result) == 3:
                raw_cmd, hand_side, wrist_y = raw_result
            elif len(raw_result) == 2:
                raw_cmd, hand_side = raw_result
            else:
                raw_cmd, hand_side = raw_result[0], "Unknown"
        else:
            raw_cmd, hand_side = raw_result, "Unknown"

        return {
            "raw_cmd": raw_cmd,
            "hand_side": hand_side,
            "wrist_y": wrist_y,
            "landmarks": landmarks,
            "timestamp": time.time(),
        }

    return _run_gesture

# -----------------------------------------------------------------------------
# Target lock
# -----------------------------------------------------------------------------
class FaceTargetLocker:
    """Maintains a stable follow lock from face worker outputs using confirmed & authorized."""

    def __init__(self, hold_timeout_s: float = 0.6) -> None:
        self.hold_timeout_s = max(0.0, float(hold_timeout_s))
        self._lock = threading.Lock()
        self.locked_name: Optional[str] = None
        self.locked_role: Optional[str] = None
        self.locked_bbox: Optional[list[int]] = None
        self.locked_dist: Optional[float] = None
        self.last_seen_ts: float = 0.0

    def clear(self) -> None:
        with self._lock:
            self.locked_name = None
            self.locked_role = None
            self.locked_bbox = None
            self.locked_dist = None
            self.last_seen_ts = 0.0

    def update(
        self,
        face_result: Optional[dict[str, Any]],
        *,
        override_name: Optional[str] = None,
        now: Optional[float] = None,
    ) -> dict[str, Any]:
        ts = time.monotonic() if now is None else float(now)
        desired_name = override_name.strip().lower() if override_name else None

        matches = []
        if isinstance(face_result, dict):
            raw_matches = face_result.get("matches")
            if isinstance(raw_matches, list):
                matches = raw_matches

        # Only consider faces that passed both streak confirmation (confirmed)
        # and role/override authorization (authorized).
        eligible: list[dict[str, Any]] = []
        for m in matches:
            if not isinstance(m, dict):
                continue
            if not (m.get("confirmed") is True and m.get("authorized") is True):
                continue
            m_name = str(m.get("name", "")).strip()
            if not m_name or m_name == "Unknown":
                continue
            if desired_name is not None and m_name.lower() != desired_name:
                continue
            eligible.append(m)

        with self._lock:
            if eligible:
                # Prefer lowest cosine distance, breaking ties by largest bbox area.
                def _rank(item: dict[str, Any]) -> tuple[float, float]:
                    dist_val = item.get("dist")
                    d = float(dist_val) if isinstance(dist_val, (int, float)) else 999.0
                    bbox = item.get("bbox") or [0, 0, 0, 0]
                    area = max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1]))
                    return (d, -float(area))

                best = min(eligible, key=_rank)
                raw_bbox = best.get("bbox")
                self.locked_name = str(best["name"])
                self.locked_role = str(best.get("role", "none"))
                self.locked_bbox = [int(v) for v in raw_bbox] if isinstance(raw_bbox, (list, tuple)) and len(raw_bbox) == 4 else None
                self.locked_dist = (
                    float(best["dist"]) if isinstance(best.get("dist"), (int, float)) else None
                )
                self.last_seen_ts = ts
                return {
                    "locked": True,
                    "name": self.locked_name,
                    "role": self.locked_role,
                    "bbox": list(self.locked_bbox) if self.locked_bbox is not None else None,
                    "dist": self.locked_dist,
                    "state": "LOCKED",
                }

            # No newly confirmed+authorized match this cycle: check grace window.
            if self.locked_name is not None:
                if desired_name is not None and self.locked_name.lower() != desired_name:
                    self.locked_name = None
                    self.locked_role = None
                    self.locked_bbox = None
                    self.locked_dist = None
                    self.last_seen_ts = 0.0
                elif (ts - self.last_seen_ts) <= self.hold_timeout_s:
                    return {
                        "locked": True,
                        "name": self.locked_name,
                        "role": self.locked_role,
                        "bbox": list(self.locked_bbox) if self.locked_bbox is not None else None,
                        "dist": self.locked_dist,
                        "state": "HOLDING",
                    }
                else:
                    self.locked_name = None
                    self.locked_role = None
                    self.locked_bbox = None
                    self.locked_dist = None
                    self.last_seen_ts = 0.0

            return {
                "locked": False,
                "name": None,
                "role": None,
                "bbox": None,
                "dist": None,
                "state": "UNLOCKED",
            }

# -----------------------------------------------------------------------------
# Smooths noisy per-frame hand gesture outputs
# -----------------------------------------------------------------------------
class GestureDebouncer:
    """Debounces noisy per-frame gesture predictions with sliding-window consensus."""

    def __init__(
        self,
        window_size=5,
        min_votes=3,
        max_age_sec=0.45,
        action_thresholds=None,
    ):
        win_len = max(1, int(window_size))
        self.window_size = win_len
        self.min_votes = max(1, min(int(min_votes), win_len))
        self.max_age_sec = max(0.0, float(max_age_sec))
        self.action_thresholds = (
            dict(action_thresholds)
            if isinstance(action_thresholds, dict)
            else {"TAKE_PHOTO": win_len}
        )

        self._lock = threading.Lock()
        self.window = deque(maxlen=win_len)
        self.confirmed_cmd = None
        self._last_seen_done_ts = None

    def reset(self):
        with self._lock:
            self.window.clear()
            self.confirmed_cmd = None
            self._last_seen_done_ts = None

    def update(self, gesture_res, now_mono=None):
        """
        Returns:
            (active_cmd, newly_confirmed_event_cmd, is_stale, latest_raw_cmd)
        """
        ts = time.monotonic() if now_mono is None else float(now_mono)

        with self._lock:
            if not gesture_res:
                return None, None, True, None

            done_ts = gesture_res.get("done_ts")
            capture_ts = gesture_res.get("capture_ts", done_ts)
            age_sec = (ts - float(capture_ts)) if capture_ts is not None else 0.0
            is_stale = age_sec > self.max_age_sec
            latest_raw_cmd = gesture_res.get("raw_cmd")

            if done_ts is not None and done_ts != self._last_seen_done_ts:
                self._last_seen_done_ts = done_ts
                if not is_stale:
                    self.window.append(latest_raw_cmd)
                else:
                    self.window.clear()
                    self.confirmed_cmd = None

            if is_stale:
                return None, None, True, None

            if not self.window:
                return self.confirmed_cmd, None, False, latest_raw_cmd

            counts = Counter(self.window)
            candidate, votes = counts.most_common(1)[0]
            needed_votes = self.action_thresholds.get(candidate, self.min_votes)

            event_cmd = None
            if candidate is None:
                # Clear active gesture once neutral/closed/no-gesture dominates
                if votes >= self.min_votes:
                    self.confirmed_cmd = None
            elif votes >= needed_votes:
                if candidate != self.confirmed_cmd:
                    self.confirmed_cmd = candidate
                    event_cmd = candidate

            return self.confirmed_cmd, event_cmd, False, latest_raw_cmd
    
# -----------------------------------------------------------------------------
# Main Application Loop
# -----------------------------------------------------------------------------
def main():
    motor = MotorController(SERIAL_PORT, BAUD_RATE)
    face_db = FaceDatabase(DB_PATH)
    speaker = RobotSpeaker()
    voice_queue = queue.Queue()
    voice_rec = None
    gesture_rec = None

    print("[TRACE] Initializing IMX500 camera pipeline...")
    try:
        cam = IMX500Camera(model_path=IMX500_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] IMX500 failed to bind: {e}")
        speaker.stop()
        motor.close()
        return

    try:
        hailo_embedder = Hailo10FaceEmbedder(HAILO10_FACE_HEF)
        print("[INFO] Hailo Face Embedder successfully initialized!")
    except Exception as e:
        print(f"[WARNING] Hailo initialization skipped: {e}")
        hailo_embedder = None

    face_detector = None
    if os.path.exists(YUNET_MODEL_PATH):
        try:
            face_detector = cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config="",
                input_size=(FRAME_WIDTH, FRAME_HEIGHT),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000,
            )
            print("[INFO] YuNet Face Detector initialized successfully.")
        except Exception as e:
            print(f"[WARNING] Failed to create YuNet detector: {e}")
    else:
        print(
            f"[WARNING] YuNet model file missing at {YUNET_MODEL_PATH}. "
            "Falling back to bounding-box estimation."
        )

    pose_fall_detector = PoseFallDetector(frame_height=FRAME_HEIGHT, frame_width=FRAME_WIDTH)
    terrain_classifier = TerrainClassifier()

    print("[TRACE] Initializing GestureRecognizer...")
    try:
        gesture_rec = GestureRecognizer(max_proc_width=320, debug_log=False)
        print("[INFO] Hand Gesture Recognizer initialized.")
    except Exception as e:
        print(f"[WARNING] Gesture module failed: {e}")

    print("[TRACE] Initializing OfflineVoiceRecognizer...")
    try:
        voice_rec = OfflineVoiceRecognizer(
            command_queue=voice_queue,
            model_path=MODEL_PATH,
            speaker=speaker,
        )
        print("[INFO] Offline Voice Control initialized.")
    except Exception as e:
        print(f"[WARNING] Voice module failed: {e}")

    print("[TRACE] Voice module block cleared. Setting up loop variables...")

    manual_voice_override = None
    voice_override_expiry_time = 0.0
    VOICE_OVERRIDE_DURATION = 4.0
    target_role_to_follow = "VIP_FOLLOW"
    target_name_override = None
    override_lock = threading.Lock()

    def _get_target_override_name() -> Optional[str]:
        with override_lock:
            return target_name_override

    def _set_target_override_name(name: Optional[str]) -> None:
        nonlocal target_name_override
        with override_lock:
            target_name_override = name.strip().lower() if name else None

    class DefaultServoTracker:
        def __init__(self):
            self.pan_angle = 135
            self.tilt_angle = 85

    PAN_MIN, PAN_MAX = 40, 220
    TILT_MIN, TILT_MAX = 45, 105

    servo_tracker = DefaultServoTracker()
    last_snapshot_time = 0.0
    snapshot_cooldown = 15.0
    photo_armed = True
    last_greeted_times = {}
    greeting_cooldown_seconds = 3600.0

    last_sent_source = None
    last_sent_command = None
    last_send_time = 0.0

    frame_count = 0
    current_frame_command = "STOP"
    current_source = "HOLD"
    last_serial_send_time = 0.0
    SERIAL_SEND_INTERVAL = 0.033
    missing_target_count = 0
    MAX_MISSING_FRAMES = 5
    last_valid_target = None
    current_terrain = "PAVEMENT"
    is_target_locked = False
    locked_body_box = None
    lock_timer = 0.0
    LOCK_TIMEOUT = 5.0
    fall_hold_until = 0.0
    target_clothing_hist = None
    clothing_locked_name: Optional[str] = None
    is_clothing_locked = False
    CLOTHING_LOCK_TIMEOUT = 2.0
    clothing_lock_timer = 0.0
    buzzer_triggered = False
    last_fall_speech_time = 0.0
    FALL_SPEECH_REPEAT_INTERVAL = 10.0
    current_tracked_name = "Unknown"
    IDEAL_BOX_HEIGHT = 125.0

    profiler = LoopProfiler(interval=5.0)
    face_locker = FaceTargetLocker(hold_timeout_s=0.6)

    def _clear_follow_locks() -> None:
        nonlocal target_clothing_hist, clothing_locked_name, is_clothing_locked, clothing_lock_timer
        face_locker.clear()
        target_clothing_hist = None
        clothing_locked_name = None
        is_clothing_locked = False
        clothing_lock_timer = 0.0

    dist_ctrl = DistanceTrackingController(
        ideal_box_height=IDEAL_BOX_HEIGHT,
        height_tolerance=20.0,
        exit_margin=8.0,
        box_height_alpha=0.18,
        dist_cmd_min_dwell=0.75,
    )

    pending_voice_cooldown_until = None
    VOICE_POST_SPEECH_COOLDOWN = 1.2
    ignore_voice_until = 0.0
    VOICE_SPEECH_IGNORE_DURATION = 9.0
    VIP_LOCK_GRACE_SEC = 0.65
    vip_grace_active = False
    last_vip_lock_mono = 0.0
    last_vip_bbox: list[int] | None = None
    last_vip_name: str | None = None
    last_vip_role: str | None = None

    FACE_STRIDE = 4
    GESTURE_STRIDE = 5
    MAX_FACE_RESULT_AGE = 1.0
    MAX_GESTURE_RESULT_AGE = 0.8

    _last_vip_summary = None
    _last_ui_time = 0.0
    _last_vip_log_time = 0.0

    face_worker = LatestFrameWorker(
        name="face",
        infer_fn=make_face_worker_fn(
            face_detector,
            hailo_embedder,
            face_db,
            get_override_name=_get_target_override_name,
            max_distance=0.72,
            required_streak=2,
            frame_is_rgb=True,
        ),
        log_every_sec=5.0,
        min_interval_sec=0.12,
        max_age_sec=MAX_FACE_RESULT_AGE,
    )
    gesture_worker = LatestFrameWorker(
        name="gesture",
        infer_fn=make_gesture_worker_fn(gesture_rec, frame_is_rgb=True),
        log_every_sec=5.0,
        min_interval_sec=0.10,
        max_age_sec=MAX_GESTURE_RESULT_AGE,
    )
    gesture_debouncer = GestureDebouncer(
        window_size=5,
        min_votes=3,
        max_age_sec=MAX_GESTURE_RESULT_AGE,
        action_thresholds={"TAKE_PHOTO": 5},
    )

    def say_async(text: Optional[str]) -> None:
        if not text or speaker is None:
            return
        speaker.say(text)

    try:
        _last_vip_summary = None
        last_gesture_time = 0.0
        persisted_gesture = "STOP"
        IMSHOW_STALL_SEC = 0.160    # Flag only if imshow + waitKey exceeds 160 ms
        LOOP_GAP_STALL_SEC = 0.250  # Flag only if full loop-to-loop gap exceeds 250 ms
        prev_loop_perf = None
        print("[DEBUG] Entering main loop now...", flush=True)
        while True:
            now_loop_perf = time.perf_counter()
            if prev_loop_perf is not None:
                _loop_gap_dt = now_loop_perf - prev_loop_perf
                profiler.record("loop_gap", _loop_gap_dt)
                if frame_count > 1 and _loop_gap_dt > LOOP_GAP_STALL_SEC:
                    print(f"[STALL:LOOP_GAP] {_loop_gap_dt * 1000.0:.1f} ms", flush=True)
            prev_loop_perf = now_loop_perf

            frame_count += 1
            if frame_count % 30 == 0:
                print(
                    f"[STATUS] Running... Frame: {frame_count} | "
                    f"Command: {current_frame_command} | Source: {current_source}"
                )
            try:
                t0 = time.perf_counter()
                frame, metadata = cam.read()
                capture_ts = getattr(cam, "last_capture_ts", time.monotonic())
                _cam_dt = time.perf_counter() - t0
                profiler.record("cam_read", _cam_dt)
                if frame is not None and _cam_dt > 0.12:
                    print(f"[STALL:CAM_READ] {_cam_dt * 1000.0:.1f} ms", flush=True)

                if frame is None:
                    profiler.maybe_report()
                    time.sleep(0.005)
                    continue

                now_ts = time.time()
                current_time = now_ts

                t0 = time.perf_counter()
                display_frame = frame.copy()
                target_bbox = None
                target_x_center = None

                if frame_count % 3 == 0:
                    current_terrain = terrain_classifier.analyze_surface(
                        frame, display_frame, draw_overlay=True, frame_is_rgb=True
                    )

                detections = cam.parse_detections(
                    metadata, CONFIDENCE_THRESHOLD, TARGET_CLASS_ID
                )
                profiler.record("prep_and_det", time.perf_counter() - t0)

                # --- Gesture Worker (Submit + Debounce + Stale Check) ---
                t0 = time.perf_counter()
                if gesture_rec and frame_count % GESTURE_STRIDE == 0:
                    gesture_worker.submit(frame, capture_ts=capture_ts)

                latest_gesture = gesture_worker.get_latest()
                now_mono = time.monotonic()
                gest_age_ms = (
                    ((now_mono - latest_gesture["capture_ts"]) * 1000.0)
                    if latest_gesture and "capture_ts" in latest_gesture
                    else -1.0
                )

                active_gesture, new_gesture_event, is_stale, latest_raw_cmd = gesture_debouncer.update(
                    latest_gesture, now_mono=now_mono
                )

                if latest_raw_cmd != "TAKE_PHOTO":
                    photo_armed = True

                take_photo_triggered = False
                if (
                    not is_stale
                    and new_gesture_event == "TAKE_PHOTO"
                    and latest_raw_cmd == "TAKE_PHOTO"
                    and photo_armed
                    and (current_time - last_snapshot_time) >= snapshot_cooldown
                ):
                    take_photo_triggered = True
                    photo_armed = False
                    last_snapshot_time = current_time

                def _pt_xy(pt):
                    if hasattr(pt, "x") and hasattr(pt, "y"):
                        return float(pt.x), float(pt.y)
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        return float(pt[0]), float(pt[1])
                    raise TypeError(
                        f"Unsupported landmark point type: {type(pt).__name__} ({dir(pt)})"
                    )

                hand_side = (
                    latest_gesture.get("hand_side", "Unknown") if latest_gesture else "Unknown"
                )
                lm = (
                    latest_gesture.get("landmarks")
                    if (latest_gesture and (not is_stale or (0.0 <= gest_age_ms <= 250.0)))
                    else None
                )
                if lm:
                    disp_h, disp_w = display_frame.shape[:2]
                    for conn in HAND_CONNECTIONS:
                        p1_idx = getattr(conn, "start", conn[0])
                        p2_idx = getattr(conn, "end", conn[1])
                        if 0 <= p1_idx < len(lm) and 0 <= p2_idx < len(lm):
                            x1_norm, y1_norm = _pt_xy(lm[p1_idx])
                            x2_norm, y2_norm = _pt_xy(lm[p2_idx])
                            x1 = int(x1_norm * disp_w)
                            y1 = int(y1_norm * disp_h)
                            x2 = int(x2_norm * disp_w)
                            y2 = int(y2_norm * disp_h)
                            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)

                if new_gesture_event:
                    persisted_gesture = new_gesture_event
                    last_gesture_time = current_time
                    print(
                        f"[GESTURE] Confirmed new gesture: {new_gesture_event} "
                        f"(infer={latest_gesture.get('infer_ms', 0):.1f}ms, age={gest_age_ms:.1f}ms)"
                    )

                if active_gesture is not None and not is_stale:
                    gesture_cmd = active_gesture
                    last_gesture_time = current_time
                elif (current_time - last_gesture_time) < 0.5 and not is_stale:
                    gesture_cmd = persisted_gesture
                else:
                    gesture_cmd = None
                    persisted_gesture = "STOP"
                profiler.record("gesture_stage", time.perf_counter() - t0)

                # --- Face Worker (Submit + Stale Check + Clothing Lock) ---
                t0 = time.perf_counter()
                if (
                    face_detector is not None
                    and hailo_embedder is not None
                    and frame_count % FACE_STRIDE == 0
                ):
                    face_worker.submit(frame, capture_ts=capture_ts)

                latest_face_result = face_worker.get_latest()
                face_matches = []
                if latest_face_result:
                    result_ts = latest_face_result.get("timestamp", 0.0)
                    if (current_time - result_ts) <= MAX_FACE_RESULT_AGE:
                        face_matches = latest_face_result.get("matches", [])

                for match in face_matches:
                    name = match.get("name")
                    if name and name != "Unknown":
                        if name not in last_greeted_times or (
                            current_time - last_greeted_times[name] > greeting_cooldown_seconds
                        ):
                            if speaker:
                                if voice_rec and hasattr(voice_rec, "say_muted"):
                                    voice_rec.say_muted(
                                        f"Hello {name}. Welcome back.",
                                        extra_hold=2.2,
                                        cooldown=1.4,
                                    )
                                else:
                                    if voice_rec:
                                        voice_rec.set_speaking(True)
                                    try:
                                        speaker.say(f"Hello {name}. Welcome back.")
                                    finally:
                                        if voice_rec:
                                            voice_rec.set_speaking(False, cooldown=1.4)
                            last_greeted_times[name] = current_time

                filtered_face_result = (
                    {
                        "matches": face_matches,
                        "timestamp": latest_face_result.get("timestamp", current_time),
                    }
                    if latest_face_result
                    else None
                )
                active_override = _get_target_override_name()
                lock_status = face_locker.update(
                    filtered_face_result,
                    override_name=active_override,
                    now=now_mono,
                )

                chosen_det = None
                authorized_face_detected = False

                if lock_status["locked"] and lock_status["bbox"] is not None:
                    authorized_face_detected = True
                    current_tracked_name = lock_status["name"] or current_tracked_name
                    fx1, fy1, fx2, fy2 = lock_status["bbox"]
                    face_conf = next(
                        (
                            m.get("confidence", 0.9)
                            for m in face_matches
                            if m.get("name") == lock_status["name"]
                        ),
                        0.9,
                    )

                    matched_body_det = None
                    for det in detections:
                        bx1, by1, bx2, by2 = det["bbox"]
                        if bx1 <= fx1 <= bx2 and by1 <= fy1 <= by2:
                            matched_body_det = det
                            break

                    chosen_det = (
                        matched_body_det
                        if matched_body_det
                        else {
                            "bbox": [fx1, fy1, fx2, fy2],
                            "confidence": face_conf,
                        }
                    )

                    # Only capture/refresh clothing histogram on a fresh LOCKED state
                    if lock_status["state"] == "LOCKED":
                        bx1, by1, bx2, by2 = chosen_det["bbox"]
                        torso_ymin = int(by1 + (by2 - by1) * 0.2)
                        torso_ymax = int(by1 + (by2 - by1) * 0.7)
                        img_h, img_w = frame.shape[:2]
                        torso_crop = frame[
                            max(0, torso_ymin) : min(img_h, torso_ymax),
                            max(0, bx1) : min(img_w, bx2),
                        ]

                        if torso_crop.size > 0:
                            hsv_torso = cv2.cvtColor(torso_crop, cv2.COLOR_RGB2HSV)
                            target_clothing_hist = cv2.calcHist(
                                [hsv_torso], [0, 1], None, [32, 32], [0, 180, 0, 256]
                            )
                            cv2.normalize(
                                target_clothing_hist,
                                target_clothing_hist,
                                alpha=0,
                                beta=1,
                                norm_type=cv2.NORM_MINMAX,
                            )
                            clothing_locked_name = lock_status["name"]
                            is_clothing_locked = True
                            clothing_lock_timer = current_time

                override_matches_clothing = active_override is None or (
                    clothing_locked_name is not None
                    and clothing_locked_name.strip().lower() == active_override
                )

                if (
                    not chosen_det
                    and is_clothing_locked
                    and detections
                    and override_matches_clothing
                ):
                    if (current_time - clothing_lock_timer) < CLOTHING_LOCK_TIMEOUT and target_clothing_hist is not None:
                        best_match_det = None
                        highest_score = -1.0
                        img_h, img_w = frame.shape[:2]

                        for det in detections:
                            bx1, by1, bx2, by2 = det["bbox"]
                            torso_ymin = int(by1 + (by2 - by1) * 0.2)
                            torso_ymax = int(by1 + (by2 - by1) * 0.7)
                            candidate_torso = frame[
                                max(0, torso_ymin) : min(img_h, torso_ymax),
                                max(0, bx1) : min(img_w, bx2),
                            ]

                            if candidate_torso.size > 0:
                                hsv_cand = cv2.cvtColor(candidate_torso, cv2.COLOR_RGB2HSV)
                                cand_hist = cv2.calcHist(
                                    [hsv_cand], [0, 1], None, [32, 32], [0, 180, 0, 256]
                                )
                                cv2.normalize(
                                    cand_hist,
                                    cand_hist,
                                    alpha=0,
                                    beta=1,
                                    norm_type=cv2.NORM_MINMAX,
                                )

                                score = cv2.compareHist(target_clothing_hist, cand_hist, 0)

                                if score > highest_score and score > 0.75:
                                    highest_score = score
                                    best_match_det = det

                        if best_match_det:
                            chosen_det = best_match_det
                            authorized_face_detected = True
                            if clothing_locked_name:
                                current_tracked_name = clothing_locked_name
                            cv2.putText(display_frame, f"LOCKED TO CLOTHING (Score: {highest_score:.2f})", (20, 120),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                    else:
                        target_clothing_hist = None
                        clothing_locked_name = None
                        is_clothing_locked = False

                elif not override_matches_clothing:
                    target_clothing_hist = None
                    clothing_locked_name = None
                    is_clothing_locked = False
                profiler.record("face_and_lock", time.perf_counter() - t0)

                t0 = time.perf_counter()

                # --- TARGET BUFFER & GRACE PERIOD LOGIC ---
                if chosen_det is not None:
                    missing_target_count = 0
                    last_valid_target = chosen_det
                    active_target = chosen_det
                else:
                    missing_target_count += 1
                    if missing_target_count < MAX_MISSING_FRAMES and last_valid_target is not None:
                        active_target = last_valid_target
                    else:
                        active_target = None
                        last_valid_target = None

                # --- VOICE QUEUE PROCESSING ---
                while True:
                    try:
                        v_cmd = voice_queue.get_nowait()
                        voice_queue.task_done()
                    except queue.Empty:
                        break

                    if time.time() < ignore_voice_until:
                        print(f"[VOICE IGNORED]: Discarded self-speech/echo command '{v_cmd}'")
                        continue

                    v_cmd_str = str(v_cmd).upper()

                    if v_cmd in ["VIP_FOLLOW", "GUEST", "CARE_TAKER", "PATIENT"]:
                        target_role_to_follow = v_cmd
                        print(f"[MODE] Target Follow Role Updated: {target_role_to_follow}")
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        _set_target_override_name(None)
                        _clear_follow_locks()
                        dist_ctrl.reset()

                    elif v_cmd == "AUTO":
                        print("[MODE] Clearing voice override. Returning to Auto Tracking.")
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        _set_target_override_name(None)
                        _clear_follow_locks()
                        dist_ctrl.reset()

                    elif "ELOY" in v_cmd_str:
                        if _get_target_override_name() != "eloy":
                            _set_target_override_name("eloy")
                            _clear_follow_locks()
                            dist_ctrl.reset()
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        print("[VOICE OVERRIDE]: Switching target directly to Eloy")

                    elif "CHIA" in v_cmd_str:
                        if _get_target_override_name() != "chia":
                            _set_target_override_name("chia")
                            _clear_follow_locks()
                            dist_ctrl.reset()
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        print("[VOICE OVERRIDE]: Switching target directly to Chia")

                    else:
                        print(f"[MODE] Temporary Voice Override Activated: {v_cmd}")
                        manual_voice_override = v_cmd
                        voice_override_expiry_time = time.time() + VOICE_OVERRIDE_DURATION

                if manual_voice_override is not None and time.time() >= voice_override_expiry_time:
                    manual_voice_override = None
                    voice_override_expiry_time = 0.0
                    print("[MODE] Temporary voice override expired -> returning to GESTURE / TRACKING_DISTANCE")

                BOX_ALPHA = 0.25
                BASE_FOLLOW_SPEED = 30
                latest_dist_cmd = "STOP"

                now_mono = time.monotonic()
                _raw_face_res = (
                    filtered_face_result
                    if isinstance(filtered_face_result, dict)
                    else None
                )
                face_matches = (
                    _raw_face_res.get("matches", [])
                    if isinstance(_raw_face_res, dict)
                    else []
                )
                if face_matches:
                    _vip_summary = [
                        (m.get("name"), m.get("confirmed"), m.get("authorized"))
                        for m in face_matches
                    ]
                    _vip_changed = _vip_summary != _last_vip_summary
                    if (_vip_changed and (now_mono - _last_vip_log_time >= 2.0)) or (
                        now_mono - _last_vip_log_time >= 5.0
                    ):
                        print(f"[VIP GATE] matches={_vip_summary}", flush=True)
                        _last_vip_log_time = now_mono
                    _last_vip_summary = _vip_summary

                vip_match = next(
                    (m for m in face_matches if m.get("authorized") and m.get("confirmed")),
                    None,
                )

                if vip_match is not None:
                    vip_grace_active = True
                    last_vip_lock_mono = now_mono
                    last_vip_bbox = list(vip_match["bbox"])
                    last_vip_name = str(vip_match["name"])
                    last_vip_role = str(vip_match["role"])
                    active_target = vip_match
                    authorized_face_detected = True
                    current_tracked_name = last_vip_name
                    target_role_to_follow = last_vip_role

                elif vip_grace_active and (now_mono - last_vip_lock_mono) <= VIP_LOCK_GRACE_SEC:
                    authorized_face_detected = True
                    current_tracked_name = last_vip_name or current_tracked_name
                    target_role_to_follow = last_vip_role or target_role_to_follow

                    if face_matches and last_vip_bbox is not None:
                        lx1, ly1, lx2, ly2 = last_vip_bbox
                        lcx, lcy = (lx1 + lx2) * 0.5, (ly1 + ly2) * 0.5
                        best_box_match = min(
                            face_matches,
                            key=lambda m: (
                                ((m["bbox"][0] + m["bbox"][2]) * 0.5 - lcx) ** 2
                                + ((m["bbox"][1] + m["bbox"][3]) * 0.5 - lcy) ** 2
                            ),
                        )
                        active_target = {
                            **best_box_match,
                            "name": last_vip_name,
                            "role": last_vip_role,
                            "confirmed": False,
                            "authorized": True,
                        }
                        last_vip_bbox = list(best_box_match["bbox"])

                    elif last_vip_bbox is not None:
                        active_target = {
                            "name": last_vip_name,
                            "role": last_vip_role,
                            "bbox": list(last_vip_bbox),
                            "confirmed": False,
                            "authorized": True,
                        }
                else:
                    vip_grace_active = False
                    last_vip_bbox = None

                # Always reset per-frame target variables before evaluating active_target
                target_bbox = None
                target_x_center = None

                if active_target is not None:
                    raw_xmin, raw_ymin, raw_xmax, raw_ymax = active_target["bbox"]

                    if not hasattr(servo_tracker, "smoothed_bbox") or servo_tracker.smoothed_bbox is None:
                        servo_tracker.smoothed_bbox = [
                            float(raw_xmin),
                            float(raw_ymin),
                            float(raw_xmax),
                            float(raw_ymax),
                        ]
                    else:
                        sx1, sy1, sx2, sy2 = servo_tracker.smoothed_bbox
                        servo_tracker.smoothed_bbox = [
                            BOX_ALPHA * raw_xmin + (1.0 - BOX_ALPHA) * sx1,
                            BOX_ALPHA * raw_ymin + (1.0 - BOX_ALPHA) * sy1,
                            BOX_ALPHA * raw_xmax + (1.0 - BOX_ALPHA) * sx2,
                            BOX_ALPHA * raw_ymax + (1.0 - BOX_ALPHA) * sy2,
                        ]

                    xmin, ymin, xmax, ymax = [int(round(v)) for v in servo_tracker.smoothed_bbox]
                    target_bbox = [xmin, ymin, xmax, ymax]
                    target_x_center = int((xmin + xmax) / 2)
                    person_box_height = float(max(0, ymax - ymin))

                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)
                    cv2.putText(display_frame, f"TRACKING: {str(current_tracked_name).upper()} [{target_role_to_follow}]", (xmin, max(40, ymin - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    latest_dist_cmd, _, _ = dist_ctrl.update(person_box_height, now=now_mono)
                    if manual_voice_override is None:
                        current_frame_command = latest_dist_cmd
                        current_source = "TRACKING_DISTANCE"
                else:
                    servo_tracker.smoothed_bbox = None
                    target_bbox = None
                    target_x_center = None
                    dist_ctrl.reset()
                    latest_dist_cmd = "STOP"

                img_h, img_w = display_frame.shape[:2]
                cv2.line(display_frame, (img_w // 2, 0), (img_w // 2, img_h), (100, 100, 100), 1)

                is_fall = False
                fall_status = "OK"
                fresh_bbox = chosen_det["bbox"] if chosen_det is not None else None

                if frame_count > 0 and frame_count % 15 == 0:
                    is_fall, fall_status = pose_fall_detector.process_crop_or_frame(
                        frame,
                        display_frame,
                        bbox=fresh_bbox,
                        is_rgb=True,
                        draw_overlay=True,
                    )
                    persisted_is_fall = is_fall
                    persisted_fall_status = fall_status

                elif frame_count > 0:
                    is_fall = persisted_is_fall if "persisted_is_fall" in locals() else False
                    fall_status = (
                        persisted_fall_status if "persisted_fall_status" in locals() else "OK"
                    )

                now = time.time()
                now_mono = time.monotonic()

                if is_fall:
                    fall_hold_until = max(fall_hold_until, now_mono + 2.0)

                    if not buzzer_triggered:
                        try:
                            motor.send_buzzer_command(state=True)
                            buzzer_triggered = True
                            print("[ALARM] Fall detected! Buzzer latched ON.", flush=True)
                        except Exception as e:
                            print(f"[ERROR] Failed to turn buzzer ON: {e}", flush=True)

                        if speaker:
                            say_async("Warning! Fall detected! Requesting assistance.")
                            last_fall_speech_time = now
                    elif (
                        speaker
                        and FALL_SPEECH_REPEAT_INTERVAL is not None
                        and (now - last_fall_speech_time) >= FALL_SPEECH_REPEAT_INTERVAL
                    ):
                        say_async("Warning! Fall detected! Requesting assistance.")
                        last_fall_speech_time = now
                else:
                    if buzzer_triggered:
                        try:
                            motor.send_buzzer_command(state=False)
                            buzzer_triggered = False
                            last_fall_speech_time = 0.0
                            print("[INFO] Fall cleared. Buzzer turned OFF and latch reset.", flush=True)
                        except Exception as e:
                            print(f"[ERROR] Failed to turn buzzer OFF: {e}", flush=True)

                if gesture_cmd:
                    cv2.putText(display_frame, f"GESTURE: {gesture_cmd} ({hand_side})", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "GESTURE: NONE", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                is_tracking_active = False

                # --- COMMAND DECISION HIERARCHY ---
                now_mono = time.monotonic()
                if is_fall:
                    fall_hold_until = max(fall_hold_until, now_mono + 2.0)
                    cv2.putText(display_frame, f"ALARM: {fall_status}", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    current_frame_command = "STOP"
                    current_source = "FALL_SAFETY"

                elif now_mono < fall_hold_until:
                    cv2.putText(display_frame, "HOLD: POST-FALL SAFETY", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    current_frame_command = "STOP"
                    current_source = "FALL_HOLD"

                elif current_terrain == "STAIRCASE":
                    current_frame_command = "STOP"
                    current_source = "TERRAIN"
                    cv2.putText(display_frame, "CAUTION: STAIRCASE AHEAD", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)

                elif manual_voice_override is not None and time.time() < voice_override_expiry_time:
                    current_frame_command = manual_voice_override
                    current_source = "VOICE"
                    cv2.putText(display_frame, f"VOICE OVERRIDE: {current_frame_command}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                else:
                    if manual_voice_override is not None:
                        manual_voice_override = None
                        dist_ctrl.reset()

                    if (
                        authorized_face_detected
                        and gesture_cmd
                        and gesture_cmd in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]
                    ):
                        current_frame_command = gesture_cmd
                        current_source = "GESTURE"

                        if current_frame_command != last_sent_command:
                            _gesture_phrases = {
                                "FORWARD": "Moving forward.",
                                "BACKWARD": "Moving backward.",
                                "LEFT": "Turning left.",
                                "RIGHT": "Turning right.",
                                "STOP": "Stopping robot.",
                            }
                            if gesture_cmd in _gesture_phrases:
                                say_async(_gesture_phrases[gesture_cmd])

                        cv2.putText(display_frame, f"GESTURE CMD: {gesture_cmd}", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    elif take_photo_triggered:
                        current_frame_command = "STOP"
                        current_source = "GESTURE"

                        display_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        file_timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        snapshot_img = frame.copy()

                        cv2.putText(snapshot_img, display_timestamp_str, (20, img_h - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        filename = f"robot_snapshot_{file_timestamp_str}.jpg"
                        cv2.imwrite(filename, cv2.cvtColor(snapshot_img, cv2.COLOR_RGB2BGR))
                        print(f"[SNAPSHOT] Photo successfully saved: {filename}")

                        say_async("Photo taken.")

                        cv2.putText(display_frame, "GESTURE CMD: TAKE_PHOTO", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    elif target_x_center is not None and target_bbox is not None:
                        missing_target_count = 0

                        ALPHA = 0.35
                        raw_y_center = int((target_bbox[1] + target_bbox[3]) / 2)
                        if (
                            not hasattr(servo_tracker, "smoothed_x")
                            or servo_tracker.smoothed_x is None
                            or not hasattr(servo_tracker, "smoothed_y")
                            or servo_tracker.smoothed_y is None
                        ):
                            servo_tracker.smoothed_x = float(target_x_center)
                            servo_tracker.smoothed_y = float(raw_y_center)
                        else:
                            servo_tracker.smoothed_x = (ALPHA * target_x_center) + (
                                (1.0 - ALPHA) * servo_tracker.smoothed_x
                            )
                            servo_tracker.smoothed_y = (ALPHA * raw_y_center) + (
                                (1.0 - ALPHA) * servo_tracker.smoothed_y
                            )

                        steering_error = int(round(servo_tracker.smoothed_x - (img_w // 2)))
                        vertical_error = int(round(servo_tracker.smoothed_y - (img_h // 2)))

                        DEADZONE_X = 24
                        DEADZONE_Y = 24

                        if abs(steering_error) < DEADZONE_X:
                            steering_error = 0
                        if abs(vertical_error) < DEADZONE_Y:
                            vertical_error = 0

                        MAX_STEP = 1.2
                        pan_delta = max(-MAX_STEP, min(MAX_STEP, steering_error * 0.012))
                        tilt_delta = max(-MAX_STEP, min(MAX_STEP, vertical_error * 0.012))
                        servo_tracker.pan_angle = max(
                            PAN_MIN, min(PAN_MAX, servo_tracker.pan_angle - pan_delta)
                        )
                        servo_tracker.tilt_angle = max(
                            TILT_MIN, min(TILT_MAX, servo_tracker.tilt_angle - tilt_delta)
                        )

                        if servo_tracker.pan_angle <= PAN_MIN or servo_tracker.pan_angle >= PAN_MAX:
                            steering_error = 0
                        if servo_tracker.tilt_angle <= TILT_MIN or servo_tracker.tilt_angle >= TILT_MAX:
                            vertical_error = 0

                        current_frame_command = latest_dist_cmd
                        current_source = "TRACKING_DISTANCE"
                        is_tracking_active = True

                        turn_boost = int(abs(steering_error) * 0.25)
                        if current_frame_command != "STOP":
                            drive_speed = max(
                                BASE_FOLLOW_SPEED, min(90, BASE_FOLLOW_SPEED + turn_boost)
                            )
                        elif abs(steering_error) > 0:
                            drive_speed = max(15, min(60, int(abs(steering_error) * 0.35)))
                        else:
                            drive_speed = 0

                        if current_time - last_serial_send_time >= SERIAL_SEND_INTERVAL:
                            try:
                                motor.send_hybrid_tracking_command(
                                    servo_tracker.pan_angle,
                                    servo_tracker.tilt_angle,
                                    steering_error,
                                    drive_speed,
                                )
                                last_serial_send_time = current_time
                            except Exception as e:
                                print(f"[SERIAL WARNING]: Failed to send tracking command: {e}")

                        cv2.putText(display_frame, "TRACKING ACTIVE", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

                    else:
                        servo_tracker.smoothed_x = None
                        servo_tracker.smoothed_y = None
                        is_tracking_active = False

                        current_frame_command = "STOP"
                        current_source = "HOLD"

                        if current_time - last_serial_send_time >= SERIAL_SEND_INTERVAL:
                            try:
                                motor.send_hybrid_tracking_command(
                                    servo_tracker.pan_angle,
                                    servo_tracker.tilt_angle,
                                    0,
                                    0,
                                )
                                last_serial_send_time = current_time
                            except Exception:
                                pass

                        cv2.putText(display_frame, "STATUS: TARGET LOST", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if current_frame_command:
                    current_pair = (current_source, current_frame_command)
                    last_pair = (last_sent_source, last_sent_command)
                    if current_pair != last_pair or (time.time() - last_send_time) > 0.5:
                        try:
                            motor.send_command(current_source, current_frame_command)
                        except Exception as e:
                            print(f"[SERIAL WARNING]: Failed to send command: {e}")
                        last_sent_source = current_source
                        last_sent_command = current_frame_command
                        last_send_time = time.time()

                cv2_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                cv2.putText(cv2_display_frame, f"Cmd: {current_frame_command} ({current_source})", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                _now_ui = time.monotonic()
                _key = 255
                if _now_ui - _last_ui_time >= 0.065:
                    _last_ui_time = _now_ui
                    _ui_t0 = time.perf_counter()
                    bgr_display = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                    _imshow_t0 = time.perf_counter()
                    cv2.imshow("Robot Vision", bgr_display)
                    _imshow_dt = time.perf_counter() - _imshow_t0

                    _wait_t0 = time.perf_counter()
                    _key = cv2.waitKey(1) & 0xFF
                    _wait_dt = time.perf_counter() - _wait_t0

                    _ui_total_dt = time.perf_counter() - _ui_t0
                    profiler.record("imshow_ui", _ui_total_dt)
                    if frame_count > 1 and _ui_total_dt > IMSHOW_STALL_SEC:
                        print(
                            f"[STALL:IMSHOW] total={_ui_total_dt * 1000.0:.1f} ms "
                            f"(imshow={_imshow_dt * 1000.0:.1f} ms, waitKey={_wait_dt * 1000.0:.1f} ms)",
                            flush=True,
                        )

                if _key == ord("q") or _key == 27:
                    break

                profiler.record("decision_and_io", time.perf_counter() - t0)
                profiler.maybe_report()
                time.sleep(0.005)

            except Exception as loop_err:
                print(f"[ERROR INSIDE LOOP]: {loop_err}")
                import traceback
                traceback.print_exc()

    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")

    finally:
        print("[TRACE] Cleaning up workers, peripherals, and camera...")
        try:
            if face_worker:
                face_worker.stop()
        except Exception as e:
            print(f"[WARNING] face_worker stop error: {e}")

        try:
            if gesture_worker:
                gesture_worker.stop()
        except Exception as e:
            print(f"[WARNING] gesture_worker stop error: {e}")

        try:
            if motor:
                motor.send_command("HOLD", "STOP")
                motor.close()
        except Exception as e:
            print(f"[WARNING] motor cleanup error: {e}")

        try:
            if speaker:
                speaker.stop()
        except Exception as e:
            print(f"[WARNING] speaker stop error: {e}")

        try:
            if voice_rec:
                if hasattr(voice_rec, "stop"):
                    voice_rec.stop()
                elif hasattr(voice_rec, "close"):
                    voice_rec.close()
        except Exception as e:
            print(f"[WARNING] voice_rec stop error: {e}")

        try:
            if gesture_rec and hasattr(gesture_rec, "close"):
                gesture_rec.close()
        except Exception as e:
            print(f"[WARNING] gesture_rec close error: {e}")

        try:
            if pose_fall_detector and hasattr(pose_fall_detector, "close"):
                pose_fall_detector.close()
        except Exception as e:
            print(f"[WARNING] pose_fall_detector close error: {e}")

        try:
            if hailo_embedder and hasattr(hailo_embedder, "close"):
                hailo_embedder.close()
        except Exception as e:
            print(f"[WARNING] hailo_embedder close error: {e}")

        try:
            if "cam" in locals() and cam:
                cam.running = False
                cam.close()
        except Exception as e:
            print(f"[WARNING] camera close error: {e}")

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        print("[INFO] System shutdown complete.")


if __name__ == "__main__":
    main()

