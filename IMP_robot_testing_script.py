#!/usr/bin/env python3
import os
if os.getenv("INVOCATION_ID") or not os.getenv("DISPLAY"):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

import sys

# --- Suppress Log Noise ---
os.environ["GLOG_minloglevel"] = "2"
os.environ["LIBCAMERA_LOG_LEVELS"] = "3"
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import time
import json
import cv2
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
from datetime import datetime
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from collections import deque, defaultdict, Counter
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vosk import Model, KaldiRecognizer
from typing import Any, Callable, Optional
from hailo_platform import FormatType, VDevice

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

        print(
            f"[TRACE] IMX500 camera started ({self.frame_width}x{self.frame_height}, "
            f"target {target_fps} FPS)."
        )

    @property
    def fps(self):
        """Return the latest measured rolling FPS."""
        return self._current_fps

    def _update_fps(self):
        now = time.monotonic()
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
            frame = request.make_array("main")
            metadata = request.get_metadata() or {}

            if frame is not None and hasattr(frame, "shape") and len(frame.shape) >= 2:
                self.frame_height, self.frame_width = int(frame.shape[0]), int(frame.shape[1])

            self._update_fps()
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
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.running = True
        self.is_speaking_now = False
        self.thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.thread.start()

    def _speak_loop(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                self.is_speaking_now = True
                print(f"[ROBOT SPEAKS]: {text}")
                try:
                    subprocess.run(["espeak-ng", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"[SPEAKER ERROR] Could not play audio: {e}")
                finally:
                    self.is_speaking_now = False
                    self.speech_queue.task_done()
            except queue.Empty:
                continue

    def say(self, text):
        self.speech_queue.put(text)

    def wait_until_done(self, timeout=10.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.speech_queue.unfinished_tasks == 0 and not self.is_speaking_now:
                return
            time.sleep(0.02)

    def stop(self):
        self.running = False

# -----------------------------------------------------------------------------
# Terrain & Pavement Classifier
# -----------------------------------------------------------------------------
class TerrainClassifier:
    def __init__(self, brightness_thresh=110.0, edge_density_thresh=0.08, stair_edge_thresh=0.18):
        self.current_terrain = "ROAD"
        self.brightness_thresh = brightness_thresh
        self.edge_density_thresh = edge_density_thresh
        self.stair_edge_thresh = stair_edge_thresh  # Higher threshold for repeated horizontal steps

    def analyze_surface(self, frame, display_frame):
        h, w, _ = frame.shape
        roi = frame[int(h * 0.6):h, 0:w]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        avg_brightness = float(np.mean(gray_roi))
        edges = cv2.Canny(gray_roi, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)

        # --- Enhanced Heuristic Classification ---
        if edge_density > self.stair_edge_thresh:
            self.current_terrain = "STAIRCASE"
        elif edge_density > self.edge_density_thresh:
            self.current_terrain = "CURB_EDGE"
        elif avg_brightness > self.brightness_thresh:
            self.current_terrain = "PAVEMENT"
        else:
            self.current_terrain = "ROAD"

        # Visual debug overlays
        cv2.rectangle(display_frame, (0, int(h * 0.6)), (w, h), (255, 100, 0), 2)
        cv2.putText(display_frame, f"TERRAIN: {self.current_terrain}", (20, 190),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(display_frame, f"Bright: {int(avg_brightness)} | Edge: {edge_density:.2f}", (20, 220),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

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

    def __init__(self, hef_path: str):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"Hailo HEF model not found: {hef_path}")

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

    def get_embedding(
        self, face_crop: np.ndarray, *, input_is_rgb: bool = True
    ) -> Optional[np.ndarray]:
        try:
            if face_crop is None or face_crop.size == 0:
                return None

            resized = cv2.resize(
                face_crop,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            rgb_frame = (
                resized if input_is_rgb else cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            )
            input_data = np.expand_dims(
                np.ascontiguousarray(rgb_frame, dtype=np.uint8), axis=0
            )

            bindings = self.configured_infer_model.create_bindings()
            output_tensor = self.infer_model.output()
            output_buffer = np.empty(output_tensor.shape, dtype=np.float32)

            bindings.input().set_buffer(input_data)
            bindings.output().set_buffer(output_buffer)

            self.configured_infer_model.run([bindings], timeout=5000)
            return l2_normalize(output_buffer)
        except Exception as e:
            print(f"[LIVE EMBED ERROR]: {e}")
            return None

    def close(self) -> None:
        self.configured_infer_model = None
        self.infer_model = None
        if getattr(self, "target", None) is not None:
            release_fn = getattr(self.target, "release", None)
            if callable(release_fn):
                release_fn()
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
        self.reload_database()

    def reload_database(self) -> None:
        self.known_embeddings.clear()
        self.known_names.clear()
        self.known_roles.clear()

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

            print(f"[INFO] Loaded {loaded} profiles from face database.")
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
        finally:
            if conn is not None:
                conn.close()

    def match_face(
        self, query_vector: np.ndarray, max_distance: float = 0.65
    ) -> tuple[str, str, float]:
        if not self.known_embeddings or query_vector is None:
            return "Unknown", "none", 1.0

        q = _normalize_l2(query_vector)
        if q.size == 0:
            return "Unknown", "none", 1.0

        best_match_idx: int | None = None
        min_dist = float("inf")

        for idx, known_vec in enumerate(self.known_embeddings):
            if known_vec.shape != q.shape:
                continue
            dist = float(max(0.0, 1.0 - np.dot(q, known_vec)))
            if dist < min_dist:
                min_dist = dist
                best_match_idx = idx

        if best_match_idx is None:
            return "Unknown", "none", 1.0

        if min_dist <= max_distance:
            return (
                self.known_names[best_match_idx],
                self.known_roles[best_match_idx],
                min_dist,
            )

        return "Unknown", "none", min_dist
    
# -----------------------------------------------------------------------------
# Fall Detection using MediaPipe Pose Landmarker using key body joints like shoulder
# and hip combine with Multi-Criteria Geometric & Dynamic Rules
# -----------------------------------------------------------------------------
class PoseFallDetector:
    def __init__(self, frame_height=480, frame_width=640):
        self.frame_height = frame_height
        self.frame_width = frame_width
        
        # State tracking variables for multi-criteria logic
        self.prev_hip_y = None
        self.prev_time = None
        self.lying_start_time = None
        self.fall_detected = False
        self.fall_timestamp = 0
        self.cooldown_sec = 3.0
        
        # Initialize MediaPipe Pose Landmarker
        pose_task_path = os.path.join(BASE_DIR, "pose_landmarker.task") if 'BASE_DIR' in globals() else "pose_landmarker.task"
        if os.path.exists(pose_task_path):
            base_options = python.BaseOptions(model_asset_path=pose_task_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_tracking_confidence=0.3
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        else:
            self.pose_detector = None
            print(f"[WARNING] Pose landmark task missing at {pose_task_path}. Fall detection running in fallback mode.")

    def process_crop_or_frame(self, frame, display_frame, bbox=None):
        current_time = time.time()
        status_text = "NORMAL"
        is_fallen = False

        if self.pose_detector:
            # Run pose detection on the full frame or cropped region safely
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                # Ensure valid crop boundaries
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(self.frame_width, xmax), min(self.frame_height, ymax)
                person_roi = frame[ymin:ymax, xmin:xmax]
            else:
                person_roi = frame
                xmin, ymin = 0, 0

            if person_roi.size > 0:
                try:
                    rgb_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_roi)
                    results = self.pose_detector.detect(mp_image)

                    if results.pose_landmarks:
                        landmarks = results.pose_landmarks[0]
                        
                        # MediaPipe indices: 11=Left Shoulder, 12=Right Shoulder, 23=Left Hip, 24=Right Hip
                        left_shoulder = landmarks[11]
                        right_shoulder = landmarks[12]
                        left_hip = landmarks[23]
                        right_hip = landmarks[24]

                        roi_h, roi_w, _ = person_roi.shape
                        shoulder_x = xmin + int(((left_shoulder.x + right_shoulder.x) / 2) * roi_w)
                        shoulder_y = ymin + int(((left_shoulder.y + right_shoulder.y) / 2) * roi_h)
                        
                        hip_x = xmin + int(((left_hip.x + right_hip.x) / 2) * roi_w)
                        hip_y = ymin + int(((left_hip.y + right_hip.y) / 2) * roi_h)

                        # Draw skeleton lines on display frame
                        cv2.circle(display_frame, (shoulder_x, shoulder_y), 4, (0, 0, 255), -1)
                        cv2.circle(display_frame, (hip_x, hip_y), 4, (255, 0, 0), -1)
                        cv2.line(display_frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (0, 255, 255), 2)

                        # --- RULE 1: Torso Angle Check ---
                        dx = hip_x - shoulder_x
                        dy = hip_y - shoulder_y
                        torso_angle = np.degrees(np.arctan2(abs(dy), abs(dx + 1e-6)))
                        is_horizontal = torso_angle < 35.0

                        # --- RULE 2: Descent Velocity (Motion Dynamic) ---
                        rapid_descent = False
                        if self.prev_hip_y is not None and self.prev_time is not None:
                            dt = current_time - self.prev_time
                            if dt > 0:
                                dy_hip = hip_y - self.prev_hip_y  # Positive means moving downwards in pixels
                                velocity = dy_hip / dt            # Pixels per second
                                if velocity > 180.0:              # Sudden drop threshold
                                    rapid_descent = True
                        
                        self.prev_hip_y = hip_y
                        self.prev_time = current_time

                        # --- RULE 3: Bounding Box Aspect Ratio Check ---
                        box_flat = False
                        if bbox is not None:
                            box_w = xmax - xmin
                            box_h = ymax - ymin
                            if box_w >= box_h * 0.9:
                                box_flat = True

                        # --- Multi-Criteria Logic Combination ---
                        is_fall_candidate = is_horizontal and (rapid_descent or box_flat)

                        if is_fall_candidate:
                            if self.lying_start_time is None:
                                self.lying_start_time = current_time
                            
                            # --- RULE 4: Prolonged Inactivity / Lying Duration ---
                            lying_duration = current_time - self.lying_start_time
                            if lying_duration >= 2.0:  # Must remain down for at least 2 seconds
                                self.fall_detected = True
                                self.fall_timestamp = current_time
                                status_text = "FALL DETECTED!"
                            else:
                                status_text = f"CHECKING FALL ({lying_duration:.1f}s)..."
                        else:
                            # Reset timer if the person stands back up or moves normally
                            self.lying_start_time = None

                except Exception as e:
                    # Catch any MediaPipe runtime bounds error gracefully
                    pass

        # Check cooldown state
        if self.fall_detected:
            if current_time - self.fall_timestamp < self.cooldown_sec:
                is_fallen = True
                status_text = "FALL DETECTED!"
            else:
                self.fall_detected = False

        return is_fallen, status_text

    def _check_cooldown(self):
        if self.fall_detected:
            if time.time() - self.fall_timestamp < self.cooldown_sec:
                return True, "FALL DETECTED!"
            else:
                self.fall_detected = False
        return False, "NORMAL"

# -----------------------------------------------------------------------------
# MediaPipe Hand Gesture Recognizer
# -----------------------------------------------------------------------------
class GestureRecognizer:
    def __init__(self, task_path=HAND_TASK_PATH):
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
        self.last_wrist_y = None
        self.last_landmarks = None

    def process_frame(self, frame, display_frame):
        rgb_frame = frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
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

                for connection in HAND_CONNECTIONS:
                    p1 = (int(lm[connection[0]].x * FRAME_WIDTH), int(lm[connection[0]].y * FRAME_HEIGHT))
                    p2 = (int(lm[connection[1]].x * FRAME_WIDTH), int(lm[connection[1]].y * FRAME_HEIGHT))
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

                print(f"[GESTURE DEBUG] Hand: {hand_side} | Wrist Y: {wrist_y:.2f} | Index Up: {index_open}")
                print(f"[DEBUG GESTURE] index_open: {index_open} | middle_open: {middle_open} | ring_open: {ring_open} | pinky_open: {pinky_open}")

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
    
# -----------------------------------------------------------------------------
# Offline Voice Recognizer (Vosk + PyAudio)
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
        self.esp_audio_port = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_A4:CB:8F:20:E0:3C-if00"
        self.esp_baud_rate = 921600

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
        self.voice_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.voice_thread.start()
        print("[INIT] ESP32 Voice listener background thread explicitly triggered.")

    def send_text_command_to_arduino(self, source, command):
        """
        Sends a text command packet to the Arduino Giga using header 0x03.
        Format: [0x03] + "SOURCE:COMMAND\n"
        Example: [0x03] + "VOICE:STOP\n"
        """
        try:
            if hasattr(self, "esp_audio_serial") and self.esp_audio_serial and self.esp_audio_serial.is_open:
                payload = bytes([0x03]) + f"{source}:{command}\n".encode("utf-8")
                self.esp_audio_serial.write(payload)
        except Exception as e:
            print(f"[SERIAL TEXT ERROR]: {e}")

    def set_speaking(self, speaking: bool, cooldown: float | None = None):
        """Non-blocking TTS mute gate with post-speech reverberation hold."""
        hold = self.post_speech_hold if cooldown is None else max(0.0, float(cooldown))
        now = time.monotonic()
        with self._lock:
            if speaking:
                self._speaking_active = True
                self._mute_until = max(self._mute_until, now + hold)
                self.audio_buffer.clear()
            else:
                self._speaking_active = False
                self._mute_until = max(self._mute_until, now + hold)
                self._needs_flush = True
                self.audio_buffer.clear()

    def _is_muted_and_should_flush(self, now: float) -> tuple[bool, bool]:
        with self._lock:
            muted = self._speaking_active or (now < self._mute_until)
            flush = False
            if not muted and self._needs_flush:
                flush = True
                self._needs_flush = False
                self.audio_buffer.clear()
            return muted, flush

    def say_muted(self, text: str, extra_hold: float = 0.0, cooldown: float | None = None):
        """Speaks via self.speaker while keeping Vosk muted during playback + cooldown."""
        if not self.speaker or not text:
            return
        hold = self.post_speech_hold if cooldown is None else float(cooldown)
        total_hold = max(hold, float(extra_hold))
        self.set_speaking(True, cooldown=total_hold)
        try:
            self.speaker.say(text)
        finally:
            self.set_speaking(False, cooldown=total_hold)

    def _init_voice_encoder(self, vip_ref_paths=None):
        """Loads multi-speaker Resemblyzer embeddings from pickle."""
        try:
            model_path = "authorized_speakers.pkl"
            with open(model_path, "rb") as f:
                self.speaker_embeddings = pickle.load(f)

            if self.speaker_embeddings:
                print(f"[VOICE] Loaded authorized speaker embeddings successfully: {list(self.speaker_embeddings.keys())}")
            else:
                print(f"[VOICE WARNING] '{model_path}' loaded, but it is empty.")
                self.speaker_embeddings = {}

        except Exception as e:
            print(f"[VOICE WARNING] Could not load authorized speaker embeddings: {e}")
            self.speaker_embeddings = {}

        # Initialize Resemblyzer Voice Encoder
        try:
            from resemblyzer import VoiceEncoder
            self.voice_encoder = VoiceEncoder()
        except Exception as e:
            print(f"[ERROR] Failed to initialize Resemblyzer VoiceEncoder: {e}")
            self.voice_encoder = None

    def _verify_speaker(self, raw_audio, threshold=None, min_samples=4800):
        """Verifies incoming 16kHz 16-bit mono PCM audio against enrolled speaker embeddings."""
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
            test_norm = np.linalg.norm(test_embedding)
            if test_norm == 0.0:
                return False, "UNKNOWN"

            best_score = -1.0
            matched_speaker = "UNKNOWN"

            for name, ref_embedding in self.speaker_embeddings.items():
                ref_arr = np.asarray(ref_embedding, dtype=np.float32)
                ref_norm = np.linalg.norm(ref_arr)
                if ref_norm == 0.0:
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
        print("[VOICE DEBUG] _listen_loop thread started...")
        try:
            if not self.esp_audio_serial:
                print("[VOICE ERROR] ESP32 serial stream is not active. Exiting loop.")
                return

            print("[VOICE] Streaming raw PCM bytes from ESP32-C3 at 16kHz...")

            def register_cmd(cmd_string, speech_reply):
                print(f"[VOICE BROADCAST]: {speech_reply}")
                self.command_queue.put(cmd_string)
                self.say_muted(speech_reply, cooldown=self.post_speech_hold)

            while self.is_running:
                data = self.esp_audio_serial.read(512)
                if not data or len(data) < 2:
                    continue

                if len(data) % 2 != 0:
                    data = data[:-1]
                if not data:
                    continue

                now_mono = time.monotonic()
                muted, should_flush = self._is_muted_and_should_flush(now_mono)

                if should_flush:
                    self.recognizer.Reset()
                    self.audio_buffer.clear()

                if muted:
                    # Drain serial bytes while speaking/cooling down, but drop them
                    continue

                # Single unified RMS calculation
                audio_array = np.frombuffer(data, dtype=np.int16)
                rms = float(np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))) if audio_array.size > 0 else 0.0

                # Keep audio buffer populated for speaker verification
                self.audio_buffer.append(data)

                # Pass raw bytes directly to Vosk recognizer
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    utterance_pcm = b"".join(self.audio_buffer)
                    self.audio_buffer.clear()

                    print(f"\n[VOSK RAW RESULT JSON]: {result}")

                    if text:
                        print(f"\n[VOICE RECOGNIZED RAW TEXT]: '{text}'")
                        words = text.split()
                        matched_entry = self._select_command(words)

                        if not matched_entry:
                            print(f"[VOICE IGNORED] '{text}' is not in command allowlist.")
                            self.recognizer.Reset()
                            continue

                        req_thresh = self.short_word_threshold if len(words) == 1 else self.speaker_threshold
                        is_verified, speaker_name = self._verify_speaker(
                            utterance_pcm, threshold=req_thresh
                        )
                        print(
                            f"[VOICE DEBUG] Speaker Verification Result: {is_verified} "
                            f"({speaker_name}) for '{text}' (RMS: {rms:.1f})"
                        )

                        if not is_verified:
                            print(f"[VOICE REJECTED] Command '{text}' ignored by speaker verification.")
                            if len(words) >= 2 and (now_mono - self._last_reject_tts_time) >= 4.0:
                                self._last_reject_tts_time = now_mono
                                self.say_muted("Command ignored. Unrecognized voice.", cooldown=0.6)
                            self.recognizer.Reset()
                            continue

                        print(f"[VOICE PASSED VERIFICATION]: '{text}' (User: {speaker_name})")
                        cmd_string, speech_reply = matched_entry
                        register_cmd(cmd_string, speech_reply)
                    else:
                        self.recognizer.Reset()
                else:
                    partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        print(f"[VOICE PARTIAL]: '{partial}'          ", end="\r", flush=True)

        except Exception as e:
            print(f"[VOICE FATAL ERROR CRASH] {e}")
            import traceback
            traceback.print_exc()

# -----------------------------------------------------------------------------
# Motor Controller (with Auto-Reconnect & Thread-Safe Serial Framing)
# -----------------------------------------------------------------------------
class MotorController:
    def __init__(self, port, baud, debug=False):
        self.port = port
        self.baud = baud
        self.debug = debug
        self.ser = None
        self._io_lock = threading.Lock()
        self._last_reconnect_attempt = 0.0
        self._reconnect_interval = 2.0
        self.connect()

    def connect(self):
        with self._io_lock:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.close()
            except Exception:
                pass

            try:
                self.ser = serial.Serial(
                    self.port,
                    self.baud,
                    timeout=1,
                    write_timeout=0.05,
                )
                time.sleep(1)
                print(f"[INFO] Serial connected to {self.port}")
            except Exception as e:
                self.ser = None
                print(f"[WARNING] Serial connection failed: {e}")

    def _ensure_connection(self):
        """Non-blocking rate-limited reconnect check (must be called with _io_lock held)."""
        if self.ser is not None and self.ser.is_open:
            return True

        now = time.time()
        if (now - self._last_reconnect_attempt) < self._reconnect_interval:
            return False

        self._last_reconnect_attempt = now
        try:
            self.ser = serial.Serial(
                self.port,
                self.baud,
                timeout=1,
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

# ----------------------------------------------------------------
# Tracking and Following
# ----------------------------------------------------------------
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

        self.smoothed_height: float | None = None
        self.active_cmd: str = "STOP"
        self._candidate_cmd: str = "STOP"
        self._candidate_since: float = time.monotonic()

    def reset(self, initial_cmd: str = "STOP"):
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
        if now is None:
            now = time.monotonic()

        if raw_box_height is None or raw_box_height <= 0:
            changed = self.active_cmd != "STOP"
            self.active_cmd = "STOP"
            self._candidate_cmd = "STOP"
            self._candidate_since = now
            return self.active_cmd, self.smoothed_height, changed

        h_raw = float(raw_box_height)
        if self.smoothed_height is None:
            self.smoothed_height = h_raw
        else:
            self.smoothed_height = (self.alpha * h_raw) + ((1.0 - self.alpha) * self.smoothed_height)
        desired_cmd = self._classify_with_hysteresis(self.smoothed_height)

        if desired_cmd != self._candidate_cmd:
            self._candidate_cmd = desired_cmd
            self._candidate_since = now

        changed = False
        if self._candidate_cmd != self.active_cmd:
            if (now - self._candidate_since) >= self.min_dwell:
                self.active_cmd = self._candidate_cmd
                changed = True

        return self.active_cmd, self.smoothed_height, changed

# --------------------------------------------------
# Measure each loop iteration
# --------------------------------------------------
class LoopProfiler:
    def __init__(self, interval=5.0):
        self.interval = interval
        self.totals = defaultdict(float)
        self.counts = defaultdict(int)
        self.last_report = time.monotonic()

    def record(self, stage, elapsed_sec):
        self.totals[stage] += elapsed_sec
        self.counts[stage] += 1

    def maybe_report(self):
        now = time.monotonic()
        if now - self.last_report >= self.interval:
            parts = []
            for stage, total in self.totals.items():
                cnt = self.counts[stage]
                avg_ms = (total / cnt * 1000.0) if cnt else 0.0
                parts.append(f"{stage}={avg_ms:.1f}ms")
            print("[PROFILE] " + " | ".join(parts))
            self.totals.clear()
            self.counts.clear()
            self.last_report = now

# ---------------------------------------------------------------
# Decouple Heavy Vision Models with "Latest-Frame" Worker Threads
# ----------------------------------------------------------------
class LatestFrameWorker:
    """Runs a slow vision callable asynchronously on the freshest frame."""

    def __init__(self, name, infer_fn, log_every_sec=5.0):
        self.name = name
        self.infer_fn = infer_fn
        self.log_every_sec = log_every_sec
        self._in_q = queue.Queue(maxsize=1)
        self._lock = threading.Lock()
        self._latest_result = None
        self._stop_event = threading.Event()

        self._infer_sum_ms = 0.0
        self._age_sum_ms = 0.0
        self._count = 0
        self._last_log = time.monotonic()

        self._thread = threading.Thread(
            target=self._run, name=f"worker-{name}", daemon=True
        )
        self._thread.start()

    def submit(self, frame, capture_ts=None):
        if frame is None:
            return
        if capture_ts is None:
            capture_ts = time.monotonic()

        payload = (frame.copy(), capture_ts)
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

                if done_mono - self._last_log >= self.log_every_sec and self._count > 0:
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
        self._thread.join(timeout=1.0)

# ---------------------------------------------------
# Helper functions
# ---------------------------------------------------
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
) -> Callable[[np.ndarray], dict[str, Any]]:
    """Create a per-frame face detection, embedding match, and authorization worker."""
    streak_limit = max(1, int(required_streak))
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

        h, w = frame.shape[:2]
        if h <= 0 or w <= 0:
            return _empty_result(ts)

        det_input = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) if frame_is_rgb else frame
        detector.setInputSize((w, h))
        _, faces = detector.detect(det_input)

        if faces is None or len(faces) == 0:
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

            fx, fy, fw, fh = float(face[0]), float(face[1]), float(face[2]), float(face[3])
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

            now = time.monotonic()
            if (
                name != log_state["last_name"]
                or (now - log_state["last_time"]) >= 1.0
            ):
                status = "MATCH" if is_known else "REJECT"
                lock_tag = "LOCKED" if confirmed_lock else f"STREAK {streak_count}/{streak_limit}"
                dist_str = f"{dist:.4f}" if isinstance(dist, (int, float)) else "N/A"
                print(
                    f"[FACE DEBUG] {status} ({lock_tag}) | name={name} ({role}) | "
                    f"dist={dist_str} | max_distance={max_distance:.2f}",
                    flush=True,
                )
                log_state["last_time"] = now
                log_state["last_name"] = name

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
# ==========================================================================
def make_gesture_worker_fn(recognizer):
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

        raw_result = recognizer.process_frame(frame, frame.copy())
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
# ==========================================================================
def extract_face_crop(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    pad_ratio: float = 0.0,
) -> np.ndarray:
    """Extract a clamped face bounding-box crop from frame with optional padding."""
    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 2:
        return np.empty((0, 0, 3), dtype=np.uint8)

    h, w = frame.shape[:2]
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

    return frame[y1:y2, x1:x2]

# ----------------------------------------
# Target lock
# ----------------------------------------
class FaceTargetLocker:
    """Maintains a stable follow lock from face worker outputs using confirmed & authorized."""

    def __init__(self, hold_timeout_s: float = 0.6) -> None:
        self.hold_timeout_s = max(0.0, float(hold_timeout_s))
        self.locked_name: Optional[str] = None
        self.locked_role: Optional[str] = None
        self.locked_bbox: Optional[list[int]] = None
        self.locked_dist: Optional[float] = None
        self.last_seen_ts: float = 0.0

    def clear(self) -> None:
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

        if eligible:
            # Prefer lowest cosine distance, breaking ties by largest bbox area.
            def _rank(item: dict[str, Any]) -> tuple[float, float]:
                dist_val = item.get("dist")
                d = float(dist_val) if isinstance(dist_val, (int, float)) else 999.0
                bbox = item.get("bbox") or [0, 0, 0, 0]
                area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
                return (d, -float(area))

            best = min(eligible, key=_rank)
            self.locked_name = str(best["name"])
            self.locked_role = str(best.get("role", "none"))
            self.locked_bbox = best.get("bbox")
            self.locked_dist = (
                float(best["dist"]) if isinstance(best.get("dist"), (int, float)) else None
            )
            self.last_seen_ts = ts
            return {
                "locked": True,
                "name": self.locked_name,
                "role": self.locked_role,
                "bbox": self.locked_bbox,
                "dist": self.locked_dist,
                "state": "LOCKED",
            }

        # No newly confirmed+authorized match this cycle: check grace window.
        if self.locked_name is not None:
            if desired_name is not None and self.locked_name.lower() != desired_name:
                self.clear()
            elif (ts - self.last_seen_ts) <= self.hold_timeout_s:
                return {
                    "locked": True,
                    "name": self.locked_name,
                    "role": self.locked_role,
                    "bbox": self.locked_bbox,
                    "dist": self.locked_dist,
                    "state": "HOLDING",
                }
            else:
                self.clear()

        return {
            "locked": False,
            "name": None,
            "role": None,
            "bbox": None,
            "dist": None,
            "state": "UNLOCKED",
        }
    
# -------------------------------------------------------------------
# smooths noisy per-frame hand gesture outputs
# -------------------------------------------------------------------
class GestureDebouncer:
    """Debounces noisy per-frame gesture predictions with sliding-window consensus."""

    def __init__(
        self,
        window_size=5,
        min_votes=3,
        max_age_sec=0.45,
        action_thresholds=None,
    ):
        self.window = deque(maxlen=window_size)
        self.min_votes = min_votes
        self.max_age_sec = max_age_sec
        self.action_thresholds = action_thresholds or {"TAKE_PHOTO": window_size}
        self.confirmed_cmd = None
        self._last_seen_done_ts = None

    def update(self, gesture_res, now_mono=None):
        """
        Returns:
            (active_cmd, newly_confirmed_event_cmd, is_stale, latest_raw_cmd)
        """
        if now_mono is None:
            now_mono = time.monotonic()

        if not gesture_res:
            return None, None, True, None

        done_ts = gesture_res.get("done_ts")
        capture_ts = gesture_res.get("capture_ts", done_ts)
        age_sec = (now_mono - capture_ts) if capture_ts is not None else 0.0
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
    last_gesture_time = 0.0
    persisted_gesture = "STOP"

    print("[TRACE] Initializing IMX500 camera pipeline...")
    try:
        cam = IMX500Camera(model_path=IMX500_MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] IMX500 failed to bind: {e}")
        return

    time.sleep(1.0)

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
        print(f"[WARNING] YuNet model file missing at {YUNET_MODEL_PATH}. Falling back to bounding-box estimation.")

    pose_fall_detector = PoseFallDetector(frame_height=FRAME_HEIGHT)
    terrain_classifier = TerrainClassifier()

    print("[TRACE] Initializing GestureRecognizer...")
    try:
        gesture_rec = GestureRecognizer()
        print("[INFO] Hand Gesture Recognizer initialized.")
    except Exception as e:
        print(f"[WARNING] Gesture module failed: {e}")

    print("[TRACE] Initializing OfflineVoiceRecognizer...")
    try:
        voice_rec = OfflineVoiceRecognizer(command_queue=voice_queue, model_path=MODEL_PATH, speaker=speaker)
        print("[INFO] Offline Voice Control initialized.")
    except Exception as e:
        print(f"[WARNING] Voice module failed: {e}")

    print("[TRACE] Voice module block cleared. Setting up loop variables...")

    manual_voice_override = None
    voice_override_expiry_time = 0.0
    VOICE_OVERRIDE_DURATION = 4.0
    target_role_to_follow = "VIP_FOLLOW"
    target_name_override = None

    def _get_target_override_name() -> Optional[str]:
        return target_name_override

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

    target_clothing_hist = None
    clothing_locked_name: Optional[str] = None
    is_clothing_locked = False
    CLOTHING_LOCK_TIMEOUT = 2.0
    clothing_lock_timer = 0.0
    buzzer_triggered = False
    last_fall_speech_time = 0.0
    FALL_SPEECH_REPEAT_INTERVAL = 10.0
    current_tracked_name = "Unknown"
    IDEAL_BOX_HEIGHT = 300.0

    def _clear_follow_locks() -> None:
        nonlocal target_clothing_hist, clothing_locked_name, is_clothing_locked, clothing_lock_timer
        face_locker.clear()
        target_clothing_hist = None
        clothing_locked_name = None
        is_clothing_locked = False
        clothing_lock_timer = 0.0

    dist_ctrl = DistanceTrackingController(
        ideal_box_height=IDEAL_BOX_HEIGHT,
        height_tolerance=65.0,
        exit_margin=15.0,
        box_height_alpha=0.18,
        dist_cmd_min_dwell=0.75,
    )
    pending_voice_cooldown_until = None
    VOICE_POST_SPEECH_COOLDOWN = 1.2
    ignore_voice_until = 0.0
    VOICE_SPEECH_IGNORE_DURATION = 9.0
    

    FACE_STRIDE = 4
    GESTURE_STRIDE = 5
    MAX_FACE_RESULT_AGE = 1.0
    MAX_GESTURE_RESULT_AGE = 0.8

    profiler = LoopProfiler(interval=5.0)
    face_locker = FaceTargetLocker(hold_timeout_s=0.6)
    face_worker = LatestFrameWorker(
        "face",
        make_face_worker_fn(
            face_detector,
            hailo_embedder,
            face_db,
            get_override_name=_get_target_override_name,
            max_distance=0.72,
            required_streak=2,
            frame_is_rgb=True,
        ),
    )
    gesture_worker = LatestFrameWorker("gesture", make_gesture_worker_fn(gesture_rec))
    gesture_debouncer = GestureDebouncer(
        window_size=5,
        min_votes=3,
        max_age_sec=MAX_GESTURE_RESULT_AGE,
        action_thresholds={"TAKE_PHOTO": 5},
    )

    try:
        print("[DEBUG] Entering main loop now...")
        while True:
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"[STATUS] Running... Frame: {frame_count} | Command: {current_frame_command} | Source: {current_source}")
            try:
                t0 = time.perf_counter()
                frame, metadata = cam.read()
                capture_ts = time.monotonic()
                profiler.record("cam_read", time.perf_counter() - t0)

                if frame is None:
                    profiler.maybe_report()
                    time.sleep(0.005)
                    continue

                now_ts = time.time()
                current_time = now_ts

                t0 = time.perf_counter()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = frame.copy()
                target_bbox = None
                target_x_center = None

                if frame_count % 3 == 0:
                    current_terrain = terrain_classifier.analyze_surface(frame, display_frame)

                detections = cam.parse_detections(metadata, CONFIDENCE_THRESHOLD, TARGET_CLASS_ID)
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
                    and (time.time() - last_snapshot_time) >= snapshot_cooldown
                ):
                    take_photo_triggered = True
                    photo_armed = False
                    last_snapshot_time = time.time()

                def _pt_xy(pt):
                    if hasattr(pt, "x") and hasattr(pt, "y"):
                        return float(pt.x), float(pt.y)
                    if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                        return float(pt[0]), float(pt[1])
                    raise TypeError(f"Unsupported landmark point type: {type(pt).__name__} ({dir(pt)})")

                hand_side = latest_gesture.get("hand_side", "Unknown") if latest_gesture else "Unknown"
                lm = (
                    latest_gesture.get("landmarks")
                    if (latest_gesture and (not is_stale or (0.0 <= gest_age_ms <= 250.0)))
                    else None
                )
                if lm:
                    for conn in HAND_CONNECTIONS:
                        p1_idx = getattr(conn, "start", conn[0])
                        p2_idx = getattr(conn, "end", conn[1])
                        if 0 <= p1_idx < len(lm) and 0 <= p2_idx < len(lm):
                            x1_norm, y1_norm = _pt_xy(lm[p1_idx])
                            x2_norm, y2_norm = _pt_xy(lm[p2_idx])
                            x1 = int(x1_norm * FRAME_WIDTH)
                            y1 = int(y1_norm * FRAME_HEIGHT)
                            x2 = int(x2_norm * FRAME_WIDTH)
                            y2 = int(y2_norm * FRAME_HEIGHT)
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
                if face_detector is not None and hailo_embedder is not None and frame_count % FACE_STRIDE == 0:
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
                        if name not in last_greeted_times or (current_time - last_greeted_times[name] > greeting_cooldown_seconds):
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
                    {"matches": face_matches, "timestamp": latest_face_result.get("timestamp", current_time)}
                    if latest_face_result
                    else None
                )
                lock_status = face_locker.update(
                    filtered_face_result,
                    override_name=target_name_override,
                    now=now_mono,
                )

                chosen_det = None
                authorized_face_detected = False

                if lock_status["locked"] and lock_status["bbox"] is not None:
                    authorized_face_detected = True
                    current_tracked_name = lock_status["name"] or current_tracked_name
                    fx1, fy1, fx2, fy2 = lock_status["bbox"]
                    face_conf = next(
                        (m.get("confidence", 0.9) for m in face_matches if m.get("name") == lock_status["name"]),
                        0.9,
                    )

                    matched_body_det = None
                    for det in detections:
                        bx1, by1, bx2, by2 = det["bbox"]
                        if bx1 <= fx1 <= bx2 and by1 <= fy1 <= by2:
                            matched_body_det = det
                            break

                    chosen_det = matched_body_det if matched_body_det else {
                        "bbox": [fx1, fy1, fx2, fy2],
                        "confidence": face_conf,
                    }

                    # only capture/refresh clothing histogram on a fresh LOCKED state
                    if lock_status["state"] == "LOCKED":
                        bx1, by1, bx2, by2 = chosen_det["bbox"]
                        torso_ymin = int(by1 + (by2 - by1) * 0.2)
                        torso_ymax = int(by1 + (by2 - by1) * 0.7)
                        torso_crop = frame[
                            max(0, torso_ymin):min(FRAME_HEIGHT, torso_ymax),
                            max(0, bx1):min(FRAME_WIDTH, bx2),
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

                override_matches_clothing = (
                    target_name_override is None
                    or (
                        clothing_locked_name is not None
                        and clothing_locked_name.strip().lower() == target_name_override.strip().lower()
                    )
                )

                if not chosen_det and is_clothing_locked and detections and override_matches_clothing:
                    if (current_time - clothing_lock_timer) < CLOTHING_LOCK_TIMEOUT and target_clothing_hist is not None:
                        best_match_det = None
                        highest_score = -1.0

                        for det in detections:
                            bx1, by1, bx2, by2 = det["bbox"]
                            torso_ymin = int(by1 + (by2 - by1) * 0.2)
                            torso_ymax = int(by1 + (by2 - by1) * 0.7)
                            candidate_torso = frame[
                                max(0, torso_ymin):min(FRAME_HEIGHT, torso_ymax),
                                max(0, bx1):min(FRAME_WIDTH, bx2),
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
                            cv2.putText(
                                display_frame,
                                f"LOCKED TO CLOTHING (Score: {highest_score:.2f})",
                                (20, 120),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 165, 255),
                                2,
                            )
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
                        target_name_override = None
                        _clear_follow_locks()
                        dist_ctrl.reset()

                    elif v_cmd == "AUTO":
                        print("[MODE] Clearing voice override. Returning to Auto Tracking.")
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        target_name_override = None
                        _clear_follow_locks()
                        dist_ctrl.reset()

                    elif "ELOY" in v_cmd_str:
                        if target_name_override != "eloy":
                            target_name_override = "eloy"
                            _clear_follow_locks()
                            dist_ctrl.reset()
                        manual_voice_override = None
                        voice_override_expiry_time = 0.0
                        print("[VOICE OVERRIDE]: Switching target directly to Eloy")

                    elif "CHIA" in v_cmd_str:
                        if target_name_override != "chia":
                            target_name_override = "chia"
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
                latest_dist_cmd = "STOP"

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
                    cv2.putText(
                        display_frame,
                        f"TRACKING: {current_tracked_name.upper()} [{target_role_to_follow}]",
                        (xmin, max(40, ymin - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    latest_dist_cmd, _, _ = dist_ctrl.update(person_box_height)
                    if manual_voice_override is None:
                        current_frame_command = latest_dist_cmd
                        current_source = "TRACKING_DISTANCE"
                else:
                    servo_tracker.smoothed_bbox = None
                    latest_dist_cmd, _, _ = dist_ctrl.update(None)

                cv2.line(display_frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (100, 100, 100), 1)

                is_fall = False
                fall_status = "OK"
                fresh_bbox = chosen_det["bbox"] if chosen_det is not None else None

                if frame_count > 0 and frame_count % 15 == 0:
                    is_fall, fall_status = pose_fall_detector.process_crop_or_frame(
                        frame, display_frame, bbox=fresh_bbox
                    )
                    persisted_is_fall = is_fall
                    persisted_fall_status = fall_status
                elif frame_count > 0:
                    is_fall = persisted_is_fall if "persisted_is_fall" in locals() else False
                    fall_status = persisted_fall_status if "persisted_fall_status" in locals() else "OK"

                now = time.time()
                if is_fall:
                    if not buzzer_triggered:
                        try:
                            motor.send_buzzer_command(state=True)
                            buzzer_triggered = True
                            print("[ALARM] Fall detected! Buzzer latched ON.")
                        except Exception as e:
                            print(f"[ERROR] Failed to turn buzzer ON: {e}")

                        if speaker:
                            speaker.say("Warning! Fall detected! Requesting assistance.")
                            last_fall_speech_time = now
                    elif (
                        speaker
                        and FALL_SPEECH_REPEAT_INTERVAL is not None
                        and (now - last_fall_speech_time) >= FALL_SPEECH_REPEAT_INTERVAL
                    ):
                        speaker.say("Warning! Fall detected! Requesting assistance.")
                        last_fall_speech_time = now
                else:
                    if buzzer_triggered:
                        try:
                            motor.send_buzzer_command(state=False)
                            buzzer_triggered = False
                            last_fall_speech_time = 0.0
                            print("[INFO] Fall cleared. Buzzer turned OFF and latch reset.")
                        except Exception as e:
                            print(f"[ERROR] Failed to turn buzzer OFF: {e}")

                if gesture_cmd:
                    cv2.putText(display_frame, f"GESTURE: {gesture_cmd} ({hand_side})", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "GESTURE: NONE", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                is_tracking_active = False

                # --- COMMAND DECISION HIERARCHY ---
                if is_fall:
                    cv2.putText(display_frame, f"ALARM: {fall_status}", (20, 140),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
                    current_frame_command = "STOP"
                    current_source = "HOLD"

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

                    if authorized_face_detected and gesture_cmd and gesture_cmd in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]:
                        current_frame_command = gesture_cmd
                        current_source = "GESTURE"

                        if speaker and current_frame_command != last_sent_command:
                            if gesture_cmd == "FORWARD":
                                speaker.say("Moving forward.")
                            elif gesture_cmd == "BACKWARD":
                                speaker.say("Moving backward.")
                            elif gesture_cmd == "LEFT":
                                speaker.say("Turning left.")
                            elif gesture_cmd == "RIGHT":
                                speaker.say("Turning right.")
                            elif gesture_cmd == "STOP":
                                speaker.say("Stopping robot.")

                        cv2.putText(display_frame, f"GESTURE CMD: {gesture_cmd}", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    elif take_photo_triggered:
                        display_timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        file_timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        snapshot_img = frame.copy()

                        cv2.putText(
                            snapshot_img,
                            display_timestamp_str,
                            (20, FRAME_HEIGHT - 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2,
                            cv2.LINE_AA,
                        )

                        filename = f"robot_snapshot_{file_timestamp_str}.jpg"
                        cv2.imwrite(filename, cv2.cvtColor(snapshot_img, cv2.COLOR_RGB2BGR))
                        print(f"[SNAPSHOT] Photo successfully saved: {filename}")

                        if speaker:
                            speaker.say("Photo taken.")

                        cv2.putText(display_frame, "GESTURE CMD: TAKE_PHOTO", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                    elif target_x_center is not None:
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
                            servo_tracker.smoothed_x = (ALPHA * target_x_center) + ((1.0 - ALPHA) * servo_tracker.smoothed_x)
                            servo_tracker.smoothed_y = (ALPHA * raw_y_center) + ((1.0 - ALPHA) * servo_tracker.smoothed_y)

                        steering_error = int(round(servo_tracker.smoothed_x - (FRAME_WIDTH // 2)))
                        vertical_error = int(round(servo_tracker.smoothed_y - (FRAME_HEIGHT // 2)))

                        DEADZONE_X = 24
                        DEADZONE_Y = 24

                        if abs(steering_error) < DEADZONE_X:
                            steering_error = 0
                        if abs(vertical_error) < DEADZONE_Y:
                            vertical_error = 0

                        MAX_STEP = 1.2
                        pan_delta = max(-MAX_STEP, min(MAX_STEP, steering_error * 0.012))
                        tilt_delta = max(-MAX_STEP, min(MAX_STEP, vertical_error * 0.012))
                        servo_tracker.pan_angle = max(PAN_MIN, min(PAN_MAX, servo_tracker.pan_angle - pan_delta))
                        servo_tracker.tilt_angle = max(TILT_MIN, min(TILT_MAX, servo_tracker.tilt_angle - tilt_delta))

                        if servo_tracker.pan_angle <= PAN_MIN or servo_tracker.pan_angle >= PAN_MAX:
                            steering_error = 0
                        if servo_tracker.tilt_angle <= TILT_MIN or servo_tracker.tilt_angle >= TILT_MAX:
                            vertical_error = 0

                        raw_speed = int(abs(steering_error) * 0.35 + abs(vertical_error) * 0.35)
                        dynamic_speed = max(10, min(90, raw_speed)) if abs(steering_error) > 10 else 0

                        current_frame_command = latest_dist_cmd
                        current_source = "TRACKING_DISTANCE"
                        is_tracking_active = True

                        drive_speed = dynamic_speed if current_frame_command != "STOP" else 0
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

                        cv2.putText(
                            display_frame,
                            "TRACKING ACTIVE",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 165, 0),
                            2,
                        )

                    else:
                        servo_tracker.smoothed_x = None
                        servo_tracker.smoothed_y = None
                        is_tracking_active = False

                        current_frame_command = latest_dist_cmd
                        current_source = "TRACKING_DISTANCE" if latest_dist_cmd != "STOP" else "HOLD"

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
                cv2.putText(cv2_display_frame, f"Cmd: {current_frame_command} ({current_source})",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("Robot Vision", cv2_display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

                profiler.record("decision_and_io", time.perf_counter() - t0)
                profiler.maybe_report()
                time.sleep(0.01)

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