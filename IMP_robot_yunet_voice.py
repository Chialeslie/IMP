#!/usr/bin/env python3
import os
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
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from vosk import Model, KaldiRecognizer

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
    def __init__(self, model_path):
        print("[TRACE] Initializing native Picamera2 + IMX500 context...")
        self.imx500 = IMX500(model_path)
        # Pass the imx500 camera identifier to Picamera2
        self.picam2 = Picamera2(self.imx500.camera_num)
        
        config = self.picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        print("[TRACE] IMX500 camera started successfully.")

    def read(self):
        try:
            frame = self.picam2.capture_array("main")
            metadata = self.picam2.capture_metadata()
            return frame, metadata if metadata else {}
        except Exception:
            return None, {}

    def parse_detections(self, metadata, confidence_threshold=0.5, target_class_id=0):
        detections = []
        try:
            outputs = self.imx500.get_outputs(metadata)
            if outputs is not None:
                for detection in outputs:
                    flat_det = detection.flatten() if hasattr(detection, 'flatten') else detection
                    
                    if len(flat_det) >= 6:
                        ymin, xmin, ymax, xmax = float(flat_det[0]), float(flat_det[1]), float(flat_det[2]), float(flat_det[3])
                        conf = float(flat_det[4])
                        class_id = int(flat_det[5])
                        
                        if not (0.0 <= conf <= 1.0):
                            continue
                            
                        if conf >= confidence_threshold and (target_class_id is None or class_id == target_class_id):
                            x1 = int(xmin * FRAME_WIDTH)
                            y1 = int(ymin * FRAME_HEIGHT)
                            x2 = int(xmax * FRAME_WIDTH)
                            y2 = int(ymax * FRAME_HEIGHT)
                            
                            if 0 <= x1 < x2 <= FRAME_WIDTH and 0 <= y1 < y2 <= FRAME_HEIGHT:
                                detections.append({
                                    "bbox": [x1, y1, x2, y2], 
                                    "confidence": conf
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
        self.thread = threading.Thread(target=self._speak_loop, daemon=True)
        self.thread.start()

    def _speak_loop(self):
        while self.running:
            try:
                text = self.speech_queue.get(timeout=0.1)
                print(f"[ROBOT SPEAKS]: {text}")
                try:
                    subprocess.run(["espeak-ng", text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception as e:
                    print(f"[SPEAKER ERROR] Could not play audio: {e}")
                self.speech_queue.task_done()
            except queue.Empty:
                continue

    def say(self, text):
        self.speech_queue.put(text)

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
class Hailo10FaceEmbedder:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"[ERROR] HEF model missing at {hef_path}")

        params = VDevice.create_params()
        self.target = VDevice(params)
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        self.infer_model.input().set_format_type(FormatType.FLOAT32)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)
        self.configured_infer_model = self.infer_model.configure()

    def get_embedding(self, face_crop):
        try:
            if face_crop is None or face_crop.size == 0:
                return None

            input_height, input_width = 112, 112
            
            # Note: Since the main loop reads and converts frames to RGB, 
            # face_crop is already RGB. We skip cv2.cvtColor(..., cv2.COLOR_BGR2RGB) 
            # to prevent channel inversion.
            resized = cv2.resize(face_crop, (input_width, input_height))
            normalized = resized.astype(np.float32) / 255.0
            input_data = np.expand_dims(normalized, axis=0)

            bindings = self.configured_infer_model.create_bindings()
            output_tensor = self.infer_model.output()

            bindings.input().set_buffer(input_data)
            output_buffer = np.empty(output_tensor.shape, dtype=np.float32)
            bindings.output().set_buffer(output_buffer)

            self.configured_infer_model.run([bindings], timeout=5000)
            embedding = output_buffer.flatten()
            norm = np.linalg.norm(embedding)
            return embedding / norm if norm > 0 else embedding
        except Exception as e:
            print(f"[LIVE EMBED ERROR]: {e}")
            return None

# -----------------------------------------------------------------------------
# SQLite Face Database Manager
# -----------------------------------------------------------------------------
class FaceDatabase:
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.known_embeddings = []
        self.known_names = []
        self.known_roles = []
        self.reload_database()

    def reload_database(self):
        self.known_embeddings.clear()
        self.known_names.clear()
        self.known_roles.clear()

        if not os.path.exists(self.db_path):
            print(f"[WARNING] Face database missing at {self.db_path}. Operating in generic person mode.")
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT name, role, embedding FROM users")
            rows = cursor.fetchall()
            for name, role, emb_bytes in rows:
                vec = np.frombuffer(emb_bytes, dtype=np.float32)
                self.known_embeddings.append(vec)
                self.known_names.append(name)
                self.known_roles.append(role)
            print(f"[INFO] Loaded {len(self.known_names)} profiles from face database.")
        except Exception as e:
            print(f"[ERROR] Database error: {e}")
        finally:
            conn.close()

    def match_face(self, query_vector, max_distance=0.65):
        if not self.known_embeddings or query_vector is None:
            return "Unknown", "none", 1.0

        distances = [1.0 - np.dot(query_vector, known_vec) for known_vec in self.known_embeddings]
        best_match_idx = int(np.argmin(distances))
        min_dist = distances[best_match_idx]

        if min_dist <= max_distance:
            return self.known_names[best_match_idx], self.known_roles[best_match_idx], min_dist

        return "Unknown", "none", min_dist
    
# -----------------------------------------------------------------------------
# Skeleton-Based Fall Detector (Using Body Keypoints)
# -----------------------------------------------------------------------------
class PoseFallDetector:
    def __init__(self, frame_height=FRAME_HEIGHT):
        self.frame_height = frame_height
        self.fall_detected = False
        self.fall_timestamp = 0
        self.cooldown_sec = 3.0
        
        # Initialize MediaPipe Pose Landmarker
        pose_task_path = os.path.join(BASE_DIR, "pose_landmarker.task")
        if os.path.exists(pose_task_path):
            base_options = python.BaseOptions(model_asset_path=pose_task_path)
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=1,
                min_pose_detection_confidence=0.2,
                min_tracking_confidence=0.2
            )
            self.pose_detector = vision.PoseLandmarker.create_from_options(options)
        else:
            self.pose_detector = None
            print(f"[WARNING] Pose landmark task missing at {pose_task_path}. Fall detection running in fallback mode.")

    def process_crop_or_frame(self, frame, display_frame, bbox=None):
        current_time = time.time()
        torso_angle = 90.0
        is_fallen = False
        status_text = "NORMAL"

        if self.pose_detector:
            # Run pose detection on the full frame or cropped region safely
            if bbox is not None:
                xmin, ymin, xmax, ymax = bbox
                # Ensure valid crop boundaries
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(FRAME_WIDTH, xmax), min(FRAME_HEIGHT, ymax)
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

                        # Draw skeleton line on display frame
                        cv2.circle(display_frame, (shoulder_x, shoulder_y), 4, (0, 0, 255), -1)
                        cv2.circle(display_frame, (hip_x, hip_y), 4, (255, 0, 0), -1)
                        cv2.line(display_frame, (shoulder_x, shoulder_y), (hip_x, hip_y), (0, 255, 255), 2)

                        # Compute torso angle relative to horizontal ground
                        dx = hip_x - shoulder_x
                        dy = hip_y - shoulder_y
                        torso_angle = np.degrees(np.arctan2(abs(dy), abs(dx + 1e-6)))

                        # Fall condition: torso is nearly parallel to the floor (< 35 degrees)
                        if torso_angle < 35.0:
                            self.fall_detected = True
                            self.fall_timestamp = current_time
                except Exception as e:
                    # Catch any MediaPipe runtime bounds error gracefully
                    pass

        # Check cooldown state
        if self.fall_detected:
            if time.time() - self.fall_timestamp < self.cooldown_sec:
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
            min_hand_detection_confidence=0.60,  # Increased for strictness
            min_tracking_confidence=0.60         # Increased for strictness
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame, display_frame):
        rgb_frame = frame
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = self.detector.detect(mp_image)
        gesture_cmd = None  # Default to None so it doesn't force ghost values

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                lm = hand_landmarks

                for connection in HAND_CONNECTIONS:
                    p1 = (int(lm[connection[0]].x * FRAME_WIDTH), int(lm[connection[0]].y * FRAME_HEIGHT))
                    p2 = (int(lm[connection[1]].x * FRAME_WIDTH), int(lm[connection[1]].y * FRAME_HEIGHT))
                    cv2.line(display_frame, p1, p2, (0, 255, 255), 2)

                index_open = lm[8].y < lm[6].y
                middle_open = lm[12].y < lm[10].y
                ring_open = lm[16].y < lm[14].y
                pinky_open = lm[20].y < lm[18].y

                thumb_index_dist = ((lm[4].x - lm[8].x)**2 + (lm[4].y - lm[8].y)**2)**0.5
                wrist_y = lm[0].y

                print(f"[GESTURE DEBUG] Wrist Y: {wrist_y:.2f} | Index Up: {lm[8].y < lm[6].y}")
                print(f"[DEBUG GESTURE] index_open: {index_open} | middle_open: {middle_open} | ring_open: {ring_open} | pinky_open: {pinky_open}")

                # ONLY evaluate gestures if the hand is raised above 0.50 height gate
                if wrist_y < 0.50:
                    # 1. FORWARD: ONLY Index finger open (Highest priority for pointing)
                    if index_open and not middle_open and not ring_open and not pinky_open:
                        gesture_cmd = "FORWARD"
                        break

                    # 2. BACKWARD: Index AND Middle open, others closed
                    elif index_open and middle_open and not ring_open and not pinky_open:
                        gesture_cmd = "BACKWARD"
                        break
                        
                    # 3. OK Sign (Take Photo)
                    elif thumb_index_dist < 0.07 and middle_open and ring_open and pinky_open:
                        gesture_cmd = "TAKE_PHOTO"
                        break
                        
                    # 4. Thumb Left / Right
                    elif not middle_open and not ring_open and not pinky_open:
                        thumb_dx = lm[4].x - lm[0].x
                        if abs(thumb_dx) > 0.12:
                            gesture_cmd = "LEFT" if thumb_dx < 0 else "RIGHT"
                            break
                            
                    # 5. STOP: All fingers open or flat hand (only when raised)
                    elif index_open and middle_open and ring_open and pinky_open:
                        gesture_cmd = "STOP"
                        break

        return gesture_cmd

# -----------------------------------------------------------------------------
# Offline Voice Recognizer (Vosk + PyAudio)
# -----------------------------------------------------------------------------
class OfflineVoiceRecognizer:

    def __init__(self, command_queue, model_path, speaker=None, vip_ref_paths=None):
        self.command_queue = command_queue
        self.speaker = speaker
        self.is_running = True
        
        # Audio configuration (Matches ESP32-C3 firmware sample rate of 16kHz)
        self.target_rate = 16000  
        self.audio_buffer = deque(maxlen=150)

        print("[VOICE] Loading offline Vosk model...")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, self.target_rate)

        # --- ESP32-C3 SERIAL STREAM SETUP ---
        self.esp_audio_port = "/dev/serial/by-id/usb-Espressif_USB_JTAG_serial_debug_unit_A4:CB:8F:20:E0:3C-if00"
        self.esp_baud_rate = 921600
        
        try:
            self.esp_audio_serial = serial.Serial(self.esp_audio_port, self.esp_baud_rate, timeout=0.01)
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
            if hasattr(self, 'esp_audio_serial') and self.esp_audio_serial and self.esp_audio_serial.is_open:
                payload = bytes([0x03]) + f"{source}:{command}\n".encode('utf-8')
                self.esp_audio_serial.write(payload)
        except Exception as e:
            print(f"[SERIAL TEXT ERROR]: {e}")

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

    def _verify_speaker(self, raw_audio):
        """Verifies incoming raw audio against stored multi-sample speaker embeddings using cosine similarity."""
        if not self.speaker_embeddings or not self.voice_encoder:
            print("[VOICE WARNING] Speaker verification bypassed (no models or encoder loaded).")
            return True, "unknown"

        try:
            # Convert raw 16-bit PCM bytes into float32 array normalized between -1.0 and 1.0
            audio_np = np.frombuffer(raw_audio, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Trim silence using librosa
            audio_trimmed, _ = librosa.effects.trim(audio_np, top_db=20)
            
            # Resemblyzer requires a minimum length of audio (1 second at 16kHz)
            if len(audio_trimmed) < 8000:
                print("[VOICE DEBUG] Audio too short for reliable embedding extraction.")
                return False, "UNKNOWN"

            # Generate embedding for the incoming utterance
            test_embedding = self.voice_encoder.embed_utterance(audio_trimmed)

            best_score = -1.0
            matched_speaker = "UNKNOWN"
            
            # Compare test embedding against all enrolled speakers
            for name, ref_embedding in self.speaker_embeddings.items():
                similarity = np.dot(test_embedding, ref_embedding) / (
                    np.linalg.norm(test_embedding) * np.linalg.norm(ref_embedding)
                )
                if similarity > best_score:
                    best_score = similarity
                    matched_speaker = name
            
            print(f"[VOICE DEBUG] Best Match: '{matched_speaker}' | Cosine Similarity Score: {best_score:.4f}")
            
            # Threshold for acceptance (adjust between 0.70 to 0.82 as needed during testing)
            THRESHOLD = 0.45
            if best_score >= THRESHOLD:
                return True, matched_speaker
                
            return False, matched_speaker

        except Exception as e:
            print(f"[VOICE ERROR] Error during speaker verification: {e}")
            return False, "UNKNOWN"

    def _listen_loop(self):
        print("[VOICE DEBUG] _listen_loop thread started...")
        try:
            if not self.esp_audio_serial:
                print("[VOICE ERROR] ESP32 serial stream is not active. Exiting loop.")
                return

            print("[VOICE] Streaming raw PCM bytes from ESP32-C3 at 16kHz...")

            while self.is_running:
                if getattr(self, 'is_speaking', False):
                    time.sleep(0.1)
                    continue
                
                # Read chunks of raw PCM bytes from the ESP32-C3 USB buffer
                data = self.esp_audio_serial.read(512)
                
                if not data or len(data) < 2:
                    continue

                if len(data) % 2 != 0:
                    data = data[:-1]
                if len(data) == 0:
                    continue

                # Single unified RMS calculation and logging
                audio_array = np.frombuffer(data, dtype=np.int16)
                if audio_array.size > 0:
                    rms = np.sqrt(np.mean(audio_array.astype(np.float32) ** 2))
                else:
                    rms = 0
                    #print(f"[AUDIO BYTES] Read {len(data)} bytes | RMS: {int(rms)}", end="\r")
                    #if rms > 25:  # Sound threshold
                        #print(f"\n[AUDIO DEBUG] Sound detected! RMS level: {int(rms)}")

                # Keep audio buffer populated for speaker verification
                self.audio_buffer.append(data)
                if len(self.audio_buffer) > 150:
                    self.audio_buffer.popleft()

                # Pass raw bytes directly to Vosk recognizer
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip().lower()
                    print(f"\n[VOSK RAW RESULT JSON]: {result}")
                    
                    if text:
                        print(f"\n[VOICE RECOGNIZED RAW TEXT]: '{text}'")
                       
                        # Speaker Verification Check
                        combined_audio = b"".join(self.audio_buffer)
                        is_verified, speaker_name = self._verify_speaker(combined_audio)
                        print(f"[VOICE DEBUG] Speaker Verification Result: {is_verified} ({speaker_name}) for '{text}'")

                        if not is_verified:
                            print(f"[VOICE REJECTED] Command '{text}' ignored by speaker verification.")
                            if self.speaker:
                                self.speaker.say("Command ignored. Unrecognized voice.")
                            self.audio_buffer.clear()
                            continue 
                        
                        print(f"[VOICE PASSED VERIFICATION]: '{text}' (User: {speaker_name})")

                        def register_cmd(cmd_string, speech_reply):
                            print(f"[VOICE BROADCAST]: {speech_reply}")
                            self.command_queue.put(cmd_string)
                            if self.speaker:
                                self.speaker.say(speech_reply)
    
                        # Optional: Mute flag to prevent robot from hearing itself
                            self.is_speaking = True
                            self.audio_buffer.clear()
                            self.recognizer = KaldiRecognizer(self.model, self.target_rate)
                            time.sleep(0.5)
                            self.is_speaking = False

                        # Match Commands
                        if "forward" in text or "go" in text or " come here" in text:
                            register_cmd("FORWARD", "Moving forward.")
                        elif "stop" in text or "halt" in text:
                            register_cmd("STOP", "Stopping robot.")
                        elif "back" in text or "backward" in text or "reverse" in text:
                            register_cmd("BACKWARD", "Reversing.")
                        elif "left" in text:
                            register_cmd("LEFT", "Turning left.")
                        elif "right" in text:
                            register_cmd("RIGHT", "Turning right.")
                        elif "auto" in text or "track" in text:
                            register_cmd("AUTO", "Resuming auto mode.")
                        elif "eloy" in text:
                            register_cmd("ELOY", "Switching target to Eloy.")
                        elif "chia" in text:
                            register_cmd("CHIA", "Switching target to Chia.")
                        else:
                            register_cmd(text.upper(), f"Executing {text}.")

                    self.audio_buffer.clear()
                    self.recognizer = KaldiRecognizer(self.model, self.target_rate)
                else:
                    partial = json.loads(self.recognizer.PartialResult()).get("partial", "").strip()
                    if partial:
                        print(f"[VOICE PARTIAL]: '{partial}'          ", end="\r", flush=True)

        except Exception as e:
            print(f"[VOICE FATAL ERROR CRASH] {e}")
            import traceback
            traceback.print_exc()

# -----------------------------------------------------------------------------
# Motor Controller (with Auto-Reconnect)
# -----------------------------------------------------------------------------
class MotorController:
    def __init__(self, port, baud):
        self.port = port
        self.baud = baud
        self.ser = None
        self.connect()

    def connect(self):
        try:
            # ADD write_timeout=0.05 HERE:
            self.ser = serial.Serial(self.port, self.baud, timeout=1, write_timeout=0.05)
            time.sleep(1)
            print(f"[INFO] Serial connected to {self.port}")
        except Exception as e:
            self.ser = None
            print(f"[WARNING] Serial connection failed: {e}")

    def send_command(self, source, cmd):
        if self.ser is None or not self.ser.is_open:
            return  # Skip instead of blocking the main loop trying to reconnect synchronously

        try:
            payload = f"{source}:{cmd}\n"
            packet = bytes([0x03]) + payload.encode('utf-8')
            print(f"[SERIAL OUT] Source: {source} | Cmd: '{cmd}' | Hex Bytes: {packet.hex()}")
            self.ser.write(packet)
        except serial.SerialTimeoutException:
            # Just drop this packet if the buffer is busy, keeping the camera fluid!
            pass
        except Exception as e:
            print(f"[WARNING] Serial error: {e}")
            self.ser = None

    def close(self):
        if self.ser and self.ser.is_open:
            try:
                self.send_command("HOLD", "STOP")
                self.ser.close()
            except Exception:
                pass

    def send_hybrid_tracking_command(self, pan_angle, tilt_angle, steering_error, base_speed):
        if self.ser is None or not self.ser.is_open:
            return

        try:
            p_angle = pan_angle if pan_angle is not None else 135.0
            t_angle = tilt_angle if tilt_angle is not None else 90.0
            
            pan_scaled = max(30, min(240, int(float(p_angle))))
            tilt_byte = max(0, min(180, int(t_angle)))
            
            err = steering_error if steering_error is not None else 0
            encoded_error = int(err) + 100
            encoded_error = max(0, min(255, encoded_error))
            base_speed = max(0, min(200, int(base_speed)))
            
            # Single unified packet: [Header 0x02, Pan, Tilt, Steering Error, Speed]
            packet = bytes([0x02, pan_scaled, tilt_byte, encoded_error, base_speed])
            print(f"DEBUG: Sending packet -> {packet.hex()}")
            self.ser.write(packet)
            
            print(f"[SERIAL OUT] Hybrid Cmd | Pan: {pan_scaled} | Tilt: {tilt_byte} | Err: {encoded_error} | Speed: {base_speed} | Hex: {packet.hex()}")
            
        except serial.SerialTimeoutException:
            pass
        except Exception as e:
            print(f"[WARNING] Failed to send hybrid tracking command over serial: {e}")

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

    # --- Initialize YuNet Face Detector ---
    face_detector = None
    if os.path.exists(YUNET_MODEL_PATH):
        try:
            face_detector = cv2.FaceDetectorYN.create(
                model=YUNET_MODEL_PATH,
                config="",
                input_size=(FRAME_WIDTH, FRAME_HEIGHT),
                score_threshold=0.6,
                nms_threshold=0.3,
                top_k=5000
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
    target_role_to_follow = "VIP_FOLLOW"
    target_name_override = None

    class DefaultServoTracker:
        def __init__(self):
            self.pan_angle = 135
            self.tilt_angle = 90

    servo_tracker = DefaultServoTracker()
    last_snapshot_time = 0
    snapshot_cooldown = 2.0  
    last_greeted_times = {}
    greeting_cooldown_seconds = 3600.0

    last_sent_command = None
    last_send_time = 0
    
    frame_count = 0
    current_frame_command = "STOP"
    current_source = "HOLD"
    last_serial_send_time = 0
    SERIAL_SEND_INTERVAL = 0.25
    missing_target_count = 0
    MAX_MISSING_FRAMES = 5
    last_valid_target = None  # Explicitly initialized to prevent scope bugs
    voice_override_expiry_time = 0
    VOICE_OVERRIDE_DURATION = 2.0
    current_terrain = "PAVEMENT"

    try:
        print("[DEBUG] Entering main loop now...")
        while True:
            frame_count += 1  
            if frame_count % 30 == 0:
                print(f"[STATUS] Running... Frame: {frame_count} | Command: {current_frame_command} | Source: {current_source}")
            try:
                frame, metadata = cam.read()
                if frame is None:
                    time.sleep(0.005)
                    continue
            
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display_frame = frame.copy()

                target_bbox = None
                target_x_center = None
                
                if frame_count % 3 == 0:
                    current_terrain = terrain_classifier.analyze_surface(frame, display_frame)
                    is_fall, fall_status = pose_fall_detector.process_crop_or_frame(frame, display_frame, bbox=target_bbox)
                
                gesture_cmd = None
                if gesture_rec:
                    if frame_count % 5 == 0:
                        gesture_cmd = gesture_rec.process_frame(frame, display_frame)
                        if gesture_cmd:
                            persisted_gesture = gesture_cmd
                    else:
                        if time.time() - last_gesture_time < 1.5:
                            gesture_cmd = persisted_gesture

                detections = cam.parse_detections(metadata, CONFIDENCE_THRESHOLD, TARGET_CLASS_ID)

                # --- Run YuNet Face & Landmark Detection ---
                chosen_det = None
                authorized_face_detected = False
                faces = []
                
                if face_detector is not None:
                    bgr_frame_for_yunet = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    face_detector.setInputSize((FRAME_WIDTH, FRAME_HEIGHT))
                    _, faces = face_detector.detect(bgr_frame_for_yunet)

                if faces is not None and len(faces) > 0:
                    for face in faces:
                        fx, fy, fw, fh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                        fx1, fy1, fx2, fy2 = max(0, fx), max(0, fy), min(FRAME_WIDTH, fx + fw), min(FRAME_HEIGHT, fy + fh)
                        
                        landmarks = face[4:14].reshape((5, 2))
                        landmark_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
                        for idx, pt in enumerate(landmarks):
                            cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, landmark_colors[idx], -1)

                        person_crop = frame[fy1:fy2, fx1:fx2]
                        name = "Unknown"
                        role = "none"

                        if person_crop.size > 0 and hailo_embedder:
                            query_vector = hailo_embedder.get_embedding(person_crop)
                            name, role, dist = face_db.match_face(query_vector, max_distance=0.65)
                            #print(f"[FACE DEBUG] Matched: {name} ({role}) | Distance: {dist:.4f}")
                            if name != "Unknown" and name is not None:
                                if role.upper() == "VIP_FOLLOW":
                                    authorized_face_detected = True

                                current_time = time.time()
                                if name not in last_greeted_times or (current_time - last_greeted_times[name]) > greeting_cooldown_seconds:
                                    last_greeted_times[name] = current_time
                                    speaker.say(f"Hello {name}. Welcome back.")

                        matched_body_det = None
                        for det in detections:
                            bx1, by1, bx2, by2 = det["bbox"]
                            if bx1 <= fx1 <= bx2 and by1 <= fy1 <= by2:
                                matched_body_det = det
                                break

                        target_evaluation_det = matched_body_det if matched_body_det else {"bbox": [fx1, fy1, fx2, fy2], "confidence": face[14]}

                        label = f"{name} ({role})"
                        cv2.putText(display_frame, label, (fx1, max(20, fy1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 0, 255), 2)

                        if target_name_override and name.lower() == target_name_override.lower():
                            chosen_det = target_evaluation_det
                            break
                        elif not target_name_override and role.upper() == target_role_to_follow.upper():
                            chosen_det = target_evaluation_det

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
                try:
                    v_cmd = voice_queue.get_nowait()
                
                    v_cmd_str = str(v_cmd).upper()
                
                    if v_cmd in ["VIP_FOLLOW", "GUEST", "CARE_TAKER", "PATIENT"]:
                        target_role_to_follow = v_cmd
                        print(f"[MODE] Target Follow Role Updated: {target_role_to_follow}")
                        manual_voice_override = None
                        target_name_override = None  
                    
                    elif v_cmd == "AUTO":
                        print("[MODE] Clearing voice override. Returning to Auto Tracking.")
                        manual_voice_override = None
                        target_name_override = None
                    
                    elif "ELOY" in v_cmd_str:
                        target_name_override = "eloy"
                        print(f"[VOICE OVERRIDE]: Switching target directly to Eloy")

                    elif "CHIA" in v_cmd_str:
                        target_name_override = "chia"
                        print(f"[VOICE OVERRIDE]: Switching target directly to Chia")

                    else:
                        print(f"[MODE] Temporary Voice Override Activated: {v_cmd}")
                        manual_voice_override = v_cmd
                        voice_override_expiry_time = time.time() + VOICE_OVERRIDE_DURATION
                    voice_queue.task_done()
                except queue.Empty:
                    pass 

                if active_target is not None:
                    xmin, ymin, xmax, ymax = active_target["bbox"]
                    target_bbox = active_target["bbox"]
                    target_x_center = int((xmin + xmax) / 2)
                    
                    cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), (255, 255, 0), 3)
                    cv2.putText(display_frame, f"TRACKING TARGET [{target_name_override or target_role_to_follow}]", (xmin, max(40, ymin - 30)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.line(display_frame, (FRAME_WIDTH // 2, 0), (FRAME_WIDTH // 2, FRAME_HEIGHT), (100, 100, 100), 1)

                is_fall = False
                fall_status = "OK"

                if frame_count > 0 and frame_count % 15 == 0:
                    is_fall, fall_status = pose_fall_detector.process_crop_or_frame(frame, display_frame, bbox=target_bbox)
                    persisted_is_fall = is_fall
                    persisted_fall_status = fall_status
                elif frame_count > 0:
                    is_fall = locals().get('persisted_is_fall', False)
                    fall_status = locals().get('persisted_fall_status', "OK")

                if gesture_cmd:
                    cv2.putText(display_frame, f"GESTURE: {gesture_cmd}", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    cv2.putText(display_frame, "GESTURE: NONE", (20, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 1)

                is_tracking_active = False
                current_frame_command = None
                current_source = "HOLD"

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
                    
                elif manual_voice_override and time.time() < voice_override_expiry_time:
                    current_frame_command = manual_voice_override
                    current_source = "VOICE"
                    cv2.putText(display_frame, f"VOICE OVERRIDE: {current_frame_command}", (20, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                else:
                    if manual_voice_override:
                        manual_voice_override = None

                    if authorized_face_detected and gesture_cmd and gesture_cmd in ["FORWARD", "BACKWARD", "LEFT", "RIGHT", "STOP"]:
                        persisted_gesture = gesture_cmd
                        last_gesture_time = time.time()
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
                        
                    elif authorized_face_detected and gesture_cmd == "TAKE_PHOTO":
                        current_time = time.time()
                        if current_time - last_snapshot_time > snapshot_cooldown:
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
                                cv2.LINE_AA
                            )
                            
                            filename = f"robot_snapshot_{file_timestamp_str}.jpg"
                            cv2.imwrite(filename, cv2.cvtColor(snapshot_img, cv2.COLOR_RGB2BGR)) 
                            print(f"[SNAPSHOT] Photo successfully saved: {filename}")
                            
                            if speaker:
                                speaker.say("Photo taken.")
                                
                            last_snapshot_time = current_time
                            
                        cv2.putText(display_frame, f"GESTURE CMD: TAKE_PHOTO", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                                                                  
                    elif target_x_center is not None:
                        missing_target_count = 0  
                        
                        steering_error = target_x_center - (FRAME_WIDTH // 2)
                        servo_tracker.pan_angle -= steering_error * 0.05
                        servo_tracker.pan_angle = max(0, min(270, servo_tracker.pan_angle))
                        
                        target_y_center = int((target_bbox[1] + target_bbox[3]) / 2)
                        vertical_error = target_y_center - (FRAME_HEIGHT // 2)
                        servo_tracker.tilt_angle -= vertical_error * 0.05
                        servo_tracker.tilt_angle = max(30, min(120, servo_tracker.tilt_angle))
                        
                        base_speed = 150
                        
                        current_time = time.time()
                        if current_time - last_serial_send_time >= SERIAL_SEND_INTERVAL:
                            try:
                                motor.send_hybrid_tracking_command(servo_tracker.pan_angle, servo_tracker.tilt_angle, steering_error, base_speed)
                                last_serial_send_time = current_time
                            except Exception as e:
                                print(f"[SERIAL WARNING]: Failed to send tracking command: {e}")
                                
                        is_tracking_active = True
                        current_source = "TRACKING"
                        cv2.putText(display_frame, f"TRACKING TARGET (Err X:{steering_error} Y:{vertical_error})", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                    elif missing_target_count < MAX_MISSING_FRAMES and current_source == "TRACKING":
                        missing_target_count += 1
                        is_tracking_active = True
                        current_source = "TRACKING"
                        
                        current_time = time.time()
                        if current_time - last_serial_send_time >= SERIAL_SEND_INTERVAL:
                            try:
                                motor.send_hybrid_tracking_command(servo_tracker.pan_angle, servo_tracker.tilt_angle, 0, 150)
                                last_serial_send_time = current_time
                            except Exception as e:
                                pass

                        cv2.putText(display_frame, "TRACKING (Holding target loss...)", (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        
                    else:
                        missing_target_count += 1
                        if last_sent_command in ["FORWARD", "BACKWARD", "LEFT", "RIGHT"]:
                            current_frame_command = last_sent_command
                            current_source = "HOLD"
                            cv2.putText(display_frame, f"HOLDING CMD: {current_frame_command}", (20, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                        else:
                            current_frame_command = "STOP"
                            current_source = "HOLD"
                            cv2.putText(display_frame, "STATUS: IDLE / STOPPED", (20, 90),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if not is_tracking_active and current_frame_command:
                    if current_frame_command != last_sent_command or (time.time() - last_send_time) > 0.5:
                        motor.send_command(current_source, current_frame_command)
                        last_sent_command = current_frame_command
                        last_send_time = time.time()

                cv2_display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                cv2.putText(cv2_display_frame, f"Cmd: {current_frame_command} ({current_source})", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Robot Vision", cv2_display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                time.sleep(0.01)

            except Exception as loop_err:
                print(f"[ERROR INSIDE LOOP]: {loop_err}")
                import traceback
                traceback.print_exc()
                break
    
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down gracefully...")
        
    finally:
        try:
            if 'cam' in locals() and cam:
                cam.running = False
        except Exception:
            pass
        try:
            if speaker:
                speaker.stop()
        except Exception:
            pass
        try:
            if voice_rec:
                voice_rec.stop()
        except Exception:
            pass
        try:
            if motor:
                motor.close()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
            
        print("[INFO] System shutdown complete.")
        #import os
        os._exit(0)

if __name__ == "__main__":
    main()