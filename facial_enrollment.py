#!/usr/init/env python3
#!/usr/bin/env python3
import os
import cv2
import sqlite3
import numpy as np
from picamera2 import Picamera2
from hailo_platform import VDevice, FormatType

# Configuration & Paths
BASE_DIR = "/home/chiagekliang/chiagekliangIMP"
DB_PATH = os.path.join(BASE_DIR, "robot_faces.db")
HAILO10_FACE_HEF = os.path.join(BASE_DIR, "arcface_mobilefacenet_hailo10.hef")
YUNET_MODEL_PATH = os.path.join(BASE_DIR, "face_detection_yunet_2023mar.onnx")

class Hailo10FaceEmbedder:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"[ERROR] HEF model missing at {hef_path}")

        self.target = VDevice()
        self.infer_model = self.target.create_infer_model(hef_path)
        self.infer_model.set_batch_size(1)
        
        self.infer_model.input().set_format_type(FormatType.FLOAT32)
        self.infer_model.output().set_format_type(FormatType.FLOAT32)

        self.configured_infer_model = self.infer_model.configure()

        self.input_shape = self.infer_model.input().shape
        self.output_shape = self.infer_model.output().shape

    def get_embedding(self, face_image):
        h, w = self.input_shape[0], self.input_shape[1]
        resized = cv2.resize(face_image, (w, h))
        
        # Ensure image is RGB for the embedder model
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # CRITICAL: Normalize pixel values to [-1, 1] (standard for ArcFace/MobileFaceNet)
        normalized_frame = (rgb_frame.astype(np.float32) - 127.5) / 128.0
        
        input_data = np.expand_dims(normalized_frame, axis=0).astype(np.float32)

        bindings = self.configured_infer_model.create_bindings()
        output_buffer = np.empty(self.output_shape, dtype=np.float32)

        bindings.input().set_buffer(input_data)
        bindings.output().set_buffer(output_buffer)

        self.configured_infer_model.run([bindings], timeout=10000)

        vec = np.squeeze(output_buffer).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

def init_database(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            role TEXT NOT NULL,
            embedding BLOB NOT NULL
        )
    """)
    conn.commit()
    conn.close()

def add_user_to_db(db_path, name, role, embedding):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    emb_bytes = embedding.tobytes()
    cursor.execute(
        "INSERT INTO users (name, role, embedding) VALUES (?, ?, ?)",
        (name, role, emb_bytes)
    )
    conn.commit()
    conn.close()
    print(f"\n[SUCCESS] Registered {name} ({role}) with YuNet pipeline!")

def enroll_user(name, role):
    init_database(DB_PATH)
    embedder = Hailo10FaceEmbedder(HAILO10_FACE_HEF)

    if not os.path.exists(YUNET_MODEL_PATH):
        print(f"[ERROR] YuNet model missing at {YUNET_MODEL_PATH}")
        return

    # Initialize Picamera2 for capture preview
    # 1. Configure Picamera2 to output RGB888 explicitly
    picam2 = Picamera2()
    picam2.configure(picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"}))
    picam2.start()

    face_detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(640, 480),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000
    )

    print(f"\n[ENROLLMENT] Look at the camera for {name}. Press [SPACE] to capture or [q] to cancel.")

    face_crop_to_save = None

    try:
        while True:
            frame_raw = picam2.capture_array()
            if frame_raw is None:
                continue
            
            # Picamera2 RGB888 outputs RGB -> Convert to BGR for OpenCV display & YuNet
            frame = frame_raw.copy()
            h_frame, w_frame = frame.shape[:2]
            face_detector.setInputSize((w_frame, h_frame))

            _, faces = face_detector.detect(frame)

            display_frame = frame.copy()
            if faces is not None and len(faces) > 0:
                best_face = max(faces, key=lambda f: f[-1])
                x, y, w, h = map(int, best_face[:4])
                
                ymin, ymax = max(0, y), min(h_frame, y + h)
                xmin, xmax = max(0, x), min(w_frame, x + w)
                
                # Keep original BGR crop for saving/embedding conversion later
                face_crop_to_save = frame[ymin:ymax, xmin:xmax].copy()

                # Draw bounding box and landmarks
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                landmarks = best_face[4:14].reshape((5, 2))
                for pt in landmarks:
                    cv2.circle(display_frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)

                cv2.putText(display_frame, "Face Detected - Press SPACE to Enroll", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display_frame, "No Face Detected", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Enrollment Preview", display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '): # Press SPACE to capture
                if faces is not None and len(faces) > 0:
                    break
                else:
                    print("[WARNING] No face detected to capture. Try again.")
            elif key == ord('q'):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()

    if face_crop_to_save is not None and face_crop_to_save.size > 0:
        print("[INFO] Computing embedding vector on Hailo-10H NPU...")
        embedding = embedder.get_embedding(face_crop_to_save)
        add_user_to_db(DB_PATH, name, role, embedding)
    else:
        print("[ERROR] Enrollment canceled or no face captured.")

if __name__ == "__main__":
    user_name = input("Enter Name: ").strip()
    user_role = input("Enter Role (e.g., VIP_FOLLOW): ").strip()
    enroll_user(user_name, user_role)