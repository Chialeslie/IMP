#!/usr/bin/env python3
import os
import sqlite3
import time
import cv2
import numpy as np
from picamera2 import Picamera2

from facial_enrollment import (
    BASE_DIR,
    DB_PATH,
    HAILO10_FACE_HEF,
    YUNET_MODEL_PATH,
    Hailo10FaceEmbedder,
    extract_face_crop,
    l2_normalize,
)


def load_target_profile(db_path: str, target_name: str) -> tuple[str, str, np.ndarray] | None:
    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found at {db_path}")
        return None

    conn = sqlite3.connect(db_path)
    row = conn.execute(
        "SELECT name, role, embedding FROM users WHERE LOWER(name) = LOWER(?) ORDER BY id DESC LIMIT 1",
        (target_name.strip(),),
    ).fetchone()
    conn.close()

    if row is None:
        print(f"[ERROR] No profile found for '{target_name}' in {db_path}")
        return None

    name, role, blob = row
    emb = l2_normalize(np.frombuffer(blob, dtype=np.float32))
    return str(name), str(role), emb


def main() -> None:
    target_name = input("Enter target name to test (e.g., chia): ").strip() or "chia"
    max_distance = 0.65

    profile = load_target_profile(DB_PATH, target_name)
    if profile is None:
        return

    enrolled_name, enrolled_role, enrolled_emb = profile
    print(f"[INFO] Loaded target '{enrolled_name}' ({enrolled_role}), dim={enrolled_emb.shape[0]}")

    embedder = Hailo10FaceEmbedder(HAILO10_FACE_HEF)
    detector = cv2.FaceDetectorYN.create(
        model=YUNET_MODEL_PATH,
        config="",
        input_size=(640, 480),
        score_threshold=0.85,
        nms_threshold=0.3,
        top_k=5000,
    )

    picam2 = Picamera2()
    picam2.configure(
        picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
    )
    picam2.start()

    print("[INFO] Press 'q' in preview window (or Ctrl+C in terminal) to quit.")
    last_log = 0.0

    try:
        while True:
            frame_raw = picam2.capture_array()
            if frame_raw is None:
                continue

            # Picamera2 "RGB888" is BGR in memory for OpenCV
            bgr_frame = frame_raw
            h_frame, w_frame = bgr_frame.shape[:2]
            detector.setInputSize((w_frame, h_frame))
            _, faces = detector.detect(bgr_frame)

            vis = bgr_frame.copy()
            if faces is not None and len(faces) > 0:
                best_face = max(faces, key=lambda f: f[-1])
                fx, fy, fw, fh = float(best_face[0]), float(best_face[1]), float(best_face[2]), float(best_face[3])

                crop = extract_face_crop(bgr_frame, (fx, fy, fw, fh), pad_ratio=0.0)
                if crop.size > 0:
                    vec = embedder.get_embedding(crop, input_is_rgb=False)
                    if vec is not None:
                        # Both vectors are L2-normalized, so dot product == cosine similarity
                        dist = float(1.0 - np.dot(vec, enrolled_emb))
                        is_match = dist <= max_distance
                        label = f"{enrolled_name} ({dist:.3f})" if is_match else f"NOT {enrolled_name} ({dist:.3f})"
                        color = (0, 255, 0) if is_match else (0, 0, 255)

                        x1, y1 = max(0, int(round(fx))), max(0, int(round(fy)))
                        x2, y2 = min(w_frame, int(round(fx + fw))), min(h_frame, int(round(fy + fh)))
                        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(vis, label, (x1, max(20, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

                        now = time.monotonic()
                        if (now - last_log) >= 0.5:
                            status = "MATCH" if is_match else "REJECT"
                            print(f"[ONE-FACE] {status} | target={enrolled_name} | dist={dist:.4f}", flush=True)
                            last_log = now

            cv2.imshow("Single-Person Face Test", vis)
            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break
    finally:
        picam2.stop()
        cv2.destroyAllWindows()
        embedder.close()


if __name__ == "__main__":
    main()