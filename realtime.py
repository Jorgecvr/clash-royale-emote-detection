import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import threading
from PIL import Image
from playsound import playsound

MODEL_PATH = "models/face_expr_model.joblib"
EMOTE_DIR = "emotes"
SOUND_DIR = "sounds"
THRESHOLD = 0.10  
SOUND_DELAY = 2.0  

def extract_face_landmarks_landmark_vector(landmarks, image_w, image_h):
    pts = np.array([[lm.x * image_w, lm.y * image_h] for lm in landmarks], dtype=np.float32)
    center = pts.mean(axis=0)
    pts_centered = pts - center
    max_dist = np.linalg.norm(pts_centered, axis=1).max() + 1e-6
    pts_norm = pts_centered / max_dist
    return pts_norm.flatten()

def overlay_png_alpha(bg_bgr, png_img, x, y, scale=1.0):
    bg_h, bg_w = bg_bgr.shape[:2]
    w, h = png_img.size
    w = int(w * scale)
    h = int(h * scale)
    if x >= bg_w or y >= bg_h:
        return bg_bgr
    png_resized = png_img.resize((w, h), Image.Resampling.LANCZOS)
    rgba = np.array(png_resized)
    bgr = rgba[..., :3][:, :, ::-1].copy()
    alpha = rgba[..., 3] / 255.0

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    roi = bg_bgr[y1:y2, x1:x2].astype(np.float32)
    fg = bgr[:y2 - y1, :x2 - x1].astype(np.float32)
    a = alpha[:y2 - y1, :x2 - x1][..., None]
    out = (1.0 - a) * roi + a * fg
    bg_bgr[y1:y2, x1:x2] = out.astype(np.uint8)
    return bg_bgr

def load_emotes(labels):
    mapping = {}
    for lab in labels:
        path = os.path.join(EMOTE_DIR, f"{lab}.png")
        mapping[lab] = Image.open(path).convert("RGBA") if os.path.exists(path) else None
    return mapping

def play_sound_async(path):
    threading.Thread(target=playsound, args=(path,), daemon=True).start()

def main():
    data = joblib.load(MODEL_PATH)
    clf = data["model"]
    labels = list(data["labels"])
    emote_map = load_emotes(labels)

    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)

    current_label = None
    current_prob = 0.0
    last_sound_time = 0

    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=1,
                          refine_landmarks=False,
                          min_detection_confidence=0.5,
                          min_tracking_confidence=0.5) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            display = frame.copy()

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    vec = extract_face_landmarks_landmark_vector(face_landmarks.landmark, w, h).reshape(1, -1)
                    probs = clf.predict_proba(vec)[0]
                    idx = int(np.argmax(probs))
                    label = clf.classes_[idx]
                    prob = probs[idx]

                    cv2.putText(display, f"{label} {prob:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    now = time.time()
                    if (label != current_label or (now - last_sound_time) > SOUND_DELAY) and prob >= THRESHOLD:
                        sound_path = os.path.join(SOUND_DIR, f"{label}.mp3")
                        if os.path.exists(sound_path):
                            play_sound_async(sound_path)
                            last_sound_time = now
                        current_label = label
                        current_prob = prob

            if current_label and current_prob >= THRESHOLD and emote_map.get(current_label) is not None:
                emote_img = emote_map[current_label]
                scale = 0.5  
                x = w - int(emote_img.width * scale) - 20
                y = h - int(emote_img.height * scale) - 20
                display = overlay_png_alpha(display, emote_img, x, y, scale=scale)

            cv2.imshow("Real-time Emotes", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
