# realtime.py
import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
from PIL import Image

MODEL_PATH = "models/face_expr_model.joblib"
EMOJI_DIR = "emojis"
THRESHOLD = 0.60  # probabilidade mínima para exibir emoji

def extract_face_landmarks_landmark_vector(landmarks, image_w, image_h):
    pts = np.array([[lm.x * image_w, lm.y * image_h] for lm in landmarks], dtype=np.float32)
    center = pts.mean(axis=0)
    pts_centered = pts - center
    max_dist = np.linalg.norm(pts_centered, axis=1).max() + 1e-6
    pts_norm = pts_centered / max_dist
    return pts_norm.flatten()

def overlay_png_alpha(bg_bgr, png_img, x, y, scale=1.0):
    # png_img: PIL Image with RGBA
    bg_h, bg_w = bg_bgr.shape[:2]
    w, h = png_img.size
    w = int(w * scale); h = int(h * scale)
    if x >= bg_w or y >= bg_h:
        return bg_bgr
    png_resized = png_img.resize((w, h), Image.ANTIALIAS)
    rgba = np.array(png_resized)
    bgr = rgba[..., :3][:, :, ::-1].copy()
    alpha = rgba[..., 3] / 255.0

    # compute overlay region
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(bg_w, x + w), min(bg_h, y + h)
    bx1 = x1; by1 = y1
    px1 = x1 - x; py1 = y1 - y
    px2 = px1 + (x2 - x1); py2 = py1 + (y2 - y1)

    if x1 >= x2 or y1 >= y2:
        return bg_bgr

    roi = bg_bgr[by1:y2, bx1:x2].astype(np.float32)
    fg = bgr[py1:py2, px1:px2].astype(np.float32)
    a = alpha[py1:py2, px1:px2][..., None]
    out = (1.0 - a) * roi + a * fg
    bg_bgr[by1:y2, bx1:x2] = out.astype(np.uint8)
    return bg_bgr

def load_emojis(labels):
    mapping = {}
    for lab in labels:
        path = os.path.join(EMOJI_DIR, f"{lab}.png")
        if os.path.exists(path):
            mapping[lab] = Image.open(path).convert("RGBA")
        else:
            mapping[lab] = None
    return mapping

def main():
    data = joblib.load(MODEL_PATH)
    clf = data["model"]
    labels = list(data["labels"])
    emoji_map = load_emojis(labels)

    mp_face = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    with mp_face.FaceMesh(static_image_mode=False,
                          max_num_faces=2,
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
                    # draw label and probability
                    cv2.putText(display, f"{label} {prob:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255),2)

                    if prob >= THRESHOLD and emoji_map.get(label) is not None:
                        # find approximate face location to place emoji (use nose tip index 1 or landmark 1)
                        # We'll compute bounding box of landmarks
                        pts = np.array([[lm.x*w, lm.y*h] for lm in face_landmarks.landmark])
                        x_min, y_min = int(pts[:,0].min()), int(pts[:,1].min())
                        x_max, y_max = int(pts[:,0].max()), int(pts[:,1].max())
                        face_w = x_max - x_min
                        face_h = y_max - y_min
                        # position emoji above face
                        emoji_img = emoji_map[label]
                        scale = face_w / 150.0  # ajuste para adequar tamanho
                        ex = x_min + face_w//2 - int((emoji_img.width*scale)//2)
                        ey = y_min - int(emoji_img.height*scale) - 10
                        ey = max(0, ey)
                        display = overlay_png_alpha(display, emoji_img, ex, ey, scale=scale)

            cv2.imshow("Real-time Emojis", display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
