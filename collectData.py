# collectData_face_only.py
import cv2
import numpy as np
import mediapipe as mp
import os
import time

mp_face = mp.solutions.face_mesh

# --- Configs ---
DATA_DIR = "data"
LABELS = ["sad", "happy", "angry", "laugh"]  # edite conforme necessidade
SAMPLES_PER_LABEL = 30
DELAY_BETWEEN = 0.5  # segundos entre capturas

def ensure_dirs():
    for lab in LABELS:
        os.makedirs(os.path.join(DATA_DIR, lab), exist_ok=True)

def extract_face_landmarks_landmark_vector(landmarks, image_w, image_h):
    pts = np.array([[lm.x * image_w, lm.y * image_h] for lm in landmarks], dtype=np.float32)
    center = pts.mean(axis=0)
    pts_centered = pts - center
    max_dist = np.linalg.norm(pts_centered, axis=1).max() + 1e-6
    pts_norm = pts_centered / max_dist
    return pts_norm.flatten()

def main():
    ensure_dirs()
    cap = cv2.VideoCapture(0)
    
    with mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as face_mesh:

        for label in LABELS:
            print(f"\nPressione ENTER para começar a coletar label: {label}")
            input()
            saved = 0
            last_time = 0

            while saved < SAMPLES_PER_LABEL:
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                display = frame.copy()

                # --- Face ---
                face_results = face_mesh.process(rgb)
                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark
                    face_vec = extract_face_landmarks_landmark_vector(landmarks, w, h)

                    # desenhar alguns pontos para visualização
                    num_points_to_show = 100
                    step = max(1, len(landmarks) // num_points_to_show)
                    for lm in landmarks[::step]:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (cx, cy), 2, (0,255,0), -1)

                    # --- Salvar vetor ---
                    now = time.time()
                    if now - last_time > DELAY_BETWEEN:
                        fname = os.path.join(DATA_DIR, label, f"{label}_{saved:04d}.npy")
                        np.save(fname, face_vec)
                        saved += 1
                        last_time = now
                        print(f"Saved {saved}/{SAMPLES_PER_LABEL} for {label}")

                # --- Mostrar na tela ---
                cv2.putText(display, f"Label: {label} ({saved}/{SAMPLES_PER_LABEL})", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                cv2.imshow("Collect", display)

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC para sair
                    cap.release()
                    cv2.destroyAllWindows()
                    return

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
