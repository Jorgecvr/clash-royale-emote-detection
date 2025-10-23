# collectData.py
import cv2
import numpy as np
import mediapipe as mp
import os
import time

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# --- Configs ---
DATA_DIR = "data"
LABELS = ["sad", "happy", "angry", "laugh"]  # edite conforme necessidade
SAMPLES_PER_LABEL = 30
DELAY_BETWEEN = 0.5  # segundos entre capturas (para não salvar frames duplicados)

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

def extract_hand_landmarks_landmark_vector(landmarks, image_w, image_h):
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
            min_tracking_confidence=0.5) as face_mesh, \
         mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as hands:

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
                face_vec = np.zeros(468*2)  # vetor vazio caso não detecte
                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark
                    face_vec = extract_face_landmarks_landmark_vector(landmarks, w, h)

                    # desenhar mais pontos
                    num_points_to_show = 100
                    step = max(1, len(landmarks) // num_points_to_show)
                    for lm in landmarks[::step]:
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(display, (cx, cy), 2, (0,255,0), -1)

                # --- Hands ---
                hand_results = hands.process(rgb)
                hand_vec = np.zeros(2*21*2)  # vetor para 1 mão (2 mãos podem ser concatenadas)
                if hand_results.multi_hand_landmarks:
                    all_hands = []
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        vec = extract_hand_landmarks_landmark_vector(hand_landmarks.landmark, w, h)
                        all_hands.append(vec)
                        # desenhar pontos da mão
                        for lm in hand_landmarks.landmark:
                            cx, cy = int(lm.x * w), int(lm.y * h)
                            cv2.circle(display, (cx, cy), 3, (0,0,255), -1)
                    # concatena as mãos (até 2 mãos)
                    if len(all_hands) == 1:
                        hand_vec = all_hands[0]
                    elif len(all_hands) >= 2:
                        hand_vec = np.concatenate(all_hands[:2])

                # --- Salvar vetor ---
                now = time.time()
                if now - last_time > DELAY_BETWEEN:
                    combined_vec = np.concatenate([face_vec, hand_vec])
                    fname = os.path.join(DATA_DIR, label, f"{label}_{saved:04d}.npy")
                    np.save(fname, combined_vec)
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
