import cv2
import numpy as np
import mediapipe as mp
import pygame
import threading
import time
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #extreme error only tensorflow

EAR_THRESHOLD = 0.18
CONSEC_FRAMES = 25
COUNTER = 0
alarm_playing = False


pygame.mixer.init()  #audio_play
try:
    alarm_sound = pygame.mixer.Sound("alarm.wav")
except:
    alarm_sound = None
    print("⚠ alarm.wav not found, running without sound")

def play_alarm():
    global alarm_playing
    while alarm_playing:
        if alarm_sound:
            alarm_sound.play()
            


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)


#eye aspect ratio
def calculate_ear(landmarks, eye_idx):
    v1 = np.linalg.norm(np.array(landmarks[eye_idx[1]]) - np.array(landmarks[eye_idx[5]]))
    v2 = np.linalg.norm(np.array(landmarks[eye_idx[2]]) - np.array(landmarks[eye_idx[4]]))
    h  = np.linalg.norm(np.array(landmarks[eye_idx[0]]) - np.array(landmarks[eye_idx[3]]))
    return (v1 + v2) / (2.0 * h)

#video capture
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            coords = [(int(l.x * w), int(l.y * h)) for l in face_landmarks.landmark]
            left_ear = calculate_ear(coords, [362, 385, 387, 263, 373, 380])
            right_ear = calculate_ear(coords, [33, 160, 158, 133, 153, 144])
            avg_ear = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                COUNTER += 1
                if COUNTER >= CONSEC_FRAMES:
                    cv2.putText(frame, "WAKE UP!", (w//3, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

                    if not alarm_playing:
                        alarm_playing = True
                        threading.Thread(target=play_alarm, daemon=True).start()
            else:
                COUNTER = 0
                alarm_playing = False

            color = (0, 255, 0) if avg_ear > EAR_THRESHOLD else (0, 0, 255)
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
