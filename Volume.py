import cv2
import mediapipe as mp
import numpy as np
import subprocess

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils


def set_volume(volume):
    volume = max(0, min(100, volume))
    script = f"set volume output volume {volume}"
    subprocess.run(["osascript", "-e", script])


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera could not be opened.")
    exit()

while True:
    success, img = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            h, w, c = img.shape

            ix, iy = int(landmarks[8].x * w), int(landmarks[8].y * h)
            tx, ty = int(landmarks[4].x * w), int(landmarks[4].y * h)

            cv2.circle(img, (ix, iy), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (tx, ty), 10, (255, 0, 0), cv2.FILLED)
            cv2.line(img, (ix, iy), (tx, ty), (255, 0, 0), 3)

            distance = np.sqrt((ix - tx) ** 2 + (iy - ty) ** 2)
            vol = np.interp(distance, [20, 200], [0, 100])
            set_volume(vol)

            cv2.putText(img, f'Distance: {int(distance)}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            thumb_is_down = landmarks[4].y > landmarks[2].y
            index_is_down = landmarks[8].y > landmarks[6].y
            middle_is_down = landmarks[12].y > landmarks[10].y
            ring_is_down = landmarks[16].y > landmarks[14].y
            pinky_is_down = landmarks[20].y > landmarks[18].y

            if thumb_is_down and index_is_down and middle_is_down and ring_is_down and pinky_is_down:
                print("All fingers are down. Closing program.")
                cap.release()
                cv2.destroyAllWindows()
                exit()

    cv2.imshow('Img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
