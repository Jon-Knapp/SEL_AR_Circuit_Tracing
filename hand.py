# need pip install mediapipe==0.10.9
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

def recognize_gesture(landmarks):
    thumb = landmarks[4]
    index = landmarks[8]

    dist = np.linalg.norm(
        np.array([thumb.x, thumb.y]) -
        np.array([index.x, index.y])
    )

    if dist < 0.25:
        return "PINCH"
    else:
        return "OPEN_HAND"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            gesture = recognize_gesture(hand_landmarks.landmark)

            print(f"[GESTURE EVENT] {gesture}")

    cv2.imshow("Hand Service", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
