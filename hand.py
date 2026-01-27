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
    Middle = landmarks[12]
    Ring = landmarks[16]
    Pinky = landmarks[20]
    Rist = landmarks[0]
    tips = [landmarks[i] for i in [8, 12, 16, 20]]
    pips = [landmarks[i] for i in [6, 10, 14, 18]]

    T_I_dist = np.linalg.norm(
        np.array([thumb.x, thumb.y]) -
        np.array([index.x, index.y])
    )
    Tip_rist_dist = np.linalg.norm(
    np.array([tips[0].x - Rist.x, tips[0].y - Rist.y])
)
    Pip_rist_dist = np.linalg.norm(np.array([pips[0].x - Rist.x, pips[0].y - Rist.y])
)
    print("Tip_rist_dist", Tip_rist_dist)
    print("T_I_DIDST",T_I_dist)
    print("Pip_rist_dist", Pip_rist_dist)
    if Tip_rist_dist < Pip_rist_dist:
        return "Fist"
    if T_I_dist < 0.1:
        return "pinch"
    else:
        return "Open_hand"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
            )

            gesture = recognize_gesture(hand_landmarks.landmark)

            print(f"[GESTURE EVENT] {gesture}")

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            lm = hand.landmark

        
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        x1, y1 = int(min(xs) * w), int(min(ys) * h)
        x2, y2 = int(max(xs) * w), int(max(ys) * h)

        gesture = recognize_gesture(lm)

        # 
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 
        label_y = max(0, y1 - 10)
        cv2.putText(frame, gesture, (x1, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Hand Service", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
