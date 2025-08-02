import cv2
import mediapipe as mp
import os

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Create dataset folder
DATA_DIR = 'dataset'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Ask user for sign name
sign_name = input("Enter the sign name (e.g., A, B, Hello): ").strip()
sign_path = os.path.join(DATA_DIR, sign_name)

if not os.path.exists(sign_path):
    os.makedirs(sign_path)

# Start webcam
cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Save cropped hand image
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)

            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                cv2.imwrite(f"{sign_path}/{count}.jpg", cropped)
                count += 1

    cv2.putText(frame, f"Collected: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Data Collection", frame)

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
