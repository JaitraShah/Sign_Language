import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import os

# Load trained model
model = load_model('sign_model.h5')

# Load classes (A–E)
DATA_DIR = 'dataset'
classes = sorted(os.listdir(DATA_DIR))  # Class labels from dataset folder

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

IMG_SIZE = 100

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get bounding box around hand
            x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
            y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
            y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]

            x_min, y_min = int(x_min), int(y_min)
            x_max, y_max = int(x_max), int(y_max)

            # Crop and preprocess
            cropped = frame[y_min:y_max, x_min:x_max]
            if cropped.size > 0:
                img = cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))
                img = img.astype('float32') / 255.0
                img = np.expand_dims(img, axis=0)

                # Predict
                pred = model.predict(img)
                class_id = np.argmax(pred)
                prediction = classes[class_id]

    # Show prediction on screen
    cv2.putText(frame, f"Prediction: {prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Sign Prediction", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
