import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
import time
from collections import deque, Counter

# ===============================
# Load Model
# ===============================

model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ===============================
# Speak Function (Safe Version)
# ===============================

def speak(text):
    local_engine = pyttsx3.init()
    local_engine.setProperty('rate', 150)
    local_engine.say(text)
    local_engine.runAndWait()
    local_engine.stop()

# ===============================
# Mode Setup
# ===============================

alphabet_set = set([chr(i) for i in range(ord('A'), ord('Z')+1)])
number_set = set([str(i) for i in range(10)])

mode = "ALPHABET"   # Change to "NUMBER" if needed

# ===============================
# MediaPipe Setup
# ===============================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

prediction_buffer = deque(maxlen=5)
last_spoken = ""
last_time = time.time()

# ===============================
# Main Loop
# ===============================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    prediction_text = "Detecting..."
    confidence = 0.0

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract landmarks
            data = []
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                data.append(lm.x - wrist.x)
                data.append(lm.y - wrist.y)
                data.append(lm.z - wrist.z)

            landmarks = np.array(data).reshape(1, -1)

            probs = model.predict_proba(landmarks)
            confidence = np.max(probs)
            prediction_index = np.argmax(probs)

            predicted_label = label_encoder.inverse_transform([prediction_index])[0]

            # Mode filtering
            if confidence > 0.50:
                if mode == "ALPHABET" and predicted_label in alphabet_set:
                    prediction_buffer.append(predicted_label)
                elif mode == "NUMBER" and predicted_label in number_set:
                    prediction_buffer.append(predicted_label)

            # Majority vote smoothing
            if len(prediction_buffer) == 5:
                prediction_text = Counter(prediction_buffer).most_common(1)[0][0]

                current_time = time.time()

                # 🔊 Speech Condition
                if (
                    confidence > 0.60
                    and prediction_text != last_spoken
                    and current_time - last_time > 1.5
                ):
                    last_spoken = prediction_text
                    last_time = current_time
                    threading.Thread(
                        target=speak,
                        args=(prediction_text,),
                        daemon=True
                    ).start()

    cv2.putText(frame, f"Prediction: {prediction_text}",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2)

    cv2.imshow("Live Gesture Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
hands.close()
cv2.destroyAllWindows()