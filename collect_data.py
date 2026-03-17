import cv2
import mediapipe as mp
import csv
import os

# Configuration
LABEL = input("Enter label (A-Z or 0-9): ")
SAMPLES = 300  # number of samples per class

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not opened!")
    exit()

file_exists = os.path.isfile("dataset.csv")

with open("dataset.csv", mode='a', newline='') as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(63):
            header.append(f"f{i}")
        header.append("label")
        
        writer.writerow(header)

    count = 0

    print("Show the gesture for label:", LABEL)
    print("Collecting samples...")

    while count < SAMPLES:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                # Extract landmarks
                data = []
                wrist = hand_landmarks.landmark[0]

                for lm in hand_landmarks.landmark:
                    # Normalize relative to wrist
                    data.append(lm.x - wrist.x)
                    data.append(lm.y - wrist.y)
                    data.append(lm.z - wrist.z)

                data.append(LABEL)
                writer.writerow(data)

                count += 1
                print(f"Collected: {count}/{SAMPLES}")

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        cv2.putText(frame, f"Label: {LABEL} | Count: {count}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        cv2.imshow("Data Collection", frame)
        cv2.waitKey(1)
        cv2.setWindowProperty("Data Collection", cv2.WND_PROP_TOPMOST, 1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
hands.close()
cv2.destroyAllWindows()

print("Data collection completed for label:", LABEL)