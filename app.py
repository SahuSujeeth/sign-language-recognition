import time
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
import pyttsx3
import threading
from collections import deque, Counter

# ==============================
# Page Configuration
# ==============================
st.set_page_config(page_title="Sign Language Recognition", layout="wide")

# ==============================
# Load Model
# ==============================
model = joblib.load("gesture_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ==============================
# Threaded Speak Function
# ==============================
def speak(text):
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

# ==============================
# Mode Sets
# ==============================
alphabet_set = set([chr(i) for i in range(ord('A'), ord('Z')+1)])
number_set = set([str(i) for i in range(10)])

# ==============================
# Sidebar
# ==============================
st.sidebar.title("⚙ Settings")

mode = st.sidebar.radio("Select Mode", ["Alphabet", "Number"])
voice_enabled = st.sidebar.checkbox("Enable Voice", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Model Information")

st.sidebar.success("Model Loaded Successfully")

st.sidebar.write("🧠 Model: Random Forest Classifier")
st.sidebar.write("✋ Landmarks: 21 Hand Points")
st.sidebar.write("📐 Features: 63 (x, y, z coordinates)")
st.sidebar.write("🔤 Alphabets: A–Z")
st.sidebar.write("🔢 Numbers: 0–9")

# ==============================
# Header
# ==============================
st.markdown("""
<h1 style='text-align:center;color:#22c55e;'>
🤟 Sign Language to Speech Recognition
</h1>

<p style='text-align:center;font-size:18px;'>
Real-time gesture recognition using Machine Learning and MediaPipe
</p>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# ==============================
# Session State
# ==============================
if "prediction_buffer" not in st.session_state:
    st.session_state.prediction_buffer = deque(maxlen=5)

if "last_spoken" not in st.session_state:
    st.session_state.last_spoken = ""

if "camera_running" not in st.session_state:
    st.session_state.camera_running = False

# ==============================
# Start / Stop Buttons
# ==============================
col_start, col_stop = st.columns(2)

with col_start:
    if st.button("▶ Start Camera"):
        st.session_state.camera_running = True

with col_stop:
    if st.button("⛔ Stop Camera"):
        st.session_state.camera_running = False

# ==============================
# Layout
# ==============================
col1, col2 = st.columns([1,2])

# Camera title (ONLY ONCE)
with col2:
    st.markdown("### 📷 Live Camera Feed")
    frame_placeholder = st.empty()

info_placeholder = col1.empty()

# ==============================
# Camera Loop
# ==============================
if st.session_state.camera_running:

    cap = cv2.VideoCapture(0)

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    prev_time = 0

    while st.session_state.camera_running:

        ret, frame = cap.read()
        if not ret:
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        prediction_text = "Detecting..."
        confidence = 0.0
        status = "❌ No Hand Detected"

        if results.multi_hand_landmarks:

            status = "✋ Hand Detected"

            for hand_landmarks in results.multi_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

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

                if confidence > 0.50:

                    if mode == "Alphabet" and predicted_label in alphabet_set:
                        st.session_state.prediction_buffer.append(predicted_label)

                    elif mode == "Number" and predicted_label in number_set:
                        st.session_state.prediction_buffer.append(predicted_label)

                if len(st.session_state.prediction_buffer) == 5:

                    prediction_text = Counter(
                        st.session_state.prediction_buffer
                    ).most_common(1)[0][0]

                    if (
                        voice_enabled
                        and confidence > 0.60
                        and prediction_text != st.session_state.last_spoken
                    ):
                        st.session_state.last_spoken = prediction_text

                        threading.Thread(
                            target=speak,
                            args=(prediction_text,),
                            daemon=True
                        ).start()

        frame_placeholder.image(frame, channels="BGR")

        info_placeholder.markdown(f"""
        ### 🔍 Prediction

        ## {prediction_text}

        Confidence: {confidence:.2f}

        Status: {status}

        FPS: {fps:.2f}
        """)

    cap.release()
    hands.close()

# ==============================
# Sign Reference Guide
# ==============================
st.markdown("---")
st.markdown("## ✋ Sign Reference Guide")

letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
cols = st.columns(13)

for i, letter in enumerate(letters):
    with cols[i % 13]:
        st.markdown(
            f"<div style='text-align:center;font-size:22px;font-weight:bold'>{letter}</div>",
            unsafe_allow_html=True
        )

st.markdown("### Numbers")

numbers = list("0123456789")
cols2 = st.columns(10)

for i, num in enumerate(numbers):
    with cols2[i]:
        st.markdown(
            f"<div style='text-align:center;font-size:22px;font-weight:bold'>{num}</div>",
            unsafe_allow_html=True
        )