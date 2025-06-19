import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import tempfile
import os

# ----------------- Page Config (Must be First) ------------------
st.set_page_config(
    page_title="🎬 Human Action Recognition",
    layout="centered",
    page_icon="🧍"
)

# ----------------- Custom Function for Lambda -------------------
@tf.keras.utils.register_keras_serializable()
def repeat_channels(x):
    return tf.repeat(x, repeats=3, axis=-1)

# ----------------- Load Model and Label Encoder -----------------
@st.cache_resource
def load_model_and_encoder():
    model = load_model(
        "D:/ML PROJECTS/Human Action Detection/human_action_model.h5",
        custom_objects={'repeat_channels': repeat_channels}
    )
    with open("D:/ML PROJECTS/Human Action Detection/label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    return model, le

model, label_encoder = load_model_and_encoder()

# ----------------- UI Design -----------------
st.title("🎬 Human Action Detection")
st.markdown("Upload a short video and let the AI model predict the human action.")

uploaded_video = st.file_uploader("📹 Upload a video file", type=["mp4", "avi"])

# ----------------- Frame Extraction -----------------
def extract_frames(video_path, target_size=(64, 64), max_frames=30):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened() and count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, target_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(frame)
        count += 1
    cap.release()

    frames = np.array(frames)
    if frames.shape[0] < max_frames:
        pad = np.zeros((max_frames - frames.shape[0], *target_size))
        frames = np.concatenate([frames, pad])
    return frames / 255.0  # normalize

# ----------------- Predict -----------------
if uploaded_video is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        video_path = tmp.name

    st.video(uploaded_video)

    st.info("⏳ Extracting frames and predicting...")
    frames = extract_frames(video_path)  # (30, 64, 64)
    frames = frames[..., np.newaxis]     # Add channel dim → (30, 64, 64, 1)
    frames = np.expand_dims(frames, axis=0)  # Add batch dim → (1, 30, 64, 64, 1)

    prediction = model.predict(frames)
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])
    st.success(f"🎯 Predicted Action: **{predicted_class[0].capitalize()}**")

    # Show top 3 probabilities
    st.subheader("🔍 Top 3 Predictions")
    probs = prediction[0]
    top_3_idx = probs.argsort()[-3:][::-1]
    for i in top_3_idx:
        label = label_encoder.inverse_transform([i])[0]
        st.write(f"**{label.capitalize()}** - {probs[i]*100:.2f}%")

    os.remove(video_path)
