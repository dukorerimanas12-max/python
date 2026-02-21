import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
from inference_sdk import InferenceHTTPClients

# -----------------------
# Load the trained model
# -----------------------
@st.cache_resource
def load_model():
    model = YOLO("CarModel.pt")   # your trained model
    return model

model = load_model()

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Car Detection App", layout="wide")
st.title("🚗 AI-Based Smart Parking and Monitoring System 🚗 ")

# -----------------------
# Confidence slider
# -----------------------
confidence_threshold = st.slider(
    "Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01
)

# Radio button for input type
input_type = st.radio("Select Input Type:", ["Image", "Video", "Webcam"])

# =======================
# Shared function
# =======================
def process_detection(results, threshold):
    """Return detection details with custom label and smaller bounding box"""
    boxes = results[0].boxes

    if boxes is None or len(boxes) == 0:
        return None

    data = []
    for box, cls, conf in zip(
        boxes.xyxy.cpu().numpy(),
        boxes.cls.cpu().numpy(),
        boxes.conf.cpu().numpy()
    ):
        if conf >= threshold:
            x1, y1, x2, y2 = box

            # === SHRINK BOUNDING BOX ===
            shrink = 10
            x1 += shrink
            y1 += shrink
            x2 -= shrink
            y2 -= shrink

            # === USE CUSTOM LABEL ===
            data.append({
                "Class": "occupied",
                "Confidence": float(conf),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2)
            })
    return data


# -----------------------
# IMAGE INPUT
# -----------------------
if input_type == "Image":
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    
    if uploaded is not None:
        st.subheader("Original Image")
        image = Image.open(uploaded)
        st.image(image, use_column_width=True)
        img_array = np.array(image)
        
        st.subheader("Detection Results")
        results = model(img_array, conf=confidence_threshold)

        # YOLO plotted result (unchanged color)
        result_image = results[0].plot()
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        st.image(result_image, caption="Detected Cars", use_column_width=True)

        # Detection table
        st.subheader("Detection Details")
        data = process_detection(results, confidence_threshold)
        if data:
            st.dataframe(data)
        else:
            st.write("No objects detected above confidence threshold.")


# -----------------------
# VIDEO INPUT
# -----------------------
elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_video.read())
        
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        table_placeholder = st.empty()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence_threshold)
            result_frame = results[0].plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
            
            stframe.image(result_frame, channels="RGB", use_column_width=True)

            # Detection details
            data = process_detection(results, confidence_threshold)
            table_placeholder.dataframe(data if data else [])
        
        cap.release()


# -----------------------
# WEBCAM INPUT
# -----------------------
elif input_type == "Webcam":
    stframe = st.empty()
    table_placeholder = st.empty()
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        st.info("Press 'Stop Webcam' to end the stream.")
        stop_button = st.button("Stop Webcam")
        
        while cap.isOpened() and not stop_button:
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=confidence_threshold)
            result_frame = results[0].plot()
            result_frame = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

            stframe.image(result_frame, channels="RGB", use_column_width=True)

            # Detection details
            data = process_detection(results, confidence_threshold)
            table_placeholder.dataframe(data if data else [])

        cap.release()