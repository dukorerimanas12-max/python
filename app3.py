import streamlit as st
import cv2
import av
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# ===============================
# SETTINGS
# ===============================
MODEL_PATH = "CarModel_float32.tflite"
CONF_THRESHOLD = 0.4
# ===============================

st.title("🚗 Smart Car Parking System")

# Load model once
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        results = model(img, conf=CONF_THRESHOLD)

        for result in results:
            for box in result.boxes:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                class_name = model.names[cls]
                label = f"{class_name}: {conf:.2f}"

                # Custom colors
                if class_name.lower() == "car":
                    color = (255, 0, 0)  # Blue
                elif class_name.lower() == "free":
                    color = (0, 255, 0)  # Green
                else:
                    color = (0, 255, 255)

                # Draw box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="car-detection",
    video_processor_factory=VideoProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)