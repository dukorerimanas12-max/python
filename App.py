import streamlit as st
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from inference_sdk import InferenceHTTPClient
import tempfile

# Initialize Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="7GxkZkhZdGhyrwHvwPMU"
)

st.title("AI-Based Smart Parking And Monitoring System")
st.write("Choose an option: Image or Video detection.")

# Radio button to select source
source_type = st.radio("Select source:", ("Image", "Video"))

# Function to get color for relevant classes
def get_color(cls):
    cls_lower = cls.lower()
    if cls_lower in ['car', 'occupied']:
        return (255, 0, 0)   # Red
    elif cls_lower in ['empty']:
        return (255, 255, 0) # Yellow
    else:
        return None  # Ignore everything else

# Load font for PIL images
try:
    font = ImageFont.truetype("arial.ttf", 30)
except:
    font = ImageFont.load_default()

# Placeholder for frames
frame_placeholder = st.empty()

# Function to check aspect ratio and filter out tall/narrow boxes (likely people)
def is_car_like(det):
    width = det['width']
    height = det['height']
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 2.5:  # ignore tall/narrow boxes
        return False
    return True

# ---------- IMAGE ----------
if source_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["png","jpg","jpeg","bmp"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        file_path = tfile.name

        result = CLIENT.infer(file_path, model_id="parkingcar_detection/3")
        image = Image.open(file_path)
        draw = ImageDraw.Draw(image)

        for det in result['predictions']:
            color = get_color(det['class'])
            if color is None or not is_car_like(det):
                continue

            x1 = det['x'] - det['width'] / 2
            y1 = det['y'] - det['height'] / 2
            x2 = det['x'] + det['width'] / 2
            y2 = det['y'] + det['height'] / 2
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            draw.text((x1, y1 - 35), det['class'], fill=color, font=font)

        st.image(image, caption="Detection Result", use_column_width=True)

# ---------- VIDEO ----------
elif source_type == "Video":
    uploaded_file = st.file_uploader("Upload a video", type=["mp4","avi","mov","mkv"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        file_path = tfile.name

        cap = cv2.VideoCapture(file_path)
        stop = st.button("Stop Video")

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                break

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            cv2.imwrite(temp_file.name, frame)
            result = CLIENT.infer(temp_file.name, model_id="parkingcar_detection/3")

            for det in result['predictions']:
                color = get_color(det['class'])
                if color is None or not is_car_like(det):
                    continue

                x1 = int(det['x'] - det['width']/2)
                y1 = int(det['y'] - det['height']/2)
                x2 = int(det['x'] + det['width']/2)
                y2 = int(det['y'] + det['height']/2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, det['class'], (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()

