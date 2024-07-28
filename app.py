import streamlit as st
import torch
from PIL import Image
import numpy as np
from pathlib import Path
import os

# Function to load the YOLOv5 model
def load_model(weights_path):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    return model

# Function to run detection
def run_detection(image, model):
    # Convert the image to RGB
    img_rgb = image.convert('RGB')
    
    # Run detection
    results = model(img_rgb)

    return results

# Load the model once
model = load_model('best.pt')

# Streamlit UI
st.title('YOLOv5 Object Detection')
st.write("Upload an image to detect objects")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Running detection...")
    
    # Run detection
    results = run_detection(image, model)
    
    # Show results
    st.write("Detected objects:")
    results.render()  # Render the detected bounding boxes and labels on the image
    
    # Convert the result image to numpy array
    result_img = np.array(results.ims[0])
    
    # Display the result image with detections
    st.image(result_img, caption='Detection Results', use_column_width=True)
