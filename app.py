import streamlit as st
from PIL import Image
import torch
import sys
import cv2
import numpy as np
import os

# Add YOLOv5 directory to path before importing run
sys.path.append('yolov5')

from detect import run

st.title('Brain Tumor Detection')
st.write('Upload an MRI image to detect brain tumor.')

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    open_cv_image = create_opencv_image_from_stringio(uploaded_file)
    
    # Save the uploaded image temporarily
    temp_filename = 'temp.jpg'
    cv2.imwrite(temp_filename, open_cv_image)
    
    # Define output directory
    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run detection
    run(weights='best.pt', source=temp_filename, project=output_dir, name='result', exist_ok=True)
    
    # Construct the path to the output image
    result_img_path = os.path.join(output_dir, 'result', os.path.basename(temp_filename))
    
    # Load and display the image with detections
    if os.path.exists(result_img_path):
        detected_image = Image.open(result_img_path)
        st.image(detected_image, caption='Detection Results', use_column_width=True)
    else:
        st.write("Detection failed or no output image found.")
    
    # Clean up
    os.remove(temp_filename)
    # Remove results directory if needed
    # os.remove(result_img_path)
