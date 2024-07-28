import streamlit as st
from PIL import Image
import torch
import sys
from torchvision import transforms
import cv2
import numpy as np
from detect import run

sys.path.append('yolov5')

from detect import run

st.title('Brain Tumor Detection')
st.write('Upload an MRI image to detect brain tumor.')

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)


uploaded_file = st.file_uploader('Chose an image...', type=['jpg', 'jpeg', 'png'])

for n, img_file_buffer in enumerate(uploaded_file):
  if img_file_buffer is not None:
    open_cv_image = create_opencv_image_from_stringio(img_file_buffer)
    im0 = run(source=open_cv_image, conf_thres=0.25, weights="best.pt")
    if im0 is not None:
        st.image(im0, channels="BGR", caption=f'Detection Results ({n+1}/{len(uploaded_file)})')
    pass