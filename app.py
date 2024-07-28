import streamlit as st
from PIL import Image
import sys
import cv2
import numpy as np
import os


sys.path.append('yolov5')

from detect import run


st.title('🧠 Brain Tumor Detection')
st.markdown('<style>body {color: #6a0dad;}</style>', unsafe_allow_html=True)
st.write('Upload an MRI image to detect brain tumor.')

def create_opencv_image_from_stringio(img_stream, cv2_img_flag=1):
    img_stream.seek(0)
    img_array = np.asarray(bytearray(img_stream.read()), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2_img_flag)

sample_images = {
    'Glioma 1': 'sample_images/Glioma(1).jpg',
    'Glioma 2': 'sample_images/Glioma(2).jpg',
    'Glioma 3': 'sample_images/Glioma(3).jpg',
    'Meningioma 1': 'sample_images/Meningioma(1).jpg',
    'Meningioma 2': 'sample_images/Meningioma(2).jpg',
    'Meningioma 3': 'sample_images/Meningioma(3).jpg',
    'No_tumor 1': 'sample_images/No_tumor(1).jpg',
    'No_tumor 2': 'sample_images/No_tumor(2).jpg',
    'No_tumor 3': 'sample_images/No_tumor(3).jpg',
    'Pituitary 1': 'sample_images/Pituitary(1).jpg',
    'Pituitary 2': 'sample_images/Pituitary(2).jpg',
    'Pituitary 3': 'sample_images/Pituitary(3).jpg',
}

st.sidebar.header('Sample Images')
sample_choice = st.sidebar.selectbox('Chose a sample image', list(sample_images.keys()))

if sample_choice:
    st.sidebar.image(sample_images[sample_choice], caption=sample_choice, use_column_width=True)

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None or sample_choice: 
    if uploaded_file is not None:
        open_cv_image = create_opencv_image_from_stringio(uploaded_file)
        
        # saving image temporarily
        temp_filename = 'temp.jpg'
        cv2.imwrite(temp_filename, open_cv_image)
    else:
        temp_filename = sample_images[sample_choice]

    output_dir = 'results'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Predicting using the model
    with st.spinner('🔍 Analyzing the image, please wait...'):
        run(weights='best.pt', source=temp_filename, project=output_dir, name='result', exist_ok=True)
    
 
    result_img_path = os.path.join(output_dir, 'result', os.path.basename(temp_filename))
    
    # Image display
    if os.path.exists(result_img_path):
        detected_image = Image.open(result_img_path)
        st.image(detected_image, caption='Detection Results', use_column_width=True)
    else:
        st.write("Detection failed or no output image found.")
    
    # Clean up
    if uploaded_file is not None:
        os.remove(temp_filename)
