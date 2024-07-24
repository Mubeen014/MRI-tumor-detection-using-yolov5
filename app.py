import streamlit as st
from PIL import Image
import torch
from torchvision import transforms

model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt', source='local')


st.title('Brain Tumor Detection')
st.write('Upload an MRI image to detect brain tumor.')

uploaded_file = st.file_uploader('Chose an image...', type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded image', use_column_width=True)
    st.write("")
    st.write("Classifying")

    img = transforms.ToTensor()(image)
    results = model([img])
    st.write(results.pandas().xyxy[0])