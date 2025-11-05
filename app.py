import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

st.set_page_config(page_title="🌾 Crop Disease Detector", layout="centered")
st.title("🌿 Crop Disease Detection using CNN")
st.write("Upload a leaf image to find out if it's healthy or diseased.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('crop_disease_cnn.keras')
model = load_model()

# Class names (edit to match your dataset)
class_names = [
    'Tomato___Healthy',
    'Tomato___Bacterial_spot',
    'Potato___Healthy',
    'Potato___Early_blight',
    'Pepper__bell___Bacterial_spot'
]

uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf", use_column_width=True)
    img = img.resize((128,128))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)*100
    st.success(f"🌱 Prediction: **{result}**")
    st.info(f"Confidence: {confidence:.2f}%")
