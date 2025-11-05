import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import warnings

# Hide deprecation warning
warnings.filterwarnings("ignore", message=".*tf.lite.Interpreter is deprecated.*")

st.set_page_config(page_title="🌾 Crop Disease Detector", layout="centered")
st.title("🌿 Crop Disease Detection (TFLite Model)")
st.write("Upload a leaf image to find out if it's healthy or diseased.")

# Load the TFLite model (✅ replaces .keras loading)
@st.cache_resource
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="crop_disease_cnn.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define your class names (update if needed)
class_names = [
    'Tomato___Healthy',
    'Tomato___Bacterial_spot',
    'Potato___Healthy',
    'Potato___Early_blight',
    'Pepper__bell___Bacterial_spot'
]

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((128,128))
    st.image(img, caption="Uploaded Leaf", use_column_width=True)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    result = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Show result
    if "Healthy" in result:
        st.success(f"🌿 Prediction: **{result}**")
    else:
        st.error(f"🍂 Prediction: **{result}**")

    st.info(f"Confidence: {confidence:.2f}%")
