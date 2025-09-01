# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import tarfile
import os
import chatbot_helper  # Your existing helper

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Load model and class names
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("improved_model.keras")

@st.cache_data
def load_class_names():
    with open("class_names_improved.json", "r") as f:
        return json.load(f)

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (150, 150)

def predict_image(image):
    """Predict plant disease from image"""
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))
    
    return predicted_class, confidence

# App UI
st.title("🌿 Plant Doctor")
st.markdown("Upload a photo of your plant leaf to detect diseases and get treatment advice!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a plant leaf"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
    
    # Predict button
    if st.button("🔍 Analyze Plant", type="primary"):
        with st.spinner("Analyzing..."):
            # Make prediction
            disease, confidence = predict_image(image)
            
            with col2:
                st.subheader("📊 Diagnosis Results")
                st.success(f"**Disease:** {disease}")
                st.success(f"**Confidence:** {confidence:.2%}")
                
                # Get plant name
                if '_' in disease:
                    plant_name = disease.split('_')[0]
                    st.info(f"**Plant Type:** {plant_name}")
                else:
                    plant_name = "plant"
            
            # Get AI advice
            with st.spinner("Getting treatment advice..."):
                advice = chatbot_helper.generate_advice(plant_name, disease)
                
            st.subheader("💡 Treatment Advice")
            st.info(advice)

# Sidebar with instructions
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown("""
    1. **Take a photo** of a plant leaf
    2. **Upload** the image
    3. **Click Analyze** for diagnosis
    4. **Follow** the treatment advice
    """)
    
    st.header("📸 Tips for Best Results")
    st.markdown("""
    - Good lighting ☀️
    - Plain background
    - Clear, focused leaf
    - No shadows or glare
    """)
    
    st.header("🌿 Supported Plants")
    st.markdown("""
    This app can detect diseases in:
    - Tomatoes
    - Potatoes
    - Peppers
    - Apples
    - Corn
    - Grapes
    - And many more!
    """)

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow and Streamlit | Plant Disease Detection AI")