# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper  # Your existing helper

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Load model and class names - UPDATED FOR ULTRA LIGHT MODEL
@st.cache_resource
def load_model():
    try:
        # Use the ultra_light_model.keras (109 KB) that fits Streamlit limits
        model = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.success("✅ Ultra Light Model Loaded (109 KB)")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("📋 Make sure 'ultra_light_model.keras' is in your repository")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return []

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (150, 150)

def predict_image(image):
    """Predict plant disease from image"""
    try:
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

# App UI
st.title("🌿 Plant Doctor")
st.markdown("Upload a photo of your plant leaf to detect diseases and get treatment advice!")

# Check if model loaded successfully
if model is None or not class_names:
    st.warning("⚠️ Model not loaded. Please check your files.")
    st.stop()

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
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error(f"❌ Prediction error: {error}")
            else:
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
    
    st.header("📊 Model Info")
    st.metric("Model Size", "109 KB")
    st.metric("File", "ultra_light_model.keras")
    st.metric("Status", "✅ Deployment Ready")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow and Streamlit | Plant Disease Detection AI")
