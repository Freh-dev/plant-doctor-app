# streamlit_app.py - CORRECTED VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Set page config FIRST
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# THEN define cached functions
@st.cache_resource
def load_model():
    try:
        # Use the fixed MobileNetV2 model
        model = tf.keras.models.load_model("plantvillage_mobilenetv2_fixed.h5")
        st.sidebar.success("✅ Advanced AI Model Loaded!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return [
            "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
            "Blueberry_healthy", "Cherry_healthy", "Cherry_Powdery_mildew", 
            "Corn_Common_rust", "Corn_Gray_leaf_spot", "Corn_Healthy", "Corn_Northern_Leaf_Blight",
            "Grape_Black_rot", "Grape_Esca", "Grape_Healthy", "Grape_Leaf_blight",
            "Orange_Haunglongbing", "Peach_Healthy", "Peach_Bacterial_spot",
            "Pepper_bell_Bacterial_spot", "Pepper_bell_Healthy",
            "Potato_Early_blight", "Potato_Healthy", "Potato_Late_blight",
            "Raspberry_Healthy", "Soybean_Healthy", "Squash_Powdery_mildew",
            "Strawberry_Healthy", "Strawberry_Leaf_scorch",
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy",
            "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
            "Tomato_Mosaic_virus"
        ]

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (128, 128)  # MobileNetV2 0.50_128 expects 128x128 input

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

def generate_advice(plant, disease):
    """Generate plant care advice"""
    advice_templates = {
        "healthy": f"🌱 Your {plant} plant looks healthy! Continue regular care.",
        "early_blight": f"🍂 {plant} Early Blight: Remove affected leaves, improve air circulation.",
        "late_blight": f"🔥 {plant} Late Blight: Remove infected plants immediately.",
        "bacterial_spot": f"🦠 {plant} Bacterial Spot: Apply copper spray, avoid wet leaves."
    }
    
    disease_lower = disease.lower()
    for key in advice_templates:
        if key in disease_lower:
            return advice_templates[key]
    
    return f"🌿 For {disease} in {plant}: Remove affected leaves and improve growing conditions."

# App UI
st.title("🌿 Plant Doctor")
st.markdown("Upload a plant leaf photo for instant diagnosis")

if model is None or not class_names:
    st.error("Service temporarily unavailable. Please check the model files.")
    st.stop()

uploaded_file = st.file_uploader(
    "Choose a plant leaf image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your plant leaf", use_container_width=True)
    
    if st.button("Analyze Plant", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error("Analysis failed. Please try another image.")
            else:
                with col2:
                    st.subheader("Diagnosis")
                    st.success(f"**Condition:** {disease.replace('_', ' ').title()}")
                    st.success(f"**Confidence:** {confidence:.0%}")
                
                advice = generate_advice("plant", disease)
                st.subheader("Care Instructions")
                st.info(advice)

# Sidebar
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. Take a clear leaf photo
    2. Upload the image  
    3. Get instant diagnosis
    4. Follow care instructions
    """)
    
    st.header("Supported Plants")
    st.markdown("""
    - Tomatoes
    - Potatoes
    - Corn
    - Peppers
    - Apples
    - Grapes
    - And many more!
    """)

st.markdown("---")
st.caption("AI-powered plant health analysis")
