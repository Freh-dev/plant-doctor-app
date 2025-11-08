# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Simple model loader with better error handling
@st.cache_resource
def load_model():
    model_options = [
        "ultra_light_model.keras",
        "plantvillage_head_cpu_v2_1.h5", 
        "plant_disease_M1.keras"
    ]
    
    for model_path in model_options:
        try:
            if os.path.exists(model_path):
                model = tf.keras.models.load_model(model_path)
                st.sidebar.success(f"✅ Model loaded: {model_path}")
                return model
        except Exception as e:
            st.sidebar.warning(f"⚠️ Failed to load {model_path}: {e}")
            continue
    
    st.error("❌ No working model found!")
    return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except:
        return ["healthy", "diseased"]  # Fallback

# Load resources
model = load_model()
class_names = load_class_names()

# Simple prediction function
def predict_image(image):
    try:
        img = image.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

# App UI
st.title("🌿 Plant Doctor - Smart Plant Diagnosis")
st.markdown("Upload a leaf photo for AI-powered disease detection and treatment advice!")
st.markdown("---")

# Check model
if model is None:
    st.error("""
    ❌ **Model not loaded!**
    
    Please ensure you have at least one of these files in your repository:
    - `ultra_light_model.keras`
    - `plantvillage_head_cpu_v2_1.h5` 
    - `plant_disease_M1.keras`
    """)
    st.stop()

# File upload
uploaded_file = st.file_uploader(
    "📸 Choose a plant leaf image...", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, use_container_width=True)
    
    if st.button("🧠 Analyze Plant", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error(f"Error: {error}")
            else:
                with col2:
                    st.success(f"**Diagnosis:** {disease}")
                    st.success(f"**Confidence:** {confidence:.2%}")
                    
                    # Simple advice based on diagnosis
                    if "healthy" in disease.lower():
                        st.info("**💡 Your plant looks healthy!** Continue regular care.")
                    else:
                        st.warning("""
                        **💡 Treatment Suggestions:**
                        - Remove affected leaves
                        - Improve air circulation  
                        - Avoid overwatering
                        - Use organic fungicides if needed
                        - Monitor plant recovery
                        """)

# Sidebar
with st.sidebar:
    st.header("ℹ️ Setup Required")
    st.markdown("""
    **For AI advice, add OpenAI API key:**
    1. Click 'Manage app' → Settings
    2. Go to 'Secrets'  
    3. Add: `OPENAI_API_KEY = "your-key"`
    4. Redeploy
    """)

st.markdown("---")
st.caption("Plant Disease Detection AI | Built with TensorFlow & Streamlit")
