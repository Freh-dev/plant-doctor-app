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

# Simple model loader
@st.cache_resource
def load_model():
    try:
        # Try ultra light model first
        model = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.success("✅ Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return ["healthy", "diseased"]

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

# Simple advice function (no external dependencies)
def generate_advice(plant, disease):
    """Generate plant care advice without external API calls"""
    advice_templates = {
        "healthy": f"🌱 Your {plant} plant looks healthy! Continue with regular care including proper watering and sunlight.",
        "powdery_mildew": f"🍂 For {plant} with powdery mildew: Remove affected leaves, improve air circulation, and avoid overhead watering.",
        "leaf_spot": f"🍃 For {plant} with leaf spot: Remove damaged leaves, water at the base only, and ensure good drainage.",
        "blight": f"🔥 For {plant} with blight: Remove infected plants, avoid overcrowding, and rotate crops next season.",
        "rust": f"🟫 For {plant} with rust: Remove affected leaves, improve air flow, and avoid wetting foliage.",
        "mosaic": f"🟨 For {plant} with mosaic virus: Remove infected plants, control aphids, and use disease-free seeds.",
    }
    
    # Find matching advice or use general advice
    for key in advice_templates:
        if key in disease.lower():
            return advice_templates[key]
    
    # General advice for unknown diseases
    return f"🌿 For {disease} in {plant}: Remove affected leaves, improve growing conditions, and monitor plant health regularly."

# App UI
st.title("🌿 Plant Doctor - Smart Plant Diagnosis")
st.markdown("Upload a leaf photo for instant disease detection and treatment advice!")

# Check if model loaded successfully
if model is None or not class_names:
    st.error("⚠️ Model not loaded. Please check your files.")
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
        st.image(image, caption="Uploaded Leaf", width='stretch')
    
    # Predict button with fixed width parameter
    if st.button("🔍 Analyze Plant", type="primary", width='stretch'):
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
                
                # Get advice
                advice = generate_advice(plant_name, disease)
                    
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
    
    st.header("📊 Model Info")
    st.metric("Model Status", "✅ Active")
    st.metric("Supported Plants", "30+")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow and Streamlit | Plant Disease Detection AI")
