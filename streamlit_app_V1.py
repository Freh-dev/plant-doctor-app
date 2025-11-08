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

# Simple model loader - use what works
@st.cache_resource
def load_model():
    try:
        # Use the reliable ultra light model
        model = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.success("✅ Plant Doctor AI Ready!")
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
            "Tomato_healthy", "Tomato_early_blight", "Tomato_late_blight",
            "Potato_healthy", "Potato_early_blight", "Potato_late_blight",
            "Corn_healthy", "Corn_common_rust", "Pepper_healthy"
        ]

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

# Simple, clear advice
def generate_advice(plant, disease):
    """Generate clear plant care advice"""
    
    advice_templates = {
        "healthy": f"""
        **🌱 Your {plant} plant looks great!**
        - Continue regular watering and care
        - Keep monitoring for any changes
        - Maintain good sunlight exposure
        """,
        
        "early_blight": f"""
        **🍂 Early Blight Detected in {plant}**
        - Remove affected leaves
        - Improve air circulation  
        - Water at the base only
        - Apply organic fungicide if needed
        """,
        
        "late_blight": f"""
        **🚨 Late Blight Detected in {plant}**
        - Remove infected plants immediately
        - Avoid overhead watering
        - Improve spacing between plants
        - Use recommended fungicides
        """,
        
        "bacterial_spot": f"""
        **🦠 Bacterial Spot in {plant}**
        - Remove infected leaves
        - Avoid working with wet plants
        - Use copper-based sprays
        - Water in the morning
        """
    }
    
    # Find matching advice
    disease_lower = disease.lower()
    for key in advice_templates:
        if key in disease_lower:
            return advice_templates[key]
    
    # General advice
    return f"""
    **🌿 Treatment for {disease} in {plant}:**
    - Remove affected leaves
    - Improve growing conditions
    - Monitor plant recovery
    - Consult local garden center if needed
    """

# Clean, professional UI
st.title("🌿 Plant Doctor")
st.markdown("### Get instant diagnosis for your plant's health")
st.markdown("Upload a photo of your plant leaf to identify diseases and get treatment advice.")

# Check if model loaded
if model is None:
    st.error("Service temporarily unavailable. Please try again later.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image", 
    type=["jpg", "jpeg", "png"],
    help="Select a clear photo of a plant leaf"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your plant leaf", use_container_width=True)
    
    # Analyze button
    if st.button("Analyze Plant Health", type="primary", use_container_width=True):
        with st.spinner("Analyzing your plant..."):
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error("Analysis failed. Please try another image.")
            else:
                with col2:
                    st.subheader("Diagnosis Results")
                    
                    # Clean confidence display
                    if confidence > 0.8:
                        st.success(f"**Condition:** {disease.replace('_', ' ').title()}")
                        st.success(f"**Confidence:** {confidence:.0%}")
                    elif confidence > 0.6:
                        st.warning(f"**Condition:** {disease.replace('_', ' ').title()}")
                        st.warning(f"**Confidence:** {confidence:.0%}")
                    else:
                        st.info(f"**Condition:** {disease.replace('_', ' ').title()}")
                        st.info(f"**Confidence:** {confidence:.0%}")
                    
                    # Get plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0]
                    else:
                        plant_name = "plant"
                
                # Get advice
                advice = generate_advice(plant_name, disease)
                st.subheader("Care Instructions")
                st.info(advice)

# Simple sidebar
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    1. **Take a photo** of a plant leaf
    2. **Upload** the image
    3. **Get instant** diagnosis
    4. **Follow** care instructions
    """)
    
    st.header("Tips")
    st.markdown("""
    - Clear, well-lit photos work best
    - Focus on the affected leaves
    - Plain background recommended
    """)
    
    st.header("Supported Plants")
    st.markdown("""
    - Tomatoes
    - Potatoes  
    - Corn
    - Peppers
    - Many more
    """)

# Clean footer
st.markdown("---")
st.caption("Plant health analysis powered by AI")
