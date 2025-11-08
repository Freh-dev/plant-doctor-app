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

# Load the specific H5 model
@st.cache_resource
def load_model():
    try:
        # Specifically load the H5 model
        model = tf.keras.models.load_model("plantvillage_finetuned_mobilenetv4.h5")
        st.sidebar.success("✅ PlantVillage H5 Model Loaded Successfully!")
        st.sidebar.info(f"🔧 Using: plantvillage_finetuned_mobilenetv4.h5")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading H5 model: {e}")
        
        # Fallback to ultra light model if H5 fails
        try:
            st.sidebar.info("🔄 Trying ultra light model as fallback...")
            model = tf.keras.models.load_model("ultra_light_model.keras")
            st.sidebar.success("✅ Ultra Light Model Loaded (Fallback)")
            return model
        except:
            st.sidebar.error("❌ No working model found!")
            return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        # Provide fallback class names for PlantVillage dataset
        return [
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", "Tomato_Leaf_Mold",
            "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", "Tomato_Target_Spot", 
            "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", "Tomato_healthy",
            "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
            "Corn_(maize)_Northern_Leaf_Blight", "Corn_(maize)_Common_rust_", "Corn_(maize)_healthy",
            "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy",
            "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy"
        ]

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (224, 224)  # Standard size for PlantVillage H5 models

def predict_image(image):
    """Predict plant disease from image using H5 model"""
    try:
        # Resize to model's expected input size
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

# Enhanced advice function for PlantVillage diseases
def generate_advice(plant, disease):
    """Generate plant care advice for PlantVillage dataset diseases"""
    advice_templates = {
        # Tomato diseases
        "bacterial_spot": f"🦠 For {plant} Bacterial Spot: Remove infected leaves, apply copper-based bactericide, avoid overhead watering, and rotate crops.",
        "early_blight": f"🍂 For {plant} Early Blight: Remove affected leaves, apply fungicide, water at soil level, and improve air circulation.",
        "late_blight": f"🔥 For {plant} Late Blight: Remove infected plants immediately, use copper fungicide, avoid wet foliage, and destroy infected material.",
        "leaf_mold": f"🍄 For {plant} Leaf Mold: Improve ventilation, reduce humidity, apply fungicide, and space plants properly.",
        "septoria_leaf_spot": f"🔴 For {plant} Septoria Leaf Spot: Remove infected leaves, apply chlorothalonil, avoid overhead irrigation, and rotate crops.",
        "yellow_leaf_curl": f"🔄 For {plant} Yellow Leaf Curl Virus: Remove infected plants, control whiteflies, use resistant varieties, and destroy infected debris.",
        "mosaic_virus": f"🟨 For {plant} Mosaic Virus: Remove infected plants, control aphids, disinfect tools, and use virus-free seeds.",
        
        # Potato diseases
        "potato_blight": f"🥔 For {plant} Blight: Remove infected plants, apply fungicide, ensure good drainage, and harvest carefully.",
        
        # Corn diseases  
        "northern_leaf_blight": f"🌽 For {plant} Northern Leaf Blight: Remove infected leaves, apply fungicide, rotate crops, and use resistant hybrids.",
        "common_rust": f"🟫 For {plant} Common Rust: Apply fungicide early, remove infected leaves, and avoid late planting.",
        
        # General
        "healthy": f"🌱 Your {plant} plant looks healthy! Continue regular care: proper watering, balanced fertilizer, and pest monitoring."
    }
    
    # Find matching advice
    disease_lower = disease.lower()
    for key in advice_templates:
        if key in disease_lower:
            return advice_templates[key]
    
    # General advice for unknown diseases
    return f"🌿 For {disease} in {plant}: Remove affected leaves, improve air circulation, avoid overwatering, monitor regularly, and consider organic fungicides if needed."

# App UI
st.title("🌿 Plant Doctor - Smart Plant Diagnosis")
st.markdown("**Using PlantVillage H5 Model for accurate disease detection**")

# Check if model loaded successfully
if model is None:
    st.error("""
    ❌ Model not loaded. Please ensure you have:
    - `plantvillage_finetuned_mobilenetv4.h5` in your repository
    - Or `ultra_light_model.keras` as fallback
    """)
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a plant leaf (recommended size: 224x224 pixels)"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf", width='stretch')
        st.info(f"📏 Image size: {image.size}")
        st.info(f"🎯 Model expects: {img_size}")
    
    # Predict button
    if st.button("🔍 Analyze Plant", type="primary", width='stretch'):
        with st.spinner("Analyzing with PlantVillage H5 Model..."):
            # Make prediction
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error(f"❌ Prediction error: {error}")
            else:
                with col2:
                    st.subheader("📊 Diagnosis Results")
                    
                    # Display with confidence indicators
                    if confidence > 0.8:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} 🎯 High")
                    elif confidence > 0.6:
                        st.warning(f"**Disease:** {disease}")
                        st.warning(f"**Confidence:** {confidence:.2%} ⚠️ Medium")
                    else:
                        st.info(f"**Disease:** {disease}")
                        st.info(f"**Confidence:** {confidence:.2%} 🔍 Low")
                    
                    # Get plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0].title()
                        st.info(f"**Plant Type:** {plant_name}")
                    else:
                        plant_name = "plant"
                
                # Get advice
                advice = generate_advice(plant_name, disease)
                    
                st.subheader("💡 Treatment Advice")
                st.info(advice)

# Sidebar with PlantVillage-specific info
with st.sidebar:
    st.header("🔬 Model Information")
    st.metric("Active Model", "PlantVillage H5")
    st.metric("Input Size", "224×224")
    st.metric("Dataset", "PlantVillage")
    
    st.header("🌿 Supported Plants")
    st.markdown("""
    - **Tomatoes** (10 diseases)
    - **Potatoes** (3 conditions) 
    - **Corn/Maize** (3 diseases)
    - **Peppers** (2 conditions)
    - **Apples** (4 diseases)
    """)
    
    st.header("📸 Image Tips")
    st.markdown("""
    - Use **224×224** pixels if possible
    - **Clear, focused** leaf close-up
    - **Plain background** recommended
    - **Good lighting** without shadows
    """)

# Footer
st.markdown("---")
st.caption("Powered by PlantVillage H5 Model | Built with TensorFlow & Streamlit | Plant Disease Detection AI")
