# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import gdown
import chatbot_helper

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Download model from Google Drive - WITH CORRECT FILE ID
@st.cache_resource
def download_model():
    model_path = "improved_model.keras"
    
    # ✅ CORRECT FILE ID FROM YOUR LINK
    file_id = "1FpFdfl_UFR_6kfbsV3eQYvlYvWMYB-Az"
    
    if not os.path.exists(model_path):
        try:
            with st.spinner("📥 Downloading high-accuracy model (95.48%) from Google Drive..."):
                # Download from Google Drive
                url = f"https://drive.google.com/uc?id={file_id}"
                gdown.download(url, model_path, quiet=False)
                
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / 1024 / 1024
                st.sidebar.success(f"✅ 95.48% Accurate Model Downloaded! ({file_size:.1f} MB)")
                return model_path
            else:
                st.sidebar.error("❌ Download failed - file not created")
                return None
                
        except Exception as e:
            st.sidebar.error(f"❌ Download error: {str(e)}")
            return None
    return model_path

# Load model and class names
@st.cache_resource
def load_model():
    try:
        model_path = download_model()
        if model_path and os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path)
            st.sidebar.success("✅ 95.48% Accurate Model Loaded!")
            return model
    except Exception as e:
        st.sidebar.error(f"❌ Model loading error: {e}")
    
    # Fallback to ultra light model
    try:
        model = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.info("ℹ️ Using fallback ultra light model")
        return model
    except:
        st.sidebar.error("❌ No model files available")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
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
    st.warning("""
    ⚠️ **Model not loaded properly.** 
    - If this is the first time, wait for the model to download from Google Drive
    - This may take 2-3 minutes depending on your internet speed
    - Refresh the page if it doesn't start automatically
    """)
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
    
    st.header("📊 Model Information")
    if model:
        st.metric("Accuracy", "95.48%")
        st.metric("Model Source", "Google Drive")
        st.metric("Status", "✅ Active")
    else:
        st.metric("Status", "⏳ Downloading...")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow and Streamlit | Plant Disease Detection AI")
