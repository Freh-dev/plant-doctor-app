# streamlit_app.py
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import requests
import os
import chatbot_helper

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Google Drive file ID
FILE_ID = "1SbKwqGT7AT-J1RaP_eAuBSPG3oGRxDOo"

def download_file_from_google_drive(file_id, destination):
    """Download large files from Google Drive with confirmation token"""
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': file_id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    """Get confirmation token for Google Drive downloads"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    """Save the downloaded content with progress bar"""
    CHUNK_SIZE = 32768
    
    # Get total file size
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, "wb") as f:
        downloaded = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                
                # Update progress
                if total_size > 0:
                    progress = downloaded / total_size
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")

@st.cache_resource
def load_model():
    try:
        model_path = "plant_disease_M1.keras"
        
        if not os.path.exists(model_path):
            with st.spinner("📥 Downloading plant disease model from Google Drive (2.4 MB)..."):
                download_file_from_google_drive(FILE_ID, model_path)
                st.sidebar.success("✅ Model downloaded successfully!")
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.sidebar.success("✅ Plant Disease Model Loaded Successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.info("""
        📋 Troubleshooting Google Drive download:
        1. Make sure the file is shared with 'Anyone with the link'
        2. Try accessing the link manually in browser first
        3. Check if the file ID is correct
        """)
        
        # Fallback: Try to provide manual download instructions
        st.error("""
        🔧 Manual Download Option:
        If automatic download fails, please:
        1. Download manually from: https://drive.google.com/uc?id=1SbKwqGT7AT-J1RaP_eAuBSPG3oGRxDOo
        2. Upload the file to your Streamlit app as 'plant_disease_M1.keras'
        3. Refresh the app
        """)
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
    
    # Test the download link
    st.info("🔗 Testing Google Drive link...")
    test_url = f"https://drive.google.com/uc?id={FILE_ID}"
    st.markdown(f"Try accessing this link manually: [Google Drive File]({test_url})")
    
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

# Sidebar
with st.sidebar:
    st.header("📊 Model Info")
    st.metric("Model Source", "Google Drive")
    st.metric("File", "plant_disease_M1.keras")
    st.metric("File Size", "2.4 MB")
    st.metric("Status", "🔄 Download Required")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using TensorFlow and Streamlit | Plant Disease Detection AI")
