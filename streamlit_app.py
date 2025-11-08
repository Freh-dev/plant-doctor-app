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

# Google Drive file ID - let's try multiple possible IDs
FILE_IDS = [
    "1SbKwqGT7AT-J1RaP_eAuBSPG3oGRxDOo",  # Your original ID
    "15bKwqGT7AF_1JRaP_eAuB5PG36GRoDQo",  # ID from error message
]

def download_file_from_google_drive(file_id, destination):
    """Download large files from Google Drive with confirmation token"""
    try:
        URL = "https://docs.google.com/uc?export=download"
        session = requests.Session()

        response = session.get(URL, params={'id': file_id}, stream=True)
        response.raise_for_status()  # Check for HTTP errors
        
        token = get_confirm_token(response)

        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
            response.raise_for_status()

        # Get total file size
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, "wb") as f:
            downloaded = 0
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for chunk in response.iter_content(chunk_size=32768):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Update progress
                    if total_size > 0:
                        progress = downloaded / total_size
                        progress_bar.progress(min(progress, 1.0))
                        status_text.text(f"Downloaded: {downloaded}/{total_size} bytes ({progress:.1%})")
            
            status_text.text(f"✅ Download complete: {downloaded} bytes")
        
        return True
    except Exception as e:
        st.error(f"❌ Download failed: {e}")
        return False

def get_confirm_token(response):
    """Get confirmation token for Google Drive downloads"""
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

@st.cache_resource
def load_model():
    try:
        model_path = "plant_disease_M1.keras"
        
        # Check if file already exists
        if os.path.exists(model_path):
            st.sidebar.info("📁 Model file found locally")
            model = tf.keras.models.load_model(model_path)
            st.sidebar.success("✅ Plant Disease Model Loaded Successfully!")
            return model
        
        # Try to download from Google Drive with different file IDs
        st.sidebar.info("🔍 Searching for model file...")
        
        download_success = False
        for file_id in FILE_IDS:
            with st.sidebar:
                st.info(f"🔄 Trying File ID: {file_id}")
            
            if download_file_from_google_drive(file_id, model_path):
                if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
                    download_success = True
                    st.sidebar.success(f"✅ Downloaded using ID: {file_id}")
                    break
            else:
                # Clean up failed download
                if os.path.exists(model_path):
                    os.remove(model_path)
        
        if not download_success:
            st.error("❌ All download attempts failed")
            st.info("""
            📋 Manual Setup Required:
            1. Download the model manually from your Google Drive
            2. Go to https://drive.google.com and find 'plant_disease_M1.keras'
            3. Download the file
            4. Upload it to your GitHub repository
            5. Refresh this app
            """)
            return None
        
        # Verify the downloaded file
        file_size = os.path.getsize(model_path)
        st.sidebar.info(f"📦 Downloaded file size: {file_size} bytes")
        
        if file_size == 0:
            st.error("❌ Downloaded file is empty")
            os.remove(model_path)
            return None
        
        # Load the model
        model = tf.keras.models.load_model(model_path)
        st.sidebar.success("✅ Plant Disease Model Loaded Successfully!")
        return model
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        
        # Detailed debugging info
        st.info("🔍 Debugging Information:")
        
        # Check current directory
        st.write("📁 Files in current directory:")
        current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
        for file in current_files:
            file_size = os.path.getsize(file)
            st.write(f"- {file} ({file_size} bytes)")
        
        # Check if .keras file exists but is corrupted
        if os.path.exists("plant_disease_M1.keras"):
            file_size = os.path.getsize("plant_disease_M1.keras")
            st.error(f"⚠️ File exists but may be corrupted. Size: {file_size} bytes")
            
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
    
    # Test the download links
    st.info("🔗 Testing Google Drive links...")
    for file_id in FILE_IDS:
        test_url = f"https://drive.google.com/uc?id={file_id}"
        st.markdown(f"- [Try this Google Drive link]({test_url})")
    
    st.info("""
    💡 **Quick Fix**: 
    Since Google Drive downloads are unreliable on Streamlit Cloud, consider:
    1. **Upload the .keras file directly to GitHub** (if under 100MB)
    2. **Use a different file hosting service** like Dropbox or GitHub Releases
    3. **Use the ultra_light_model.keras** that fits GitHub's limits
    """)
    
    st.stop()

# Rest of your app code remains the same...
