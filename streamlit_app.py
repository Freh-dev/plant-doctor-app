# streamlit_app.py - IMPROVED VERSION WITH ENHANCED VALIDATION AND ERROR HANDLING
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# 🆕 EfficientNet preprocessing (must match training)
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------- CONFIGURATION ----------------------- #
class AppConfig:
    MODEL_PATH = "plant_disease_final_model.keras"
    CLASS_NAMES_PATH = "class_names_final.json"
    MAX_FILE_SIZE_MB = 200
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    DEFAULT_IMG_SIZE = (224, 224)
    CONFIDENCE_THRESHOLDS = {
        "low": 0.4,
        "medium": 0.75,
        "high": 0.9
    }

class PlantStatus:
    HEALTHY = "healthy"
    DISEASED = "diseased"

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- STYLING --------------------------- #
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .upload-area {
        border: 3px dashed #3CB371;
        border-radius: 15px;
        padding: 3rem 1.5rem;
        text-align: center;
        background: #F0FFF0;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3CB371;
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2E8B57, #228B22);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3CD, #FFEAA7);
        border: 2px solid #FFA500;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #F8D7DA, #F5C6CB);
        border: 2px solid #DC3545;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .success-box {
        background: linear-gradient(135deg, #D1ECF1, #B8E6B8);
        border: 2px solid #28A745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- SESSION STATE --------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

# ----------------------- VALIDATION HELPERS ---------------- #
def validate_environment():
    """Check all required dependencies and configurations"""
    missing_files = []
    
    if not os.path.exists(AppConfig.MODEL_PATH):
        missing_files.append(AppConfig.MODEL_PATH)
    
    if not os.path.exists(AppConfig.CLASS_NAMES_PATH):
        missing_files.append(AppConfig.CLASS_NAMES_PATH)
    
    return missing_files

def validate_image_file(uploaded_file):
    """Validate uploaded image file"""
    errors = []
    
    # Check file size
    file_size = len(uploaded_file.getvalue())
    max_size_bytes = AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024
    
    if file_size > max_size_bytes:
        errors.append(f"File size exceeds {AppConfig.MAX_FILE_SIZE_MB}MB limit")
    
    # Check file format
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in AppConfig.SUPPORTED_FORMATS:
        errors.append(f"Unsupported file format. Use: {', '.join(AppConfig.SUPPORTED_FORMATS)}")
    
    return errors

def validate_image_content(image):
    """Validate image content and dimensions"""
    errors = []
    
    # Check image dimensions
    min_dimension = 50
    if image.size[0] < min_dimension or image.size[1] < min_dimension:
        errors.append(f"Image too small. Minimum dimension: {min_dimension}px")
    
    # Check if image is valid and can be processed
    try:
        image.verify()
    except
