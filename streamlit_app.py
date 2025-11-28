# streamlit_app.py - FIXED VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper
from io import BytesIO
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------- CONFIGURATION ----------------------- #
class AppConfig:
    MODEL_PATH = "plant_disease_final_model.keras"
    CLASS_NAMES_PATH = "class_names_final.json"
    MAX_FILE_SIZE_MB = 200
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    DEFAULT_IMG_SIZE = (224, 224)

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- SESSION STATE --------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "uploaded_image_data" not in st.session_state:
    st.session_state.uploaded_image_data = None

if "uploaded_image_name" not in st.session_state:
    st.session_state.uploaded_image_name = None

# ----------------------- CORE FUNCTIONS -------------------- #
@st.cache_resource
def load_model():
    """Load the trained Keras model"""
    if not os.path.exists(AppConfig.MODEL_PATH):
        st.error(f"❌ Model file not found: {AppConfig.MODEL_PATH}")
        return None
    try:
        model = tf.keras.models.load_model(AppConfig.MODEL_PATH)
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from json file"""
    try:
        with open(AppConfig.CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        st.sidebar.info(f"✅ Loaded {len(class_names)} classes")
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return ["Tomato_healthy", "Tomato_early_blight", "Tomato_late_blight"]

def preprocess_image_for_model(image, img_size):
    """Preprocess image for EfficientNet model"""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize and preprocess
        img = image.resize(img_size)
        img_array = np.array(img).astype("float32")
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {e}")

def predict_image(image, model, class_names, img_size):
    """Run prediction on image"""
    try:
        img_batch = preprocess_image_for_model(image, img_size)
        prediction = model.predict(img_batch, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        
        if predicted_index >= len(class_names):
            return None, None, f"Prediction index out of range"
            
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

# ----------------------- INITIALIZATION -------------------- #
model = load_model()
class_names = load_class_names()

# Determine image size
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = AppConfig.DEFAULT_IMG_SIZE

# ----------------------- MAIN APP -------------------------- #
st.title("🌿 Plant Doctor")
st.write("Upload a plant leaf image for diagnosis")

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=AppConfig.SUPPORTED_FORMATS,
    help="Supported formats: JPG, JPEG, PNG"
)

# Store file data in session state immediately after upload
if uploaded_file is not None:
    # Store the actual file data to prevent None issues
    st.session_state.uploaded_image_data = uploaded_file.getvalue()
    st.session_state.uploaded_image_name = uploaded_file.name

# Display and process the uploaded image
if st.session_state.uploaded_image_data is not None:
    try:
        # Create a file-like object from stored data
        image_data = BytesIO(st.session_state.uploaded_image_data)
        
        # Open and display the image
        image = Image.open(image_data)
        
        st.success("✅ File uploaded successfully!")
        st.write(f"**Filename:** {st.session_state.uploaded_image_name}")
        
        # Display image preview
        st.image(image, caption="📷 Your Plant Leaf", width=400)
        
        # File info
        file_size_kb = len(st.session_state.uploaded_image_data) / 1024
        st.write(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size_kb:.1f} KB")
        
        # Analyze button
        if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
            if model is None:
                st.error("❌ Model not loaded. Please check if the model file exists.")
            elif not class_names:
                st.error("❌ Class names not loaded. Please check the class names file.")
            else:
                with st.spinner("🔬 Analyzing your plant..."):
                    # Recreate the image object for prediction (important!)
                    image_for_prediction = Image.open(BytesIO(st.session_state.uploaded_image_data))
                    disease, confidence, error = predict_image(image_for_prediction, model, class_names, img_size)
                
                if error:
                    st.error(f"❌ Analysis failed: {error}")
                else:
                    # Success! Display results
                    st.session_state.prediction_history.append(disease)
                    
                    # Format disease name
                    formatted_disease = disease.replace("_", " ").title()
                    
                    # Display diagnosis card
                    st.subheader("📋 Diagnosis Results")
                    
                    if "healthy" in disease.lower():
                        status_emoji = "✅"
                        status_text = "Healthy Plant"
                        status_color = "#2E8B57"
                    else:
                        status_emoji = "⚠️"
                        status_text = "Needs Attention"
                        status_color = "#FFA500"
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #ffffff, #f8fff8);
                                border-radius: 12px; padding: 1.5rem; margin: 1.2rem 0;
                                box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
                                border: 1px solid #e0f0e0; text-align: center;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                        <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem;
                                     border-radius: 15px; font-weight: 600; margin-bottom: 1rem; display: inline-block;">
                            {status_text}
                        </span>
                        <h3 style="color: {status_color}; margin-bottom: 0.8rem;">
                            {formatted_disease}
                        </h3>
                        <div>
                            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                            <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                {confidence:.1%}
                            </h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence warnings
                    if confidence < 0.4:
                        st.warning("⚠️ **Low Confidence Warning**: The model is not very confident about this diagnosis. Try a clearer, well-lit image.")
                    elif confidence < 0.75:
                        st.info("ℹ️ **Moderate Confidence**: Consider getting a second opinion or uploading additional images.")
                    else:
                        st.success("✅ **High Confidence**: Diagnosis is likely reliable.")
                    
                    # Care instructions
                    st.subheader("💡 Care Instructions")
                    try:
                        plant_name = disease.split("_")[0] if "_" in disease else "plant"
                        advice = chatbot_helper.generate_advice(plant_name, disease)
                        st.info(advice)
                    except Exception as e:
                        st.info(f"""
                        **🌱 Recommended Care for {formatted_disease}**
                        
                        - Remove affected leaves to prevent spread
                        - Ensure proper watering at the base
                        - Improve air circulation
                        - Monitor plant recovery daily
                        - Consult a plant expert for specific treatment
                        """)
        
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
        st.info("""
        **Troubleshooting tips:**
        1. Try uploading the image again
        2. Ensure the file is not corrupted
        3. Try a different image format
        4. Check if the image file is too large
        """)

else:
    # No file uploaded state
    st.markdown("""
    <div style="border: 3px dashed #3CB371; border-radius: 15px; padding: 3rem 1.5rem;
                text-align: center; background: #F0FFF0; margin: 1.5rem 0;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
        <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
        <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
        <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
    </div>
    """, unsafe_allow_html=True)

# Sidebar information
with st.sidebar:
    st.header("System Status")
    st.write(f"✅ Model: {'Loaded' if model else 'Not Loaded'}")
    st.write(f"✅ Classes: {len(class_names)}")
    st.write(f"✅ Image Size: {img_size}")
    
    if st.session_state.prediction_history:
        st.subheader("Recent Predictions")
        for pred in st.session_state.prediction_history[-3:]:
            st.write(f"- {pred.replace('_', ' ').title()}")
    
    # Debug button
    if st.button("Clear Uploaded Image"):
        st.session_state.uploaded_image_data = None
        st.session_state.uploaded_image_name = None
        st.success("Image cleared! Upload a new image.")
        st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
