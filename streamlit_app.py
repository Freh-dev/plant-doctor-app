# streamlit_app.py - ENHANCED ATTRACTIVE VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# Set page config with attractive theme
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #3CB371;
        text-align: center;
        margin-bottom: 2rem;
    }
    .diagnosis-card {
        background-color: #f0fff0;
        border-radius: 10px;
        padding: 1.5rem;
        border-left: 5px solid #3CB371;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #3CB371;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background-color: #f8fff8;
        margin: 1rem 0;
    }
    .status-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .plant-icon {
        font-size: 2rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .stButton button {
        background-color: #3CB371;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        width: 100%;
    }
    .stButton button:hover {
        background-color: #2E8B57;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Debug function
def check_openai_setup():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Test if it works
        try:
            test_advice = chatbot_helper.generate_advice("tomato", "healthy")
            if "OpenAI" not in test_advice and "API key" not in test_advice:
                return True
            else:
                return False
        except:
            return False
    else:
        return False

@st.cache_resource
def load_model():
    try:
        # Try the improved model first, then fallback to ultra light
        try:
            model = tf.keras.models.load_model("plantvillage_mobilenetv2_fixed.h5")
        except:
            model = tf.keras.models.load_model("ultra_light_model.keras")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model Error: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        return ["Tomato_healthy", "Tomato_early_blight", "Tomato_late_blight"]

# Load resources
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()
img_size = (128, 128)

def predict_image(image):
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

# App UI - Header Section
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Upload a plant leaf photo for instant diagnosis and care advice</p>', unsafe_allow_html=True)

# Main content area
if model is None:
    st.error("""
    ## Service Temporarily Unavailable
    We're experiencing technical difficulties with our plant diagnosis model. 
    Please try again later or contact support if the problem persists.
    """)
    st.stop()

# Upload section with attractive design
st.markdown("## 📸 Upload Plant Image")
st.markdown('<div class="upload-area">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop your plant leaf image here", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)
st.caption("Supported formats: JPG, JPEG, PNG • Maximum file size: 200MB")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📷 Your Plant Leaf")
        st.image(image, use_container_width=True)
        
        # Add image info
        st.info(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels")
    
    # Analysis button
    if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
        with st.spinner("🔬 Analyzing your plant... This may take a few seconds."):
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error("""
                ## Analysis Failed
                We couldn't process your image. Please try again with a clearer photo of a plant leaf.
                """)
            else:
                with col2:
                    st.markdown("### 📋 Diagnosis Results")
                    
                    # Determine status and color
                    if "healthy" in disease.lower():
                        status_emoji = "✅"
                        status_color = "green"
                    else:
                        status_emoji = "⚠️"
                        status_color = "orange"
                    
                    # Display diagnosis in a card
                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <div class="plant-icon">{status_emoji}</div>
                        <h3 style="color: {status_color}; text-align: center;">{disease.replace('_', ' ').title()}</h3>
                        <p style="text-align: center; font-size: 1.2rem;">Confidence: <strong>{confidence:.1%}</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence indicator
                    if confidence < 0.5:
                        st.warning("**Note:** Confidence level is low. Consider taking a clearer photo for more accurate diagnosis.")
                    elif confidence > 0.9:
                        st.success("**Note:** High confidence diagnosis.")
                
                # Extract plant name for advice
                if '_' in disease:
                    plant_name = disease.split('_')[0]
                else:
                    plant_name = "plant"
                
                # Care instructions section
                st.markdown("---")
                st.markdown("## 💡 Care Instructions")
                
                if openai_ready:
                    with st.spinner("🤖 Generating personalized care advice..."):
                        advice = chatbot_helper.generate_advice(plant_name, disease)
                    
                    # Check if we got real AI advice or fallback
                    if "OpenAI" in advice or "API key" in advice:
                        st.warning("⚠️ Using standard care advice")
                        st.info(f"""
                        **Standard treatment for {disease.replace('_', ' ').title()}:**
                        
                        - Remove affected leaves immediately to prevent spread
                        - Improve air circulation around plants
                        - Water at the base, avoid wetting leaves
                        - Apply appropriate fungicide if needed
                        - Monitor plant recovery daily
                        - Consult local garden center if condition worsens
                        """)
                    else:
                        st.success("✅ AI-Generated Personalized Advice")
                        st.info(advice)
                else:
                    st.warning("⚠️ Using standard care advice")
                    st.info(f"""
                    **Standard treatment for {disease.replace('_', ' ').title()}:**
                    
                    - Remove affected leaves immediately to prevent spread
                    - Improve air circulation around plants
                    - Water at the base, avoid wetting leaves
                    - Apply appropriate fungicide if needed
                    - Monitor plant recovery daily
                    - Consult local garden center if condition worsens
                    
                    *Enable AI advice for personalized recommendations.*
                    """)

# Enhanced sidebar
with st.sidebar:
    st.markdown("## 🌟 Plant Doctor")
    st.markdown("Your AI-powered plant health assistant")
    
    st.markdown("---")
    st.markdown("### 🔧 System Status")
    
    # Status cards
    col1, col2 = st.columns(2)
    with col1:
        if model:
            st.markdown('<div class="status-card">'
                       '<h4>✅ Model Active</h4>'
                       '<p>Plant diagnosis ready</p>'
                       '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card">'
                       '<h4>❌ Model Inactive</h4>'
                       '<p>Diagnosis unavailable</p>'
                       '</div>', unsafe_allow_html=True)
    
    with col2:
        if openai_ready:
            st.markdown('<div class="status-card">'
                       '<h4>✅ AI Advice Ready</h4>'
                       '<p>Personalized care tips</p>'
                       '</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-card">'
                       '<h4>⚠️ Basic Advice</h4>'
                       '<p>Standard care only</p>'
                       '</div>', unsafe_allow_html=True)
    
    st.metric("Plant Types", len(class_names))
    
    # Setup instructions
    if not openai_ready:
        st.markdown("---")
        st.markdown("### 🚀 Enable AI Features")
        st.info("""
        **To get personalized AI care advice:**
        
        1. Get an OpenAI API key
        2. In Streamlit app settings:
           - Click 'Secrets'
           - Add: `OPENAI_API_KEY = "your-key-here"`
        3. Redeploy the app
        
        Your plants will thank you! 🌱
        """)
    
    # Tips section
    st.markdown("---")
    st.markdown("### 💡 Tips for Best Results")
    st.info("""
    - Use clear, well-lit photos
    - Focus on the affected leaves
    - Include a plain background
    - Take multiple angles if unsure
    - Regular monitoring is key
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666;">
        <p>AI-powered plant health analysis • Keep your plants thriving 🌱</p>
    </div>
    """, 
    unsafe_allow_html=True
)
