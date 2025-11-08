# streamlit_app.py - UPDATED VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# Set page config
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide"
)

# Debug function
def check_openai_setup():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        st.sidebar.success("✅ OpenAI API Key Found")
        # Test if it works
        try:
            test_advice = chatbot_helper.generate_advice("tomato", "healthy")
            if "OpenAI" not in test_advice and "API key" not in test_advice:
                st.sidebar.success("✅ OpenAI Connection Working")
                return True
            else:
                st.sidebar.warning("⚠️ OpenAI Key Invalid")
                return False
        except:
            st.sidebar.error("❌ OpenAI Test Failed")
            return False
    else:
        st.sidebar.error("❌ OpenAI API Key Missing")
        return False

@st.cache_resource
def load_model():
    try:
        # Try the improved model first, then fallback to ultra light
        try:
            model = tf.keras.models.load_model("plantvillage_mobilenetv2_fixed.h5")
            st.sidebar.success("✅ Advanced Model Loaded")
        except:
            model = tf.keras.models.load_model("ultra_light_model.keras")
            st.sidebar.success("✅ Standard Model Loaded")
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

# App UI
st.title("🌿 Plant Doctor")
st.markdown("Upload a plant leaf photo for instant diagnosis")

if model is None:
    st.error("Service temporarily unavailable. Please check the model files.")
    st.stop()

uploaded_file = st.file_uploader("Choose a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Your plant leaf", use_container_width=True)
        st.info(f"Model input size: {img_size}")
    
    if st.button("Analyze Plant", type="primary", use_container_width=True):
        with st.spinner("Analyzing..."):
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error("Analysis failed. Please try another image.")
            else:
                with col2:
                    st.subheader("Diagnosis Results")
                    st.success(f"**Condition:** {disease}")
                    st.success(f"**Confidence:** {confidence:.1%}")
                    
                    # Show model debug info
                    if confidence < 0.5:
                        st.warning("⚠️ Low confidence - model may be uncertain")
                    
                    # Extract plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0]
                    else:
                        plant_name = "plant"
                
                # Get advice - show what type we're getting
                st.subheader("💡 Care Instructions")
                
                if openai_ready:
                    with st.spinner("Getting AI advice..."):
                        advice = chatbot_helper.generate_advice(plant_name, disease)
                    
                    # Check if we got real AI advice or fallback
                    if "OpenAI" in advice or "API key" in advice:
                        st.warning("⚠️ Using fallback advice (OpenAI not working)")
                        st.info(advice)
                    else:
                        st.success("✅ AI-Generated Advice")
                        st.info(advice)
                else:
                    st.warning("⚠️ Using basic advice (OpenAI not configured)")
                    basic_advice = f"""
                    **Basic treatment for {disease}:**
                    - Remove affected leaves immediately
                    - Improve air circulation around plants
                    - Water at the base, avoid wetting leaves
                    - Monitor plant recovery daily
                    - Consult local garden center if condition worsens
                    """
                    st.info(basic_advice)

# Debug sidebar
with st.sidebar:
    st.header("🔧 System Status")
    st.metric("Model Status", "✅ Active" if model else "❌ Inactive")
    st.metric("OpenAI Status", "✅ Ready" if openai_ready else "❌ Not Ready")
    st.metric("Class Count", len(class_names))
    
    st.header("🚨 Setup Required")
    if not openai_ready:
        st.error("""
        **To enable AI advice:**
        1. Go to Streamlit app settings
        2. Click 'Secrets'
        3. Add: `OPENAI_API_KEY = "your-key-here"`
        4. Redeploy app
        """)

st.markdown("---")
st.caption("AI-powered plant health analysis")
