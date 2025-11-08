# streamlit_app.py
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

# Load models - support both Keras and H5 formats
@st.cache_resource
def load_models():
    models = {}
    
    # Try loading Keras model
    try:
        models['keras'] = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.success("✅ Keras Model Loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Keras model failed: {e}")
        models['keras'] = None
    
    # Try loading H5 model
    try:
        models['h5'] = tf.keras.models.load_model("plantvillage_head_cpu_v2_1.h5")
        st.sidebar.success("✅ H5 Model Loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠️ H5 model failed: {e}")
        models['h5'] = None
    
    return models

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
models = load_models()
class_names = load_class_names()
img_size = (224, 224)  # Standard size for most H5 models

def predict_image(image, model_type='keras'):
    """Predict plant disease from image using specified model"""
    try:
        if models[model_type] is None:
            return None, None, f"{model_type.upper()} model not available"
            
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = models[model_type].predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

# Check OpenAI API key
def check_openai_key():
    try:
        # Test if OpenAI is configured
        test_advice = chatbot_helper.generate_advice("tomato", "healthy")
        return True
    except Exception as e:
        return False

# App UI
st.title("🌿 Plant Doctor - Smart Plant Diagnosis")
st.markdown("""
**Upload a leaf photo → ML model diagnoses disease → AI explains results & suggests treatments**

This smart app combines:
- 🤖 **Machine Learning** for accurate disease detection  
- 💬 **AI Language Model** for clear explanations and advice
""")

# Check if any model loaded successfully
available_models = [name for name, model in models.items() if model is not None]
if not available_models:
    st.error("❌ No models loaded. Please check your model files.")
    st.stop()

# Model selection
st.sidebar.header("🔧 Model Selection")
selected_model = st.sidebar.selectbox(
    "Choose ML Model:",
    available_models,
    format_func=lambda x: f"{x.upper()} Model"
)

# File uploader
uploaded_file = st.file_uploader(
    "📸 Choose a plant leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, well-lit photo of a plant leaf for best results"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        st.info("🔍 Image uploaded successfully!")
    
    # Check OpenAI before showing analyze button
    openai_ready = check_openai_key()
    
    if not openai_ready:
        st.warning("""
        🔑 **OpenAI API Key Required**
        
        To get AI-powered treatment advice, please:
        1. Go to your app settings (click 'Manage app' → 'Settings')
        2. Click 'Secrets' in the sidebar  
        3. Add your OpenAI API key:
        ```
        OPENAI_API_KEY = "your-api-key-here"
        ```
        4. Redeploy the app
        """)
    
    # Predict button (show even without OpenAI for ML-only functionality)
    if st.button("🧠 Analyze Plant Diagnosis", type="primary", use_container_width=True):
        with st.spinner("🤖 ML Model analyzing leaf for diseases..."):
            # Make prediction
            disease, confidence, error = predict_image(image, selected_model)
            
            if error:
                st.error(f"❌ Prediction error: {error}")
            else:
                with col2:
                    st.subheader("📊 ML Diagnosis Results")
                    
                    # Display confidence with color coding
                    if confidence > 0.8:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} 🎯")
                    elif confidence > 0.6:
                        st.warning(f"**Disease:** {disease}")
                        st.warning(f"**Confidence:** {confidence:.2%} ⚠️")
                    else:
                        st.info(f"**Disease:** {disease}")
                        st.info(f"**Confidence:** {confidence:.2%} 🔍")
                    
                    st.info(f"**Model Used:** {selected_model.upper()}")
                    
                    # Get plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0]
                        st.info(f"**Plant Type:** {plant_name.title()}")
                    else:
                        plant_name = "plant"
                
                # Get AI advice from LLM if available
                if openai_ready:
                    with st.spinner("💬 AI generating treatment advice..."):
                        advice = chatbot_helper.generate_advice(plant_name, disease)
                        
                    st.subheader("💡 AI Treatment Advice")
                    st.info(advice)
                else:
                    st.warning("""
                    💡 **Treatment Advice (Generic)**
                    
                    *Enable OpenAI API for personalized AI advice*
                    
                    **General tips for plant health:**
                    - Remove affected leaves to prevent spread
                    - Ensure proper watering (not too much/little)
                    - Provide adequate sunlight and air circulation
                    - Use organic fungicides if needed
                    - Monitor plant recovery regularly
                    """)
                
                # Additional tips
                st.subheader("🌱 General Plant Care Tips")
                st.markdown("""
                - **Water properly**: Avoid overwatering and ensure good drainage
                - **Provide sunlight**: Most plants need 4-6 hours of indirect sunlight  
                - **Check soil quality**: Use well-draining soil with proper nutrients
                - **Monitor regularly**: Check leaves weekly for early signs of issues
                - **Isolate affected plants**: Prevent spread to other plants
                """)

# Sidebar with app information
with st.sidebar:
    st.header("ℹ️ How It Works")
    st.markdown("""
    1. **📸 Capture** - Take a clear leaf photo
    2. **🤖 Analyze** - ML model detects diseases  
    3. **💬 Understand** - AI explains diagnosis
    4. **🌱 Treat** - Get personalized care advice
    """)
    
    st.header("🔬 Active Models")
    for model_name, model in models.items():
        status = "✅ Active" if model is not None else "❌ Not Available"
        st.write(f"- **{model_name.upper()}**: {status}")
    
    st.header("🌿 Supported Plants")
    st.markdown("""
    - Tomatoes 🍅
    - Potatoes 🥔
    - Peppers 🌶️
    - Apples 🍎
    - Corn 🌽
    - Grapes 🍇
    - And many more!
    """)

# Footer
st.markdown("---")
st.markdown("### 🚀 Smart Plant Diagnosis Powered by AI")
st.caption("Combining Machine Learning and Language Models for comprehensive plant care | Built with TensorFlow & Streamlit")
