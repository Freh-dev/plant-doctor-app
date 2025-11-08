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

# Load model and class names - USING THE WORKING ULTRA LIGHT MODEL
@st.cache_resource
def load_model():
    try:
        # First try the ultra light model (it's working)
        model = tf.keras.models.load_model("ultra_light_model.keras")
        st.sidebar.success("✅ Ultra Light Model Loaded Successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading ultra light model: {e}")
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
st.title("🌿 Plant Doctor - Smart Plant Diagnosis")
st.markdown("""
**Upload a leaf photo → ML model diagnoses disease → LLM explains results & suggests treatments**

This smart app combines:
- 🤖 **Machine Learning** for accurate disease detection
- 💬 **AI Language Model** for clear explanations and advice
""")

# Check if model loaded successfully
if model is None or not class_names:
    st.warning("⚠️ Model not loaded. Please check your files.")
    st.stop()

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
    
    # Predict button
    if st.button("🧠 Analyze Plant Diagnosis", type="primary", use_container_width=True):
        with st.spinner("🤖 ML Model analyzing leaf for diseases..."):
            # Make prediction
            disease, confidence, error = predict_image(image)
            
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
                    
                    # Get plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0]
                        st.info(f"**Plant Type:** {plant_name.title()}")
                    else:
                        plant_name = "plant"
                
                # Get AI advice from LLM
                with st.spinner("💬 AI generating treatment advice..."):
                    advice = chatbot_helper.generate_advice(plant_name, disease)
                    
                st.subheader("💡 AI Treatment Advice")
                st.info(advice)
                
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
    
    st.header("🔬 Technology Stack")
    st.markdown("""
    - **TensorFlow** - Machine Learning
    - **Computer Vision** - Image analysis
    - **OpenAI GPT** - Natural language explanations
    - **Streamlit** - Web interface
    """)
    
    st.header("📊 Model Info")
    st.metric("Current Model", "Ultra Light")
    st.metric("Model Size", "110 KB")
    st.metric("Status", "✅ Active")
    st.metric("Supported Plants", "30+")

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
