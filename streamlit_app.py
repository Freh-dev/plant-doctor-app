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

# Load the MobileNetV4 model with proper handling
@st.cache_resource
def load_model():
    try:
        # Load the model with custom objects if needed
        model = tf.keras.models.load_model(
            "plantvillage_finetuned_mobilenetv4.h5",
            custom_objects=None,
            compile=False
        )
        
        st.sidebar.success("✅ MobileNetV4 Model Loaded Successfully!")
        
        # Debug: Show model architecture
        st.sidebar.info(f"📊 Model Type: {type(model)}")
        if hasattr(model, 'outputs'):
            st.sidebar.info(f"🔧 Outputs: {len(model.outputs)}")
        
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading MobileNetV4: {e}")
        
        # Fallback to ultra light model
        try:
            st.sidebar.info("🔄 Trying ultra light model as fallback...")
            model = tf.keras.models.load_model("ultra_light_model.keras")
            st.sidebar.success("✅ Ultra Light Model Loaded (Fallback)")
            return model
        except Exception as fallback_error:
            st.sidebar.error(f"❌ Fallback also failed: {fallback_error}")
            return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        # Fallback class names for PlantVillage
        return [
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight", 
            "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites", 
            "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_mosaic_virus", 
            "Tomato_healthy", "Potato_Early_blight", "Potato_Late_blight", "Potato_healthy",
            "Corn_Northern_Leaf_Blight", "Corn_Common_rust", "Corn_healthy",
            "Pepper_bell_Bacterial_spot", "Pepper_bell_healthy"
        ]

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (224, 224)  # Standard for MobileNet models

def predict_image(image):
    """Predict plant disease from image with MobileNetV4 compatibility"""
    try:
        # Resize image
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Handle different model output types
        prediction = model.predict(img_array, verbose=0)
        
        # Debug: Check prediction structure
        st.sidebar.info(f"🎯 Prediction type: {type(prediction)}")
        if isinstance(prediction, list):
            st.sidebar.info(f"📦 Prediction list length: {len(prediction)}")
            # Use the first output if multiple outputs
            prediction = prediction[0]
        
        # Get the predicted class and confidence
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
        
    except Exception as e:
        return None, None, str(e)

# Enhanced advice function
def generate_advice(plant, disease):
    """Generate plant care advice"""
    advice_templates = {
        "bacterial_spot": f"🦠 **{plant} Bacterial Spot**: Remove infected leaves, apply copper-based spray, avoid overhead watering, improve air circulation.",
        "early_blight": f"🍂 **{plant} Early Blight**: Remove affected leaves, apply fungicide, water at soil level, ensure good spacing.",
        "late_blight": f"🔥 **{plant} Late Blight**: Remove infected plants immediately, use copper fungicide, avoid wet foliage, destroy infected material.",
        "leaf_mold": f"🍄 **{plant} Leaf Mold**: Increase ventilation, reduce humidity, apply fungicide, space plants properly.",
        "septoria": f"🔴 **{plant} Septoria Leaf Spot**: Remove infected leaves, apply fungicide, avoid overhead irrigation, rotate crops.",
        "yellow_curl": f"🔄 **{plant} Yellow Leaf Curl**: Remove infected plants, control whiteflies, use resistant varieties.",
        "mosaic": f"🟨 **{plant} Mosaic Virus**: Remove infected plants, control aphids, disinfect tools.",
        "healthy": f"🌱 **{plant} Healthy**: Excellent! Continue regular care: proper watering, balanced nutrition, and pest monitoring."
    }
    
    # Find matching advice
    disease_lower = disease.lower()
    for key in advice_templates:
        if key in disease_lower:
            return advice_templates[key]
    
    # General advice
    return f"🌿 **{disease} in {plant}**: Remove affected leaves, improve growing conditions, monitor regularly, and consult local experts if needed."

# App UI
st.title("🌿 Plant Doctor - MobileNetV4 Edition")
st.markdown("**Powered by fine-tuned MobileNetV4 for advanced plant disease detection**")

# Check if model loaded successfully
if model is None:
    st.error("""
    ❌ Model not loaded. Please ensure:
    - `plantvillage_finetuned_mobilenetv4.h5` is in your repository
    - File is not corrupted
    - Model architecture is compatible
    """)
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image...", 
    type=["jpg", "jpeg", "png"],
    help="Upload a clear photo of a plant leaf (224x224 pixels works best)"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf", width='stretch')
        st.info(f"📏 Original size: {image.size}")
        st.info(f"🎯 Model input: {img_size}")
    
    # Predict button
    if st.button("🔍 Analyze with MobileNetV4", type="primary", width='stretch'):
        with st.spinner("MobileNetV4 analyzing..."):
            # Make prediction
            disease, confidence, error = predict_image(image)
            
            if error:
                st.error(f"❌ Prediction error: {error}")
                st.info("💡 Try using the ultra light model instead")
            else:
                with col2:
                    st.subheader("📊 Diagnosis Results")
                    
                    # Confidence-based styling
                    if confidence > 0.85:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} 🎯 Very High")
                    elif confidence > 0.70:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} ✅ High")
                    elif confidence > 0.50:
                        st.warning(f"**Disease:** {disease}")
                        st.warning(f"**Confidence:** {confidence:.2%} ⚠️ Medium")
                    else:
                        st.info(f"**Disease:** {disease}")
                        st.info(f"**Confidence:** {confidence:.2%} 🔍 Low")
                    
                    # Extract plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0].title()
                        st.info(f"**Plant Type:** {plant_name}")
                    else:
                        plant_name = "Plant"
                
                # Get advice
                advice = generate_advice(plant_name, disease)
                    
                st.subheader("💡 AI Treatment Advice")
                st.info(advice)

# Sidebar
with st.sidebar:
    st.header("🔬 Model Info")
    st.metric("Architecture", "MobileNetV4")
    st.metric("Input Size", "224×224")
    st.metric("Fine-tuned", "PlantVillage")
    
    st.header("🌿 Supported Plants")
    st.markdown("""
    - **Tomatoes** (10 diseases)
    - **Potatoes** (3 diseases)
    - **Corn** (3 diseases)
    - **Peppers** (2 conditions)
    - **+ More PlantVillage species**
    """)
    
    st.header("⚡ Performance")
    st.markdown("""
    - **High accuracy** detection
    - **Fast inference** with MobileNetV4
    - **38 disease classes**
    - **Professional-grade** results
    """)

# Footer
st.markdown("---")
st.caption("Powered by Fine-tuned MobileNetV4 | PlantVillage Dataset | Built with TensorFlow & Streamlit")
