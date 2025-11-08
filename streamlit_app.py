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

# MobileNetV4 ONLY - no fallbacks
@st.cache_resource
def load_mobilenetv4_model():
    try:
        # Try to understand the model structure first
        st.sidebar.info("🔄 Loading MobileNetV4 model...")
        
        # Load the model without compilation first to inspect
        model = tf.keras.models.load_model(
            "plantvillage_finetuned_mobilenetv4.h5",
            compile=False
        )
        
        st.sidebar.success("✅ MobileNetV4 Model Structure Loaded!")
        
        # Debug information
        st.sidebar.info(f"📊 Inputs: {len(model.inputs)}")
        st.sidebar.info(f"📊 Outputs: {len(model.outputs)}")
        
        # Check if we need to create a wrapper for the multi-output issue
        if len(model.outputs) > 1:
            st.sidebar.warning("🔄 Model has multiple outputs, creating wrapper...")
            # Try to find the classification output
            classification_output = None
            for i, output in enumerate(model.outputs):
                st.sidebar.info(f"Output {i} shape: {output.shape}")
                # Look for output that matches our class count (38 classes)
                if len(output.shape) == 2 and (output.shape[1] == 38 or output.shape[1] == len(load_class_names())):
                    classification_output = output
                    st.sidebar.success(f"🎯 Using output {i} for classification")
                    break
            
            if classification_output is None:
                # Use first output as default
                classification_output = model.outputs[0]
                st.sidebar.info("🔧 Using first output as default")
            
            # Create a new model with single output
            model = tf.keras.Model(inputs=model.inputs, outputs=classification_output)
        
        st.sidebar.success("🚀 MobileNetV4 Ready for Predictions!")
        return model
        
    except Exception as e:
        st.sidebar.error(f"❌ MobileNetV4 loading failed: {str(e)}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        return [
            "Apple_Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple_healthy",
            "Blueberry_healthy", "Cherry_healthy", "Cherry_Powdery_mildew", 
            "Corn_Common_rust", "Corn_Gray_leaf_spot", "Corn_Healthy", "Corn_Northern_Leaf_Blight",
            "Grape_Black_rot", "Grape_Esca", "Grape_Healthy", "Grape_Leaf_blight",
            "Orange_Haunglongbing", "Peach_Healthy", "Peach_Bacterial_spot",
            "Pepper_bell_Bacterial_spot", "Pepper_bell_Healthy",
            "Potato_Early_blight", "Potato_Healthy", "Potato_Late_blight",
            "Raspberry_Healthy", "Soybean_Healthy", "Squash_Powdery_mildew",
            "Strawberry_Healthy", "Strawberry_Leaf_scorch",
            "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Healthy",
            "Tomato_Late_blight", "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot",
            "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
            "Tomato_Mosaic_virus"
        ]

# Load ONLY MobileNetV4
model = load_mobilenetv4_model()
class_names = load_class_names()
img_size = (224, 224)

def predict_with_mobilenetv4(image):
    """Predict using MobileNetV4"""
    try:
        # Preprocess image
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        
        # Ensure 3 channels
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction
        prediction = model.predict(img_array, verbose=0)
        
        # Get results
        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        
        return predicted_class, confidence, None
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

def generate_advice(plant, disease):
    """Generate plant care advice"""
    return f"""
    **💡 Treatment for {disease.replace('_', ' ').title()}:**
    - Remove affected leaves if present
    - Improve air circulation around plants
    - Water at the base to keep leaves dry
    - Monitor plant recovery regularly
    - Consult local garden experts if condition worsens
    """

# App UI
st.title("🌿 Plant Doctor - MobileNetV4 Test")
st.markdown("Testing MobileNetV4 model for plant disease detection")

# Check if model loaded
if model is None:
    st.error("""
    ❌ **MobileNetV4 Model Failed to Load**
    
    **Debug Information:**
    - Model file: `plantvillage_finetuned_mobilenetv4.h5`
    - Issue: Multi-output architecture compatibility
    - Status: Requires model architecture adjustment
    
    **Next Steps:**
    1. Check if model file exists in repository
    2. Verify model file integrity
    3. Consider re-saving model with single output
    4. Check TensorFlow version compatibility
    """)
    
    # Show file structure for debugging
    st.info("📁 Current directory files:")
    try:
        for file in os.listdir("."):
            if "plant" in file.lower() or ".h5" in file or ".keras" in file:
                st.write(f"- {file}")
    except:
        st.write("Cannot list directory contents")
    
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a plant leaf image", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)
        st.info(f"MobileNetV4 input size: {img_size}")
    
    if st.button("Test MobileNetV4 Prediction", type="primary", use_container_width=True):
        with st.spinner("MobileNetV4 processing..."):
            disease, confidence, error = predict_with_mobilenetv4(image)
            
            if error:
                st.error(f"❌ Prediction failed: {error}")
            else:
                with col2:
                    st.subheader("MobileNetV4 Results")
                    st.success(f"**Diagnosis:** {disease.replace('_', ' ').title()}")
                    st.success(f"**Confidence:** {confidence:.1%}")
                
                advice = generate_advice("plant", disease)
                st.subheader("Care Advice")
                st.info(advice)

# Debug sidebar
with st.sidebar:
    st.header("MobileNetV4 Debug Info")
    if model:
        st.success("✅ Model Loaded")
        st.info(f"Input shape: {model.input_shape}")
        st.info(f"Output shape: {model.output_shape}")
    else:
        st.error("❌ Model Failed")
    
    st.header("Test Image Requirements")
    st.markdown("""
    - Size: 224×224 pixels
    - Format: JPG, PNG, JPEG
    - Color: RGB preferred
    - Focus: Clear leaf image
    """)

st.markdown("---")
st.caption("MobileNetV4 Model Test | Plant Disease Detection")
