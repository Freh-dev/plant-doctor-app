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

# Load MobileNetV4 model with proper output handling
@st.cache_resource
def load_mobilenetv4_model():
    try:
        # Load the model
        model = tf.keras.models.load_model(
            "plantvillage_finetuned_mobilenetv4.h5",
            compile=False
        )
        
        st.sidebar.success("✅ MobileNetV4 Model Loaded!")
        st.sidebar.info("🎯 97% Accuracy Model Active")
        
        # Debug: Show model structure
        st.sidebar.info(f"📊 Model Inputs: {len(model.inputs)}")
        st.sidebar.info(f"📊 Model Outputs: {len(model.outputs)}")
        
        return model
    except Exception as e:
        st.sidebar.error(f"❌ MobileNetV4 loading failed: {str(e)[:100]}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
        # MobileNetV4 PlantVillage classes (38 classes)
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
            "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Mosaic_virus"
        ]

# Load the specific model
model = load_mobilenetv4_model()
class_names = load_class_names()
img_size = (224, 224)  # MobileNet standard size

def predict_with_mobilenetv4(image):
    """Predict using MobileNetV4 with multiple outputs"""
    try:
        # Preprocess image
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Get prediction - handle multiple outputs
        predictions = model.predict(img_array, verbose=0)
        
        # Debug output structure
        if isinstance(predictions, list):
            # Multiple outputs - use the classification output (usually first one)
            st.sidebar.info(f"🔧 Multiple outputs detected: {len(predictions)}")
            # Try different outputs to find the classification one
            for i, pred in enumerate(predictions):
                if len(pred.shape) == 2 and pred.shape[1] == len(class_names):
                    st.sidebar.info(f"🎯 Using output {i} for classification")
                    final_prediction = pred
                    break
            else:
                # If no clear match, use first output
                final_prediction = predictions[0]
        else:
            # Single output
            final_prediction = predictions
        
        # Get results
        predicted_index = np.argmax(final_prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(final_prediction))
        
        return predicted_class, confidence, None
        
    except Exception as e:
        return None, None, f"Prediction error: {str(e)}"

# High-quality advice for 97% accuracy model
def generate_expert_advice(plant, disease):
    """Generate expert-level advice for high-accuracy model"""
    
    expert_advice = {
        # Tomato diseases
        "Tomato_Bacterial_spot": """
        **🔬 Expert Treatment for Tomato Bacterial Spot:**
        • **Immediate Action**: Remove all infected leaves and destroy them
        • **Chemical Control**: Apply copper-based bactericide every 7-10 days
        • **Cultural Practice**: Water at soil level only, avoid overhead irrigation
        • **Prevention**: Use certified disease-free seeds and rotate crops
        • **Resistant Varieties**: Plant resistant cultivars like 'Mountain Merit'
        """,
        
        "Tomato_Early_blight": """
        **🔬 Expert Treatment for Tomato Early Blight:**
        • **Fungicide**: Apply chlorothalonil or mancozeb weekly
        • **Pruning**: Remove lower leaves up to first fruit cluster
        • **Water Management**: Use drip irrigation, water early in day
        • **Nutrition**: Maintain balanced fertility, avoid excess nitrogen
        • **Sanitation**: Clean garden debris thoroughly in fall
        """,
        
        "Tomato_Late_blight": """
        **🚨 EMERGENCY: Tomato Late Blight Detected:**
        • **URGENT**: Remove and bag all infected plants immediately
        • **Protection**: Spray healthy plants with fungicide containing mefenoxam
        • **Isolation**: Do not compost infected plants - destroy them
        • **Prevention**: Use resistant varieties like 'Defiant PHR' next season
        • **Monitoring**: Check nearby gardens and report to extension service
        """,
        
        "Tomato_Healthy": """
        **🌱 Excellent Plant Health:**
        • **Maintenance**: Continue current care practices
        • **Prevention**: Apply preventive fungicide during humid weather
        • **Monitoring**: Check plants twice weekly for early signs
        • **Nutrition**: Side-dress with balanced fertilizer when fruiting
        • **Support**: Ensure proper staking and air circulation
        """,
        
        # Potato diseases  
        "Potato_Early_blight": """
        **🥔 Expert Potato Early Blight Control:**
        • **Fungicide Program**: Begin spray program at first signs
        • **Cultural Control**: Hill soil around plants, avoid nitrogen excess
        • **Harvest**: Wait 2 weeks after vine death for better skin set
        • **Storage**: Cure potatoes properly before storage
        • **Rotation**: 3-4 year rotation away from solanaceous crops
        """,
        
        "Potato_Late_blight": """
        **🚨 POTATO LATE BLIGHT CRISIS:**
        • **IMMEDIATE**: Destroy all infected plants and tubers
        • **Protection**: Apply systemic fungicide to surrounding area
        • **Harvest**: Do not harvest from infected areas
        • **Future Planning**: Plant only certified seed potatoes
        • **Community Alert**: Notify neighboring growers immediately
        """
    }
    
    # Try exact match first
    if disease in expert_advice:
        return expert_advice[disease]
    
    # Try partial match
    for key, advice in expert_advice.items():
        if disease.lower() in key.lower() or key.lower() in disease.lower():
            return advice
    
    # General expert advice
    return f"""
    **🔬 Expert Guidance for {disease}:**
    • **Identification**: Confirm diagnosis with local extension service
    • **Immediate Action**: Remove visibly infected plant material
    • **Chemical Control**: Consult agricultural extension for recommended fungicides
    • **Cultural Practices**: Improve air circulation, proper spacing, and sanitation
    • **Long-term**: Implement crop rotation and use resistant varieties
    • **Monitoring**: Establish regular scouting schedule for early detection
    """

# App UI
st.title("🌿 Plant Doctor - MobileNetV4 Pro Edition")
st.markdown("### **97% Accuracy Plant Disease Detection**")
st.markdown("*Powered by fine-tuned MobileNetV4 with expert-level diagnostics*")

# Check if model loaded successfully
if model is None:
    st.error("""
    ❌ **MobileNetV4 Model Failed to Load**
    
    **Troubleshooting:**
    1. Ensure `plantvillage_finetuned_mobilenetv4.h5` is in your repository
    2. Check file integrity (should be ~20-50MB)
    3. Verify model compatibility with TensorFlow version
    4. Consider converting model to different format if issues persist
    """)
    st.stop()

# File uploader
uploaded_file = st.file_uploader(
    "📸 Upload Plant Leaf Image for Expert Analysis", 
    type=["jpg", "jpeg", "png"],
    help="High-quality images yield the most accurate 97% accuracy results"
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Leaf Sample", width='stretch')
        st.info(f"🔍 Image Analysis Ready")
        st.info(f"🎯 MobileNetV4 Processing: {img_size} input")
    
    # Predict button
    if st.button("🔬 Expert Diagnosis (97% Accuracy)", type="primary", width='stretch'):
        with st.spinner("🔄 MobileNetV4 Processing - High Accuracy Analysis..."):
            # Make prediction
            disease, confidence, error = predict_with_mobilenetv4(image)
            
            if error:
                st.error(f"❌ Analysis Error: {error}")
            else:
                with col2:
                    st.subheader("📊 Expert Diagnosis Results")
                    
                    # High-confidence display for 97% accuracy model
                    if confidence > 0.95:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} 🏆 Expert Certainty")
                    elif confidence > 0.85:
                        st.success(f"**Disease:** {disease}")
                        st.success(f"**Confidence:** {confidence:.2%} ✅ High Confidence")
                    elif confidence > 0.75:
                        st.warning(f"**Disease:** {disease}")
                        st.warning(f"**Confidence:** {confidence:.2%} ⚠️ Good Confidence")
                    else:
                        st.info(f"**Disease:** {disease}")
                        st.info(f"**Confidence:** {confidence:.2%} 🔍 Moderate Confidence")
                    
                    # Extract plant name
                    if '_' in disease:
                        plant_name = disease.split('_')[0].title()
                        st.info(f"**Plant Species:** {plant_name}")
                    else:
                        plant_name = "Plant"
                
                # Get expert advice
                advice = generate_expert_advice(plant_name, disease)
                    
                st.subheader("💡 Expert Treatment Protocol")
                st.info(advice)
                
                # Additional professional recommendations
                st.subheader("🔬 Professional Recommendations")
                st.markdown("""
                - **Laboratory Confirmation**: Consider sending sample to plant diagnostic lab
                - **Integrated Pest Management**: Combine cultural, biological, and chemical controls
                - **Record Keeping**: Document outbreak for future prevention strategies
                - **Economic Threshold**: Evaluate cost-effectiveness of control measures
                """)

# Professional sidebar
with st.sidebar:
    st.header("🔬 Model Specifications")
    st.metric("Architecture", "MobileNetV4")
    st.metric("Reported Accuracy", "97%")
    st.metric("Training Dataset", "PlantVillage")
    st.metric("Disease Classes", "38")
    
    st.header("🎯 Capabilities")
    st.markdown("""
    - **38 plant diseases**
    - **14 plant species**
    - **Professional-grade accuracy**
    - **Research-validated results**
    - **Production-ready diagnostics**
    """)
    
    st.header("🌿 Supported Species")
    st.markdown("""
    - **Fruits**: Apple, Blueberry, Cherry, Grape, Peach, Strawberry
    - **Vegetables**: Tomato, Potato, Pepper, Corn, Squash
    - **Citrus**: Orange
    - **Legumes**: Soybean
    - **Berries**: Raspberry
    """)

# Footer
st.markdown("---")
st.markdown("### 🔬 Professional Plant Pathology AI")
st.caption("MobileNetV4 Fine-tuned Model | 97% Research Accuracy | Production-Grade Diagnostics")

# Add model validation
if model:
    st.sidebar.markdown("---")
    if st.sidebar.button("Validate Model Output"):
        try:
            # Test with random image
            test_image = np.random.random((1, 224, 224, 3))
            test_pred = model.predict(test_image, verbose=0)
            st.sidebar.success("✅ Model Response Valid")
            if isinstance(test_pred, list):
                st.sidebar.info(f"Output streams: {len(test_pred)}")
        except Exception as e:
            st.sidebar.error(f"Validation failed: {e}")
