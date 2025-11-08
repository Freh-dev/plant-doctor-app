# streamlit_app.py - PROFESSIONAL UI VERSION
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# Set page config with professional theme
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2E8B57;
        --secondary-green: #3CB371;
        --light-green: #F0FFF0;
        --dark-green: #228B22;
        --accent-color: #FF6B35;
    }
    
    .main-header {
        font-size: 3.5rem;
        color: var(--primary-green);
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.4rem;
        color: #666;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: var(--primary-green);
        margin: 2rem 0 1rem 0;
        border-bottom: 2px solid var(--light-green);
        padding-bottom: 0.5rem;
    }
    
    .status-card {
        background: white;
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        border-left: 4px solid var(--secondary-green);
    }
    
    .upload-area {
        border: 3px dashed var(--secondary-green);
        border-radius: 20px;
        padding: 4rem 2rem;
        text-align: center;
        background: var(--light-green);
        margin: 2rem 0;
        transition: all 0.3s ease;
    }
    
    .upload-area:hover {
        background: #E8F5E8;
        border-color: var(--primary-green);
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 15px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .tip-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 3px solid var(--secondary-green);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 1rem 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.8rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(46, 139, 87, 0.4);
        background: linear-gradient(135deg, var(--dark-green), var(--primary-green));
    }
    
    .success-badge {
        background: var(--secondary-green);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    .warning-badge {
        background: #FFA500;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8fff8, #ffffff);
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

# Main App Header
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered plant health assistant</p>', unsafe_allow_html=True)

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    # Upload Section
    st.markdown('<div class="section-header">📸 Upload Plant Image</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="upload-area">
        <h3 style="color: #2E8B57; margin-bottom: 1rem;">🌿 Drag and drop your plant leaf here</h3>
        <p style="color: #666; font-size: 1rem;">Supported formats: JPG, JPEG, PNG</p>
        <p style="color: #888; font-size: 0.9rem; margin-top: 0.5rem;">Maximum file size: 200MB</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        # Display image and analysis
        st.image(image, caption="📷 Your Plant Leaf", use_container_width=True)
        
        if st.button("🔍 Analyze Plant Health", type="primary"):
            with st.spinner("🔬 Analyzing your plant... This may take a few seconds."):
                disease, confidence, error = predict_image(image)
                
                if error:
                    st.error("""
                    ## Analysis Failed
                    We couldn't process your image. Please try again with a clearer photo of a plant leaf.
                    """)
                else:
                    # Display results
                    st.markdown('<div class="section-header">📋 Diagnosis Results</div>', unsafe_allow_html=True)
                    
                    # Format disease name
                    formatted_disease = disease.replace('_', ' ').title()
                    
                    # Determine status
                    if "healthy" in disease.lower():
                        status_emoji = "✅"
                        status_badge = '<span class="success-badge">Healthy Plant</span>'
                    else:
                        status_emoji = "⚠️"
                        status_badge = '<span class="warning-badge">Needs Attention</span>'
                    
                    # Diagnosis card
                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <div style="text-align: center; margin-bottom: 1.5rem;">
                            <div style="font-size: 3rem; margin-bottom: 1rem;">{status_emoji}</div>
                            {status_badge}
                        </div>
                        <h3 style="color: #2E8B57; text-align: center; margin-bottom: 1rem;">{formatted_disease}</h3>
                        <div style="text-align: center;">
                            <p style="font-size: 1.2rem; color: #666;">Confidence Level</p>
                            <h2 style="color: #2E8B57; font-size: 2.5rem; margin: 0.5rem 0;">{confidence:.1%}</h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence notes
                    if confidence < 0.5:
                        st.warning("**Note:** Confidence level is moderate. For best results, ensure clear, well-lit photos of the affected leaves.")
                    elif confidence > 0.85:
                        st.success("**Note:** High confidence diagnosis detected.")
                    
                    # Extract plant name for advice
                    if '_' in disease:
                        plant_name = disease.split('_')[0]
                    else:
                        plant_name = "plant"
                    
                    # Care instructions
                    st.markdown('<div class="section-header">💡 Care Instructions</div>', unsafe_allow_html=True)
                    
                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = chatbot_helper.generate_advice(plant_name, disease)
                        
                        if "OpenAI" in advice or "API key" in advice:
                            st.warning("Using standard care advice")
                            display_fallback_advice(plant_name, disease)
                        else:
                            st.success("✅ AI-Generated Personalized Advice")
                            st.info(advice)
                    else:
                        st.warning("Using standard care advice")
                        display_fallback_advice(plant_name, disease)

def display_fallback_advice(plant_name, disease):
    """Display fallback care advice"""
    st.info(f"""
    **Recommended treatment for {disease.replace('_', ' ').title()}:**
    
    🌱 **Immediate Actions:**
    - Remove affected leaves immediately to prevent spread
    - Isolate plant if possible to protect others
    - Clean tools after handling infected plants
    
    💧 **Watering & Environment:**
    - Water at the base, avoid wetting leaves
    - Improve air circulation around plants
    - Ensure proper drainage to prevent root issues
    
    🛡️ **Treatment:**
    - Apply appropriate organic or chemical treatment
    - Monitor plant recovery daily
    - Adjust sunlight exposure as needed
    
    📞 **When to Seek Help:**
    - Condition worsens after 3-5 days
    - Multiple plants affected
    - Consult local garden center for severe cases
    """)

with col2:
    # Sidebar content
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h2 style="color: #2E8B57;">System Status</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Status cards
    st.markdown("""
    <div class="status-card">
        <h4 style="color: #2E8B57; margin-bottom: 0.5rem;">✅ Model Active</h4>
        <p style="color: #666; margin: 0;">Plant diagnosis ready</p>
    </div>
    """, unsafe_allow_html=True)
    
    if openai_ready:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #2E8B57; margin-bottom: 0.5rem;">✅ AI Advice Ready</h4>
            <p style="color: #666; margin: 0;">Personalized care tips</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #FFA500; margin-bottom: 0.5rem;">⚠️ Basic Advice</h4>
            <p style="color: #666; margin: 0;">Standard care only</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Plant Types Metric
    st.markdown(f"""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 2rem;">{len(class_names)}</h3>
        <p style="margin: 0; opacity: 0.9;">Plant Types</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips Section
    st.markdown("""
    <div style="margin-top: 2rem;">
        <h3 style="color: #2E8B57; border-bottom: 2px solid #F0FFF0; padding-bottom: 0.5rem;">💡 Tips for Best Results</h3>
    </div>
    """, unsafe_allow_html=True)
    
    tips = [
        "Use clear, well-lit photos",
        "Focus on the affected leaves",
        "Include a plain background",
        "Take multiple angles if unsure",
        "Regular monitoring is key"
    ]
    
    for tip in tips:
        st.markdown(f"""
        <div class="tip-card">
            <p style="margin: 0; color: #555;">{tip}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Setup instructions if OpenAI not ready
    if not openai_ready:
        st.markdown("""
        <div style="margin-top: 2rem; padding: 1.5rem; background: #FFF3CD; border-radius: 10px; border-left: 4px solid #FFA500;">
            <h4 style="color: #856404; margin-bottom: 0.5rem;">🚀 Enable AI Features</h4>
            <p style="color: #856404; margin: 0; font-size: 0.9rem;">
                Add your OpenAI API key to enable personalized AI care advice
            </p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p style="margin: 0; font-size: 1rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
