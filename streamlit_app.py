# streamlit_app.py - COMPLETE UPDATED VERSION
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

# Professional CSS styling with mobile responsiveness
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2E8B57;
        --secondary-green: #3CB371;
        --light-green: #F0FFF0;
        --dark-green: #228B22;
        --accent-color: #FF6B35;
        --warning-color: #FFA500;
        --error-color: #DC3545;
    }
    
    .main-header {
        font-size: 3rem;
        color: var(--primary-green);
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 300;
    }
    
    .section-header {
        font-size: 1.5rem;
        color: var(--primary-green);
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid var(--light-green);
        padding-bottom: 0.5rem;
    }
    
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid var(--secondary-green);
    }
    
    .upload-area {
        border: 3px dashed var(--secondary-green);
        border-radius: 15px;
        padding: 3rem 1.5rem;
        text-align: center;
        background: var(--light-green);
        margin: 1.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .upload-area:hover {
        background: #E8F5E8;
        border-color: var(--primary-green);
        transform: translateY(-2px);
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .tip-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.6rem 0;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
        border-left: 3px solid var(--secondary-green);
    }
    
    .metric-card {
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        color: white;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        margin: 0.8rem 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, var(--primary-green), var(--dark-green));
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.3);
        background: linear-gradient(135deg, var(--dark-green), var(--primary-green));
    }
    
    .success-badge {
        background: var(--secondary-green);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .warning-badge {
        background: var(--warning-color);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    .error-badge {
        background: var(--error-color);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 15px;
        font-weight: 600;
        display: inline-block;
        margin: 0.3rem 0;
        font-size: 0.9rem;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .sub-header {
            font-size: 1rem;
        }
        
        .section-header {
            font-size: 1.2rem;
        }
        
        .upload-area {
            padding: 2rem 1rem;
        }
        
        .status-card, .diagnosis-card, .tip-card, .metric-card {
            margin: 0.5rem 0;
            padding: 1rem;
        }
        
        /* Fix sidebar for mobile */
        .sidebar .sidebar-content {
            position: relative !important;
            height: auto !important;
        }
    }
    
    /* Fix Streamlit default styles */
    .stApp {
        max-width: 100% !important;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: var(--primary-green);
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

def display_fallback_advice(plant_name, disease):
    """Display fallback care advice"""
    formatted_disease = disease.replace('_', ' ').title()
    
    st.info(f"""
    **🌱 Recommended Treatment for {formatted_disease}**
    
    ### 🚨 Immediate Actions
    - Remove affected leaves immediately to prevent spread
    - Isolate plant if possible to protect others
    - Clean tools thoroughly after handling infected plants
    
    ### 💧 Watering & Environment
    - Water at the base only, avoid wetting leaves
    - Improve air circulation around the plant
    - Ensure proper drainage to prevent root issues
    - Maintain consistent temperature and humidity
    
    ### 🛡️ Treatment Plan
    - Apply appropriate organic or chemical treatment
    - Monitor plant recovery daily
    - Adjust sunlight exposure as needed
    - Consider soil amendments if necessary
    
    ### 📞 When to Seek Help
    - Condition worsens after 3-5 days of treatment
    - Multiple plants become affected
    - Consult local garden center for severe cases
    - Consider professional plant pathology services
    """)

# Load resources
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()
img_size = (128, 128)

# Main App Header
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your AI-powered plant health assistant</p>', unsafe_allow_html=True)

# Check if model is available
if model is None:
    st.error("""
    ## 🔧 Service Temporarily Unavailable
    
    We're experiencing technical difficulties with our plant diagnosis model. 
    
    **Please try:**
    - Refreshing the page
    - Checking your internet connection
    - Trying again in a few minutes
    
    If the problem persists, please contact support.
    """)
    st.stop()

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    # Upload Section with visible file uploader
    st.markdown('<div class="section-header">📸 Upload Plant Image</div>', unsafe_allow_html=True)
    
    # Visible file uploader with custom styling
    uploaded_file = st.file_uploader(
        "Choose a plant leaf image",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG • Maximum file size: 200MB",
        key="main_uploader"
    )
    
    # Show upload area when no file is selected
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-area" onclick="document.querySelector('[data-testid=fileUploader] input').click()">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # File is uploaded - show preview and analysis options
        try:
            image = Image.open(uploaded_file)
            
            # Image preview and info
            st.image(image, caption="📷 Your Plant Leaf", use_container_width=True)
            
            # Image details
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)  # Size in MB
            st.caption(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size:.1f} MB")
            
            # Analysis button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant... This may take a few seconds."):
                    # Add progress simulation
                    progress_bar = st.progress(0)
                    
                    # Simulate progress steps
                    progress_bar.progress(20)
                    st.write("📤 Uploading image...")
                    
                    progress_bar.progress(40)
                    st.write("🔍 Processing image...")
                    
                    # Perform prediction
                    disease, confidence, error = predict_image(image)
                    
                    progress_bar.progress(80)
                    st.write("📊 Analyzing results...")
                    
                    progress_bar.progress(100)
                    
                    if error:
                        st.error(f"""
                        ## ❌ Analysis Failed
                        
                        **Error Details:** {error}
                        
                        Please try:
                        - A different image format
                        - A clearer photo of the plant leaf
                        - Checking file integrity
                        """)
                    else:
                        # Display results
                        st.markdown('<div class="section-header">📋 Diagnosis Results</div>', unsafe_allow_html=True)
                        
                        # Format disease name
                        formatted_disease = disease.replace('_', ' ').title()
                        
                        # Determine status and styling
                        if "healthy" in disease.lower():
                            status_emoji = "✅"
                            status_badge = '<span class="success-badge">Healthy Plant</span>'
                            status_color = "#2E8B57"
                        else:
                            status_emoji = "⚠️"
                            status_badge = '<span class="warning-badge">Needs Attention</span>'
                            status_color = "#FFA500"
                        
                        # Diagnosis card
                        st.markdown(f"""
                        <div class="diagnosis-card">
                            <div style="text-align: center; margin-bottom: 1.2rem;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                                {status_badge}
                            </div>
                            <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">{formatted_disease}</h3>
                            <div style="text-align: center;">
                                <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                                <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">{confidence:.1%}</h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence feedback
                        if confidence < 0.5:
                            st.warning("""
                            **📝 Note:** Confidence level is moderate. 
                            For best results:
                            - Ensure clear, well-lit photos
                            - Focus on the affected leaves
                            - Use a plain background
                            - Take multiple angles if unsure
                            """)
                        elif confidence > 0.85:
                            st.success("**✅ High confidence diagnosis detected.**")
                        
                        # Extract plant name for advice
                        if '_' in disease:
                            plant_name = disease.split('_')[0]
                        else:
                            plant_name = "plant"
                        
                        # Care instructions section
                        st.markdown("---")
                        st.markdown('<div class="section-header">💡 Care Instructions</div>', unsafe_allow_html=True)
                        
                        if openai_ready:
                            with st.spinner("🤖 Generating personalized care advice..."):
                                advice = chatbot_helper.generate_advice(plant_name, disease)
                            
                            # Check if we got real AI advice
                            if "OpenAI" in advice or "API key" in advice:
                                st.warning("⚠️ Using standard care advice (AI service unavailable)")
                                display_fallback_advice(plant_name, disease)
                            else:
                                st.success("✅ AI-Generated Personalized Advice")
                                st.info(advice)
                        else:
                            st.warning("⚠️ Using standard care advice")
                            display_fallback_advice(plant_name, disease)
                            
                            # Show setup prompt
                            st.info("""
                            **💡 Enable AI Features:**
                            Add your OpenAI API key to get personalized AI care advice tailored to your specific plant condition.
                            """)
        
        except Exception as e:
            st.error(f"""
            ## ❌ Error Processing Image
            
            **Details:** {str(e)}
            
            Please try:
            - A different image file
            - Checking the file format
            - Ensuring the image isn't corrupted
            """)

with col2:
    # Sidebar content in a container for better mobile handling
    with st.container():
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <h2 style="color: #2E8B57; margin-bottom: 0.5rem;">System Status</h2>
            <p style="color: #666; font-size: 0.9rem;">Real-time service monitoring</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Status cards
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #2E8B57; margin-bottom: 0.3rem;">✅ Model Active</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Plant diagnosis ready</p>
        </div>
        """, unsafe_allow_html=True)
        
        if openai_ready:
            st.markdown("""
            <div class="status-card">
                <h4 style="color: #2E8B57; margin-bottom: 0.3rem;">✅ AI Advice Ready</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Personalized care tips</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card">
                <h4 style="color: #FFA500; margin-bottom: 0.3rem;">⚠️ Basic Advice</h4>
                <p style="color: #666; margin: 0; font-size: 0.9rem;">Standard care only</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Plant Types Metric
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
            <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Tips Section
        st.markdown("""
        <div style="margin-top: 1.5rem;">
            <h3 style="color: #2E8B57; border-bottom: 2px solid #F0FFF0; padding-bottom: 0.5rem; font-size: 1.2rem;">
                💡 Tips for Best Results
            </h3>
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
                <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Setup instructions if OpenAI not ready
        if not openai_ready:
            st.markdown("""
            <div style="margin-top: 1.5rem; padding: 1.2rem; background: #FFF3CD; border-radius: 8px; border-left: 4px solid #FFA500;">
                <h4 style="color: #856404; margin-bottom: 0.5rem; font-size: 1rem;">🚀 Enable AI Features</h4>
                <p style="color: #856404; margin: 0; font-size: 0.8rem;">
                    Add your OpenAI API key to enable personalized AI care advice with tailored recommendations.
                </p>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
    <p style="margin: 0.3rem 0 0 0; font-size: 0.8rem; color: #888;">
        Upload plant leaf images for instant diagnosis and care advice
    </p>
</div>
""", unsafe_allow_html=True)

# JavaScript for better upload interaction
st.markdown("""
<script>
// Make the upload area clickable
document.addEventListener('DOMContentLoaded', function() {
    const uploadArea = document.querySelector('.upload-area');
    if (uploadArea) {
        uploadArea.style.cursor = 'pointer';
        uploadArea.addEventListener('click', function() {
            const fileInput = document.querySelector('[data-testid="fileUploader"] input');
            if (fileInput) {
                fileInput.click();
            }
        });
    }
});
</script>
""", unsafe_allow_html=True)
