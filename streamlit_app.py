# streamlit_app.py - UPDATED VERSION WITH CONFIDENCE FIXES
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
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize prediction history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Simple CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .upload-area {
        border: 3px dashed #3CB371;
        border-radius: 15px;
        padding: 3rem 1.5rem;
        text-align: center;
        background: #F0FFF0;
        margin: 1.5rem 0;
        transition: all 0.3s ease;
    }
    
    .status-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #3CB371;
    }
    
    .diagnosis-card {
        background: linear-gradient(135deg, #ffffff, #f8fff8);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.2rem 0;
        box-shadow: 0 4px 12px rgba(46, 139, 87, 0.15);
        border: 1px solid #e0f0e0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #2E8B57, #228B22);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        width: 100%;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #FFF3CD, #FFEAA7);
        border: 2px solid #FFA500;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    .error-box {
        background: linear-gradient(135deg, #F8D7DA, #F5C6CB);
        border: 2px solid #DC3545;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Debug function
def check_openai_setup():
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
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

MODEL_PATH = "plant_disease_final_model.keras"
#MODEL_LINK = "/content/drive/MyDrive/ML_AI_Project/model"  # put your real link
MODEL_LINK ="https://drive.google.com/drive/folders/1AcD9IR_tWrfUSzMQS_hBrOktp_bztCkT"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error("❌ Model file not found.")
        st.sidebar.write("Please download the model from:")
        st.sidebar.write(MODEL_LINK)
        st.sidebar.write("and place it in the same folder as `streamlit_app.py`.")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ Advanced Model Loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Model Error: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_final.json", "r") as f:
            class_names = json.load(f)
            st.sidebar.info(f"✅ Loaded {len(class_names)} plant classes")
            return class_names
    except Exception as e:
        st.sidebar.warning("Using default class names")
        return ["Apple_healthy", "Apple_apple_scab", "Tomato_healthy", "Tomato_early_blight", "Tomato_late_blight"]

# SIMPLIFIED PREDICTION FUNCTION - NO CONFIDENCE CHECKS
def predict_image(image):
    try:
        img = image.resize(img_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction))
        
        # JUST RETURN THE PREDICTION - NO CONFIDENCE CHECKS HERE
        return predicted_class, confidence, None
            
    except Exception as e:
        return None, None, str(e)

def debug_model_predictions(image):
    """Debug what the model is actually predicting"""
    img = image.resize(img_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array, verbose=0)[0]
    
    # Get top 5 predictions
    top_5_indices = np.argsort(prediction)[-5:][::-1]
    
    st.write("🔍 **Debug - Top 5 Predictions:**")
    for i, idx in enumerate(top_5_indices):
        st.write(f"{i+1}. {class_names[idx]} - {prediction[idx]:.3f} ({prediction[idx]*100:.1f}%)")
    
    return prediction

def get_plant_advice(plant_name, disease):
    """Get AI advice with fallback handling"""
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "AI service rate limit reached. Using expert care guide."
        else:
            return f"AI service temporarily unavailable. Using expert care guide."

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
    
    ### 🛡️ Treatment Plan
    - Apply appropriate organic or chemical treatment
    - Monitor plant recovery daily
    - Adjust sunlight exposure as needed
    """)

# Load resources
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()
img_size = (128, 128)

# Debug information in sidebar
with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Number of classes: {len(class_names)}")
    st.write(f"OpenAI ready: {openai_ready}")
    if class_names:
        st.write("Sample classes:", class_names[:5])

# Main App Header
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #666; margin-bottom: 2rem;">Upload a plant leaf photo for instant diagnosis and care advice</p>', unsafe_allow_html=True)

# Check if model is available
if model is None:
    st.error("""
    ## 🔧 Service Temporarily Unavailable
    
    **Model loading failed.** Please check:
    - Model files exist in the repository
    - Files are not corrupted
    - TensorFlow version compatibility
    
    Currently loaded classes: {}
    """.format(len(class_names)))
    st.stop()

# Main content layout
col1, col2 = st.columns([2, 1])

with col1:
    # Upload Section
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a plant leaf image**")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG • Max 200MB",
        label_visibility="collapsed"
    )
    
    # Show custom upload area when no file is selected
    if uploaded_file is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Show file info and preview when file is uploaded
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            
            # Show success message
            st.success(f"✅ **File uploaded successfully!**")
            st.write(f"**Filename:** {uploaded_file.name}")
            
            # Image preview
            st.image(image, caption="📷 Your Plant Leaf", width=400)
            
            # File info
            file_size = len(uploaded_file.getvalue()) / (1024 * 1024)
            st.write(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size:.1f} MB")
            
            # Analysis button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant... This may take a few seconds."):
                    # Perform prediction
                    disease, confidence, error = predict_image(image)
                    
                    if error:
                        st.error(f"""
                        ## ❌ Analysis Failed
                        **Error:** {error}
                        Please try uploading a different image.
                        """)
                    else:
                        # Track prediction history for bias detection
                        st.session_state.prediction_history.append(disease)
                        if len(st.session_state.prediction_history) > 5:
                            st.session_state.prediction_history.pop(0)
                        
                        # Display results
                        st.subheader("📋 Diagnosis Results")
                        
                        # Format disease name
                        formatted_disease = disease.replace('_', ' ').title()
                        
                        # Determine status
                        if "healthy" in disease.lower():
                            status_emoji = "✅"
                            status_text = "Healthy Plant"
                            status_color = "#2E8B57"
                        else:
                            status_emoji = "⚠️"
                            status_text = "Needs Attention"
                            status_color = "#FFA500"
                        
                        # Diagnosis card
                        st.markdown(f"""
                        <div class="diagnosis-card">
                            <div style="text-align: center; margin-bottom: 1.2rem;">
                                <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                                <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem; border-radius: 15px; font-weight: 600;">{status_text}</span>
                            </div>
                            <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">{formatted_disease}</h3>
                            <div style="text-align: center;">
                                <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                                <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">{confidence:.1%}</h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # UPDATED CONFIDENCE WARNINGS WITH FIXED THRESHOLDS
                        if confidence < 0.4:
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Low Confidence Warning</h4>
                                <p>The model is not confident about this diagnosis. This could be because:</p>
                                <ul>
                                    <li>Image quality is poor</li>
                                    <li>Plant type not well represented in training data</li>
                                    <li>Unusual angle or lighting</li>
                                    <li>Multiple diseases present</li>
                                </ul>
                                <p><strong>Recommendation:</strong> Try a clearer, well-lit image of the leaf.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        elif confidence < 0.75:  # CHANGED FROM 0.7 to 0.75
                            st.markdown("""
                            <div class="warning-box">
                                <h4>⚠️ Moderate Confidence</h4>
                                <p>This prediction has moderate confidence. Consider:</p>
                                <ul>
                                    <li>Getting a second opinion from a plant expert</li>
                                    <li>Uploading additional images from different angles</li>
                                    <li>Monitoring the plant for additional symptoms</li>
                                </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.success("**✅ High Confidence** - Diagnosis is reliable.")
                        
                        # Check for corn bias
                        corn_count = sum(1 for p in st.session_state.prediction_history if 'corn' in p.lower())
                        if corn_count >= 3:
                            st.markdown("""
                            <div class="error-box">
                                <h4>🚨 Model Bias Detected</h4>
                                <p>The model appears biased toward corn predictions. This may indicate:</p>
                                <ul>
                                    <li>Training data imbalance</li>
                                    <li>Model limitations with other plant types</li>
                                    <li>Need for model retraining</li>
                                </ul>
                                <p><strong>Note:</strong> Predictions for non-corn plants may be less reliable.</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Debug info
                        with st.expander("🔍 Debug Information"):
                            st.write(f"Predicted class: {disease}")
                            st.write(f"Raw confidence: {confidence}")
                            st.write(f"Model input size: {img_size}")
                            st.write(f"Available classes: {len(class_names)}")
                            st.write(f"Recent predictions: {st.session_state.prediction_history}")
                            
                            # Show detailed predictions
                            debug_model_predictions(image)
                        
                        # User feedback for wrong predictions
                        st.markdown("---")
                        st.subheader("🤔 Prediction Accuracy")
                        feedback = st.radio(
                            "Does this prediction seem correct?",
                            ["Yes, looks accurate", "No, this seems wrong", "Unsure"]
                        )
                        if feedback == "No, this seems wrong":
                            st.warning("Thank you for the feedback! We'll use this to improve the model.")
                        
                        # Extract plant name for advice
                        if '_' in disease:
                            plant_name = disease.split('_')[0]
                        else:
                            plant_name = "plant"
                        
                        # Care instructions
                        st.markdown("---")
                        st.subheader("💡 Care Instructions")
                        
                        if openai_ready:
                            with st.spinner("🤖 Generating personalized care advice..."):
                                advice = get_plant_advice(plant_name, disease)
                            
                            if "OpenAI" in advice or "API key" in advice or "rate limit" in advice.lower():
                                st.warning("⚠️ Using standard care advice (AI service unavailable)")
                                display_fallback_advice(plant_name, disease)
                            else:
                                st.success("✅ AI-Generated Personalized Advice")
                                st.info(advice)
                        else:
                            st.warning("⚠️ Using standard care advice")
                            display_fallback_advice(plant_name, disease)
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

with col2:
    # Sidebar content
    st.subheader("System Status")
    st.write("Real-time service monitoring")
    
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
    <div style="background: linear-gradient(135deg, #2E8B57, #228B22); color: white; border-radius: 10px; padding: 1.2rem; text-align: center; margin: 0.8rem 0;">
        <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Tips Section
    st.subheader("💡 Tips for Best Results")
    
    tips = [
        "Use clear, well-lit photos",
        "Focus on the affected leaves", 
        "Include a plain background",
        "Take multiple angles if unsure",
        "Regular monitoring is key"
    ]
    
    for tip in tips:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem; margin: 0.6rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08); border-left: 3px solid #3CB371;">
            <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
