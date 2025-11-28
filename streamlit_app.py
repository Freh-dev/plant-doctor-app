# streamlit_app.py - FIXED VERSION WITH PROPER FILE HANDLING
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# 🆕 EfficientNet preprocessing (must match training)
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------- CONFIGURATION ----------------------- #
class AppConfig:
    MODEL_PATH = "plant_disease_final_model.keras"
    CLASS_NAMES_PATH = "class_names_final.json"
    MAX_FILE_SIZE_MB = 200
    SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
    DEFAULT_IMG_SIZE = (224, 224)
    CONFIDENCE_THRESHOLDS = {
        "low": 0.4,
        "medium": 0.75,
        "high": 0.9
    }

class PlantStatus:
    HEALTHY = "healthy"
    DISEASED = "diseased"

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- STYLING --------------------------- #
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
    
    .success-box {
        background: linear-gradient(135deg, #D1ECF1, #B8E6B8);
        border: 2px solid #28A745;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- SESSION STATE --------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

# ----------------------- VALIDATION HELPERS ---------------- #
def validate_environment():
    """Check all required dependencies and configurations"""
    missing_files = []
    
    if not os.path.exists(AppConfig.MODEL_PATH):
        missing_files.append(AppConfig.MODEL_PATH)
    
    if not os.path.exists(AppConfig.CLASS_NAMES_PATH):
        missing_files.append(AppConfig.CLASS_NAMES_PATH)
    
    return missing_files

def validate_image_file(uploaded_file):
    """Validate uploaded image file"""
    errors = []
    
    if uploaded_file is None:
        errors.append("No file provided")
        return errors
    
    # Check file size
    try:
        file_size = len(uploaded_file.getvalue())
        max_size_bytes = AppConfig.MAX_FILE_SIZE_MB * 1024 * 1024
        
        if file_size > max_size_bytes:
            errors.append(f"File size exceeds {AppConfig.MAX_FILE_SIZE_MB}MB limit")
        
        if file_size == 0:
            errors.append("File is empty")
    except Exception as e:
        errors.append(f"Could not read file: {e}")
    
    # Check file format
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in AppConfig.SUPPORTED_FORMATS:
            errors.append(f"Unsupported file format. Use: {', '.join(AppConfig.SUPPORTED_FORMATS)}")
    except Exception as e:
        errors.append(f"Invalid filename: {e}")
    
    return errors

def validate_image_content(image):
    """Validate image content and dimensions"""
    errors = []
    
    if image is None:
        errors.append("No image data")
        return errors
    
    # Check image dimensions
    min_dimension = 50
    if image.size[0] < min_dimension or image.size[1] < min_dimension:
        errors.append(f"Image too small. Minimum dimension: {min_dimension}px")
    
    # Check if image is valid and can be processed
    try:
        image.verify()
    except Exception as e:
        errors.append(f"Invalid image file: {str(e)}")
    
    return errors

def safe_open_image(uploaded_file):
    """Safely open image with proper error handling"""
    try:
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        # Open image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if necessary (handles PNG with transparency, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return image, None
    except Exception as e:
        return None, f"Failed to open image: {str(e)}"

# ----------------------- CORE HELPERS ---------------------- #
def check_openai_setup():
    """Check if OpenAI advice helper is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        test_advice = chatbot_helper.generate_advice("tomato", "healthy")
        # If we get a normal sentence back, assume OK
        if "OpenAI" not in test_advice and "API key" not in test_advice:
            return True
        return False
    except Exception:
        return False

@st.cache_resource
def load_model():
    """Load the trained Keras model from local file."""
    if not os.path.exists(AppConfig.MODEL_PATH):
        st.sidebar.error(f"❌ Model file not found: {AppConfig.MODEL_PATH}")
        st.sidebar.write(f"Looking for: {os.path.abspath(AppConfig.MODEL_PATH)}")
        return None

    try:
        model = tf.keras.models.load_model(AppConfig.MODEL_PATH)
        st.sidebar.success("✅ Advanced model loaded (EfficientNet-based)")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from json file."""
    try:
        with open(AppConfig.CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        st.sidebar.info(f"✅ Loaded {len(class_names)} plant classes")
        return class_names
    except Exception as e:
        st.sidebar.warning(f"Could not load {AppConfig.CLASS_NAMES_PATH}: {e}")
        # Return minimal fallback classes
        return [
            "Apple_healthy",
            "Apple_apple_scab", 
            "Tomato_healthy",
            "Tomato_early_blight",
            "Tomato_late_blight"
        ]

def validate_model_compatibility(model):
    """Verify model expects correct input format"""
    if not hasattr(model, 'input_shape'):
        st.warning("⚠️ Model compatibility cannot be verified")
        return True
    
    expected_channels = 3  # RGB
    if model.input_shape[-1] != expected_channels:
        st.error(f"❌ Model expects {model.input_shape[-1]} channels, but preprocessing uses {expected_channels}")
        return False
    
    return True

# 🆕 Central preprocessing function, matching EfficientNet training
def preprocess_image_for_model(image, img_size):
    """
    Convert PIL image → RGB → resize → EfficientNet preprocess → add batch dimension.
    """
    # Ensure image is in RGB mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
        
    img = image.resize(img_size)
    img_array = np.array(img).astype("float32")
    # IMPORTANT: use EfficientNet preprocess_input (no /255 here)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_image(image, model, class_names, img_size):
    """Run model prediction on a PIL.Image and return (class, confidence, error_msg)."""
    try:
        img_batch = preprocess_image_for_model(image, img_size)
        prediction = model.predict(img_batch, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        
        if predicted_index >= len(class_names):
            return None, None, f"Prediction index {predicted_index} out of range for {len(class_names)} classes"
            
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        return predicted_class, confidence, None
    except Exception as e:
        return None, None, str(e)

def get_confidence_level(confidence):
    """Determine confidence level for display"""
    if confidence < AppConfig.CONFIDENCE_THRESHOLDS["low"]:
        return "low"
    elif confidence < AppConfig.CONFIDENCE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "high"

def debug_model_predictions(image, model, class_names, img_size):
    """Show top-5 predictions for debugging."""
    try:
        img_batch = preprocess_image_for_model(image, img_size)
        prediction = model.predict(img_batch, verbose=0)[0]
        top_5_indices = np.argsort(prediction)[-5:][::-1]

        st.write("🔍 **Debug - Top 5 Predictions:**")
        for rank, idx in enumerate(top_5_indices, start=1):
            if idx < len(class_names):
                st.write(f"{rank}. {class_names[idx]} - {prediction[idx]:.3f} ({prediction[idx]*100:.1f}%)")
            else:
                st.write(f"{rank}. [Index {idx} out of range] - {prediction[idx]:.3f}")
    except Exception as e:
        st.error(f"Debug prediction failed: {e}")

def get_plant_advice(plant_name, disease):
    """Try to get advice from chatbot_helper, fall back if error."""
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "AI service rate limit reached. Using standard care advice instead."
        return "AI advice currently unavailable. Using standard care advice instead."

def display_fallback_advice(plant_name, disease):
    """Static care guide if AI advice is unavailable."""
    formatted_disease = disease.replace("_", " ").title()
    formatted_plant = plant_name.replace("_", " ").title()
    
    st.info(f"""
    **🌱 Recommended Treatment for {formatted_disease} on {formatted_plant}**
    
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
    - Consider soil testing for nutrient deficiencies
    """)

def handle_prediction_error(error):
    """Provide user-friendly error messages"""
    error_str = str(error).lower()
    
    if "memory" in error_str:
        return "Insufficient memory. Try uploading a smaller image or closing other applications."
    elif "shape" in error_str or "dimension" in error_str:
        return "Image format not supported. Please try a different image."
    elif "index" in error_str:
        return "Model configuration error. Please check the class names file."
    elif "nonetype" in error_str or "seek" in error_str:
        return "File upload error. Please try uploading the image again."
    else:
        return f"Analysis failed: {error}"

# ----------------------- INITIALIZATION -------------------- #
# Validate environment first
missing_files = validate_environment()

# Load resources
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

# Determine image size
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = AppConfig.DEFAULT_IMG_SIZE
    st.sidebar.warning(f"Using default image size: {img_size}")

# Validate model compatibility
if model is not None:
    model_compatible = validate_model_compatibility(model)

# ----------------------- SIDEBAR --------------------------- #
with st.sidebar:
    st.header("🔧 System Info")
    
    # Status indicators
    st.write(f"✅ Model loaded: {model is not None}")
    st.write(f"✅ Classes loaded: {len(class_names) if class_names else 0}")
    st.write(f"✅ OpenAI ready: {openai_ready}")
    st.write(f"✅ Image size: {img_size}")
    
    if missing_files:
        st.error("❌ Missing files:")
        for file in missing_files:
            st.write(f"   - {file}")
    
    # Current directory info
    with st.expander("📁 Directory Contents"):
        try:
            current_files = [f for f in os.listdir('.') if os.path.isfile(f)]
            st.write("Files in current directory:")
            for file in current_files[:15]:  # Show first 15 files
                st.write(f"   - {file}")
            if len(current_files) > 15:
                st.write(f"   ... and {len(current_files) - 15} more files")
        except Exception as e:
            st.write(f"Could not read directory: {e}")

# ----------------------- MAIN HEADER ----------------------- #
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    'Upload a plant leaf photo for instant AI-powered diagnosis and care advice.'
    '</p>',
    unsafe_allow_html=True
)

# Show critical errors and stop if necessary
if not model or not class_names:
    st.error(f"""
    ## 🔧 Service Configuration Required
    
    **Critical resources missing:**
    - Model file: {AppConfig.MODEL_PATH} {'❌' if not model else '✅'}
    - Class names: {AppConfig.CLASS_NAMES_PATH} {'❌' if not class_names else '✅'}
    
    **Please ensure:**
    - Both files exist in the application directory
    - Files are not corrupted
    - You have read permissions for these files
    
    **Current directory:** {os.getcwd()}
    """)
    st.stop()

# ----------------------- MAIN LAYOUT ----------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a clear plant leaf image**")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=AppConfig.SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(AppConfig.SUPPORTED_FORMATS)} • Max {AppConfig.MAX_FILE_SIZE_MB}MB",
        label_visibility="collapsed"
    )

    # Store uploaded file data in session state to prevent file object issues
    if uploaded_file is not None:
        st.session_state.uploaded_file_data = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name

    # Nice empty state
    if uploaded_file is None and st.session_state.uploaded_file_data is None:
        st.markdown(f"""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">
                JPG, PNG, JPEG • Max {AppConfig.MAX_FILE_SIZE_MB}MB
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Process uploaded file
    if st.session_state.uploaded_file_data is not None:
        try:
            # Create a new file-like object from stored data
            from io import BytesIO
            file_like_object = BytesIO(st.session_state.uploaded_file_data)
            
            # Validate file before processing
            file_errors = validate_image_file(uploaded_file if uploaded_file is not None else type('MockFile', (), {
                'getvalue': lambda: st.session_state.uploaded_file_data,
                'name': getattr(st.session_state, 'uploaded_file_name', 'uploaded_image')
            })())
            
            if file_errors:
                for error in file_errors:
                    st.error(f"❌ {error}")
                # Clear invalid file data
                st.session_state.uploaded_file_data = None
                st.session_state.uploaded_file_name = None
                st.stop()

            # Open and validate image
            image, open_error = safe_open_image(file_like_object)
            if open_error:
                st.error(f"❌ {open_error}")
                st.session_state.uploaded_file_data = None
                st.session_state.uploaded_file_name = None
                st.stop()
                
            image_errors = validate_image_content(image)
            if image_errors:
                for error in image_errors:
                    st.error(f"❌ {error}")
                st.session_state.uploaded_file_data = None
                st.session_state.uploaded_file_name = None
                st.stop()

            st.success("✅ **File uploaded successfully!**")
            st.write(f"**Filename:** {getattr(st.session_state, 'uploaded_file_name', 'uploaded_image')}")

            # Preview
            st.image(image, caption="📷 Your Plant Leaf", width=400)

            # File info
            file_size_mb = len(st.session_state.uploaded_file_data) / (1024 * 1024)
            st.write(
                f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size_mb:.1f} MB"
            )

            # Analyze button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant..."):
                    disease, confidence, error = predict_image(
                        image, model, class_names, img_size
                    )

                if error:
                    user_friendly_error = handle_prediction_error(error)
                    st.error(f"""
                    ## ❌ Analysis Failed
                    **Error:** {user_friendly_error}

                    Please try a different image or check the file format.
                    """)
                else:
                    # Track for bias detection
                    st.session_state.prediction_history.append(disease)
                    if len(st.session_state.prediction_history) > 5:
                        st.session_state.prediction_history.pop(0)

                    # ----------------- DIAGNOSIS CARD ----------------- #
                    st.subheader("📋 Diagnosis Results")

                    formatted_disease = (
                        disease.replace("___", " - ")
                               .replace("__", " - ")
                               .replace("_", " ")
                    )

                    if "healthy" in disease.lower():
                        status_emoji = "✅"
                        status_text = "Healthy Plant"
                        status_color = "#2E8B57"
                    else:
                        status_emoji = "⚠️"
                        status_text = "Needs Attention"
                        status_color = "#FFA500"

                    confidence_level = get_confidence_level(confidence)

                    st.markdown(f"""
                    <div class="diagnosis-card">
                        <div style="text-align: center; margin-bottom: 1.2rem;">
                            <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                            <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem;
                                         border-radius: 15px; font-weight: 600;">
                                {status_text}
                            </span>
                        </div>
                        <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">
                            {formatted_disease}
                        </h3>
                        <div style="text-align: center;">
                            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">
                                Confidence Level ({confidence_level})
                            </p>
                            <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                {confidence:.1%}
                            </h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ----------------- CONFIDENCE WARNINGS ------------- #
                    if confidence_level == "low":
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Low Confidence Warning</h4>
                            <p>The model is not very confident about this diagnosis. This may be due to:</p>
                            <ul>
                                <li>Poor image quality or lighting</li>
                                <li>Unusual angle or partial leaf view</li>
                                <li>Plant type underrepresented in training data</li>
                                <li>Multiple diseases present</li>
                            </ul>
                            <p><strong>Recommendation:</strong> Try a clearer, well-lit image focusing on the affected area.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence_level == "medium":
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Moderate Confidence</h4>
                            <p>This prediction has moderate confidence. You may want to:</p>
                            <ul>
                                <li>Get a second opinion from a plant expert</li>
                                <li>Upload additional images from different angles</li>
                                <li>Monitor the plant for new symptoms</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="success-box">
                            <h4>✅ High Confidence</h4>
                            <p>This diagnosis has high confidence and is likely reliable.</p>
                            <p>Proceed with the recommended treatment plan.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # ----------------- CORN BIAS CHECK ----------------- #
                    corn_count = sum(
                        1 for p in st.session_state.prediction_history
                        if "corn" in p.lower()
                    )
                    if corn_count >= 3:
                        st.markdown("""
                        <div class="error-box">
                            <h4>🚨 Potential Model Bias Detected</h4>
                            <p>The model has predicted <strong>corn-related</strong> classes several times in a row.</p>
                            <p>This may indicate:</p>
                            <ul>
                                <li>Training data imbalance</li>
                                <li>Limited performance on non-corn plants</li>
                                <li>Need for future retraining or fine-tuning</li>
                            </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # ----------------- DEBUG EXPANDER ------------------ #
                    with st.expander("🔍 Debug Information"):
                        st.write(f"Predicted class: {disease}")
                        st.write(f"Raw confidence: {confidence}")
                        st.write(f"Confidence level: {confidence_level}")
                        st.write(f"Model input size: {img_size}")
                        st.write(f"Available classes: {len(class_names)}")
                        st.write(f"Recent predictions: {st.session_state.prediction_history}")
                        debug_model_predictions(image, model, class_names, img_size)

                    # ----------------- USER FEEDBACK ------------------- #
                    st.markdown("---")
                    st.subheader("🤔 Prediction Accuracy")
                    
                    if not st.session_state.feedback_submitted:
                        feedback = st.radio(
                            "Does this prediction seem correct?",
                            ["Yes, looks accurate", "No, this seems wrong", "Unsure"],
                            index=0
                        )
                        
                        if st.button("Submit Feedback"):
                            st.session_state.feedback_submitted = True
                            if feedback == "No, this seems wrong":
                                st.warning(
                                    "Thank you for your feedback! This helps us improve the model. "
                                    "Consider consulting with a plant expert for confirmation."
                                )
                            else:
                                st.success("Thank you for your feedback!")
                    else:
                        st.info("📝 Thank you for your feedback!")

                    # ----------------- CARE INSTRUCTIONS --------------- #
                    st.markdown("---")
                    st.subheader("💡 Care Instructions")

                    plant_name = disease.split("_")[0] if "_" in disease else "plant"

                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = get_plant_advice(plant_name, disease)

                        if any(key in advice for key in ["OpenAI", "API key", "rate limit", "unavailable"]):
                            st.warning("⚠️ Using fallback care advice (AI service issue).")
                            display_fallback_advice(plant_name, disease)
                        else:
                            st.success("✅ AI-Generated Personalized Advice")
                            st.info(advice)
                    else:
                        st.warning("⚠️ Using standard care advice (AI not configured).")
                        display_fallback_advice(plant_name, disease)

        except Exception as e:
            st.error(f"❌ Unexpected error processing image: {e}")
            # Clear corrupted file data
            st.session_state.uploaded_file_data = None
            st.session_state.uploaded_file_name = None

with col2:
    # Sidebar / status info
    st.subheader("System Status")
    st.write("Real-time service monitoring")

    # Status cards
    status_cards = [
        ("✅ Model Active", "Plant diagnosis ready", "#2E8B57"),
        ("✅ AI Advice Ready" if openai_ready else "⚠️ Basic Advice Mode", 
         "Personalized care tips enabled" if openai_ready else "Standard care tips only",
         "#2E8B57" if openai_ready else "#FFA500"),
        ("✅ Image Processing", f"Supports {', '.join(AppConfig.SUPPORTED_FORMATS)}", "#2E8B57")
    ]

    for title, description, color in status_cards:
        st.markdown(f"""
        <div class="status-card">
            <h4 style="color: {color}; margin-bottom: 0.3rem;">{title}</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

    # Plant types metric
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2E8B57, #228B22);
                color: white; border-radius: 10px; padding: 1.2rem;
                text-align: center; margin: 0.8rem 0;">
        <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
    </div>
    """, unsafe_allow_html=True)

    # Tips
    st.subheader("💡 Tips for Best Results")

    tips = [
        "Use clear, well-lit photos of leaves",
        "Focus on the affected areas clearly",
        "Include a plain, non-distracting background",
        "Take multiple angles if you're unsure",
        "Avoid blurred or dark images",
        "Capture both sides of the leaves when possible"
    ]

    for tip in tips:
        st.markdown(
            f"""
            <div style="background: white; border-radius: 10px; padding: 1rem;
                        margin: 0.6rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        border-left: 3px solid #3CB371;">
                <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ----------------------- FOOTER ---------------------------- #
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
    <p style="margin: 0.3rem 0; font-size: 0.8rem; color: #888;">
        Supports {len(class_names)} plant types • Model: EfficientNet • Version 2.1
    </p>
</div>
""", unsafe_allow_html=True)

# Clear file data button for debugging
with st.sidebar:
    if st.button("Clear Uploaded Files"):
        st.session_state.uploaded_file_data = None
        st.session_state.uploaded_file_name = None
        st.session_state.feedback_submitted = False
        st.success("Uploaded files cleared!")
        st.rerun()
