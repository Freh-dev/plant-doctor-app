# streamlit_app.py - FIXED VERSION WITH DEBUGGING
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper
from io import BytesIO

# 🆕 EfficientNet preprocessing (must match training)
from tensorflow.keras.applications.efficientnet import preprocess_input

# ----------------------- PAGE CONFIG ----------------------- #
st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------- SESSION STATE --------------------- #
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "uploaded_file_data" not in st.session_state:
    st.session_state.uploaded_file_data = None

if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None

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
    
    .debug-box {
        background: linear-gradient(135deg, #E8F4FD, #D1ECF1);
        border: 2px solid #17A2B8;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------- HELPERS --------------------------- #
def check_openai_setup():
    """Check if OpenAI advice helper is configured."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        test_advice = chatbot_helper.generate_advice("tomato", "healthy")
        if "OpenAI" not in test_advice and "API key" not in test_advice:
            return True
        return False
    except Exception:
        return False

# Updated paths - looking in current directory
MODEL_PATH = "plant_disease_final_model.keras"
CLASS_NAMES_PATH = "class_names_final.json"

@st.cache_resource
def load_model():
    """Load the trained Keras model from local file."""
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error("❌ Model file not found in current directory.")
        st.sidebar.write(f"Looking for: {os.path.abspath(MODEL_PATH)}")
        return None

    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ Model loaded successfully")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    """Load class names from json file with comprehensive validation."""
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        
        st.sidebar.success(f"✅ Loaded {len(class_names)} plant classes")
        
        # Validate class names
        if not class_names:
            st.sidebar.error("❌ Class names file is empty!")
            return ["Unknown_Class_0", "Unknown_Class_1"]
            
        return class_names
        
    except Exception as e:
        st.sidebar.error(f"❌ Error loading class names: {e}")
        
        # Create fallback classes based on common plant diseases
        fallback_classes = [
            "Tomato_Healthy",
            "Tomato_Early_Blight", 
            "Tomato_Late_Blight",
            "Tomato_Septoria_Leaf_Spot",
            "Tomato_Yellow_Leaf_Curl",
            "Tomato_Bacterial_Spot",
            "Tomato_Target_Spot",
            "Tomato_Mosaic_Virus",
            "Tomato_Leaf_Mold",
            "Tomato_Spider_Mites"
        ]
        st.sidebar.warning("⚠️ Using fallback class names")
        return fallback_classes

# 🆕 Enhanced preprocessing with debugging
def preprocess_image_for_model(image, img_size):
    """
    Convert PIL image → RGB → resize → EfficientNet preprocess → add batch dimension.
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image
        img = image.resize(img_size)
        img_array = np.array(img).astype("float32")
        
        # 🆕 DEBUG: Show image statistics
        debug_info = {
            "original_size": image.size,
            "resized_size": img.size,
            "array_shape": img_array.shape,
            "array_range": f"{np.min(img_array):.1f} to {np.max(img_array):.1f}",
            "array_mean": f"{np.mean(img_array):.1f}",
            "array_dtype": img_array.dtype
        }
        
        # IMPORTANT: use EfficientNet preprocess_input
        img_array = preprocess_input(img_array)
        
        debug_info["preprocessed_range"] = f"{np.min(img_array):.1f} to {np.max(img_array):.1f}"
        debug_info["preprocessed_mean"] = f"{np.mean(img_array):.1f}"
        
        img_array = np.expand_dims(img_array, axis=0)
        debug_info["final_shape"] = img_array.shape
        
        return img_array, debug_info
        
    except Exception as e:
        raise Exception(f"Image preprocessing failed: {e}")

def predict_image(image, model, class_names, img_size):
    """Run model prediction on a PIL.Image and return (class, confidence, error_msg)."""
    try:
        img_batch, debug_info = preprocess_image_for_model(image, img_size)
        prediction = model.predict(img_batch, verbose=0)[0]
        predicted_index = int(np.argmax(prediction))
        
        if predicted_index >= len(class_names):
            return None, None, f"Prediction index {predicted_index} out of range for {len(class_names)} classes"
            
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction))
        return predicted_class, confidence, None, debug_info, prediction
        
    except Exception as e:
        return None, None, str(e), None, None

def debug_model_predictions(image, model, class_names, img_size):
    """Show detailed predictions for debugging."""
    try:
        img_batch, debug_info = preprocess_image_for_model(image, img_size)
        prediction = model.predict(img_batch, verbose=0)[0]
        top_5_indices = np.argsort(prediction)[-5:][::-1]

        st.write("🔍 **Detailed Predictions:**")
        for rank, idx in enumerate(top_5_indices, start=1):
            if idx < len(class_names):
                confidence = prediction[idx]
                st.write(f"{rank}. **{class_names[idx]}** - {confidence:.4f} ({confidence*100:.2f}%)")
                
                # Show confidence bar
                progress = int(confidence * 100)
                st.progress(progress, text=f"{progress}%")
            else:
                st.write(f"{rank}. [INDEX {idx} OUT OF RANGE] - {prediction[idx]:.4f}")
                
        return prediction
    except Exception as e:
        st.error(f"Debug prediction failed: {e}")
        return None

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

# ----------------------- LOAD RESOURCES -------------------- #
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

# Automatically infer image size from the model if possible
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = (224, 224)  # EfficientNet fallback

# ----------------------- SIDEBAR DEBUG --------------------- #
with st.sidebar:
    st.header("🔧 System Configuration")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Number of classes: {len(class_names)}")
    st.write(f"OpenAI ready: {openai_ready}")
    st.write(f"Image size: {img_size}")
    
    if model is not None:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
    
    if class_names:
        st.write("First 8 classes:")
        for i, cls in enumerate(class_names[:8]):
            st.write(f"  {i}: {cls}")
    
    # Model testing section
    st.markdown("---")
    st.header("🧪 Model Testing")
    
    if st.button("Test with Sample Images"):
        st.info("Testing model with basic patterns...")
        
        test_images = {
            "Green (healthy)": Image.new('RGB', img_size, (100, 200, 100)),
            "Brown (diseased)": Image.new('RGB', img_size, (150, 100, 50)),
            "Yellow (deficiency)": Image.new('RGB', img_size, (250, 250, 100)),
        }
        
        for name, test_img in test_images.items():
            with st.expander(f"Test: {name}"):
                disease, confidence, error, debug_info, raw_pred = predict_image(test_img, model, class_names, img_size)
                if error:
                    st.error(f"❌ {error}")
                else:
                    st.write(f"**Prediction:** {disease}")
                    st.write(f"**Confidence:** {confidence:.1%}")
                    if raw_pred is not None:
                        st.write(f"**Raw max:** {np.max(raw_pred):.4f}")
    
    # Clear data button
    if st.button("Clear All Data"):
        st.session_state.uploaded_file_data = None
        st.session_state.uploaded_file_name = None
        st.session_state.prediction_history = []
        st.success("All data cleared!")
        st.rerun()

# ----------------------- MAIN HEADER ----------------------- #
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    'Upload a plant leaf photo for instant AI-powered diagnosis and care advice.'
    '</p>',
    unsafe_allow_html=True
)

# If model is missing, show error and stop
if model is None:
    st.error(f"""
    ## 🔧 Service Temporarily Unavailable
    
    The model file **{MODEL_PATH}** is not available in the current directory.
    
    Please make sure:
    - The file exists in the same directory as `streamlit_app.py`
    - The filename is exactly: `{MODEL_PATH}`
    - The file matches the `{CLASS_NAMES_PATH}` label order
    
    **Current directory:** {os.getcwd()}
    **Looking for:** {os.path.abspath(MODEL_PATH)}
    """)
    st.stop()

# ----------------------- LAYOUT ---------------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a plant leaf image**")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG • Max 200MB",
        label_visibility="collapsed"
    )

    # Store file data in session state immediately
    if uploaded_file is not None:
        st.session_state.uploaded_file_data = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name

    # Nice empty state
    if uploaded_file is None and st.session_state.uploaded_file_data is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
        </div>
        """, unsafe_allow_html=True)

    # Process uploaded file from session state data
    if st.session_state.uploaded_file_data is not None:
        try:
            # Create a fresh image object from stored data for display
            image_data = BytesIO(st.session_state.uploaded_file_data)
            display_image = Image.open(image_data)

            st.success("✅ **File uploaded successfully!**")
            st.write(f"**Filename:** {st.session_state.uploaded_file_name}")

            # Preview
            st.image(display_image, caption="📷 Your Plant Leaf", width=400)

            # File info
            file_size_mb = len(st.session_state.uploaded_file_data) / (1024 * 1024)
            st.write(
                f"**Image Details:** {display_image.size[0]} × {display_image.size[1]} pixels • {file_size_mb:.1f} MB"
            )

            # Analyze button
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant..."):
                    # Create a FRESH image object for prediction
                    prediction_image_data = BytesIO(st.session_state.uploaded_file_data)
                    prediction_image = Image.open(prediction_image_data)
                    
                    disease, confidence, error, debug_info, raw_prediction = predict_image(
                        prediction_image, model, class_names, img_size
                    )

                if error:
                    st.error(f"""
                    ## ❌ Analysis Failed
                    **Error:** {error}

                    Please try a different image or check the model configuration.
                    """)
                else:
                    # Add to prediction history
                    st.session_state.prediction_history.append(disease)
                    if len(st.session_state.prediction_history) > 10:
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
                            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                            <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                {confidence:.1%}
                            </h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ----------------- CONFIDENCE WARNINGS ------------- #
                    if confidence < 0.4:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Low Confidence Warning</h4>
                            <p>The model is not very confident about this diagnosis. This may be due to:</p>
                            <ul>
                                <li>Poor image quality</li>
                                <li>Unusual angle or lighting</li>
                                <li>Plant type underrepresented in training data</li>
                                <li>Multiple diseases present</li>
                            </ul>
                            <p><strong>Recommendation:</strong> Try a clearer, well-lit image focusing on the leaf.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence < 0.75:
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
                        st.success("**✅ High Confidence** – Diagnosis is likely reliable.")

                    # ----------------- DEBUG INFORMATION ---------------- #
                    with st.expander("🔧 Technical Details"):
                        st.markdown("""
                        <div class="debug-box">
                            <h4>🛠️ Model & Preprocessing Info</h4>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.write("**Preprocessing Details:**")
                        if debug_info:
                            for key, value in debug_info.items():
                                st.write(f"- {key}: {value}")
                        
                        st.write("**Prediction Details:**")
                        st.write(f"- Predicted class: {disease}")
                        st.write(f"- Raw confidence: {confidence:.4f}")
                        st.write(f"- Model input size: {img_size}")
                        st.write(f"- Available classes: {len(class_names)}")
                        
                        if raw_prediction is not None:
                            st.write("**Raw Prediction Output:**")
                            st.write(f"- Shape: {raw_prediction.shape}")
                            st.write(f"- Max value: {np.max(raw_prediction):.6f}")
                            st.write(f"- Min value: {np.min(raw_prediction):.6f}")
                            st.write(f"- Mean value: {np.mean(raw_prediction):.6f}")
                            
                            # Show top predictions
                            st.write("**Top 5 Predictions:**")
                            top_5_indices = np.argsort(raw_prediction)[-5:][::-1]
                            for rank, idx in enumerate(top_5_indices, start=1):
                                if idx < len(class_names):
                                    conf = raw_prediction[idx]
                                    st.write(f"{rank}. {class_names[idx]} - {conf:.6f} ({conf*100:.2f}%)")
                                else:
                                    st.write(f"{rank}. [INDEX {idx} OUT OF RANGE] - {raw_prediction[idx]:.6f}")

                    # ----------------- USER FEEDBACK ------------------- #
                    st.markdown("---")
                    st.subheader("🤔 Prediction Accuracy")
                    feedback = st.radio(
                        "Does this prediction seem correct?",
                        ["Yes, looks accurate", "No, this seems wrong", "Unsure"],
                        index=0
                    )
                    if feedback == "No, this seems wrong":
                        st.warning(
                            "Thank you for your feedback! This helps us improve the model accuracy."
                        )

                    # ----------------- CARE INSTRUCTIONS --------------- #
                    st.markdown("---")
                    st.subheader("💡 Care Instructions")

                    plant_name = disease.split("_")[0] if "_" in disease else "plant"

                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = get_plant_advice(plant_name, disease)

                        if any(key in advice for key in ["OpenAI", "API key", "rate limit"]):
                            st.warning("⚠️ Using fallback care advice (AI service issue).")
                            display_fallback_advice(plant_name, disease)
                        else:
                            st.success("✅ AI-Generated Personalized Advice")
                            st.info(advice)
                    else:
                        st.warning("⚠️ Using standard care advice (AI not configured).")
                        display_fallback_advice(plant_name, disease)

        except Exception as e:
            st.error(f"❌ Error processing image: {e}")

with col2:
    # Sidebar / status info
    st.subheader("System Status")
    st.write("Real-time service monitoring")

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
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Personalized care tips enabled</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="status-card">
            <h4 style="color: #FFA500; margin-bottom: 0.3rem;">⚠️ Basic Advice Mode</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Standard care tips only</p>
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

    # Recent predictions
    if st.session_state.prediction_history:
        st.subheader("📊 Recent Predictions")
        unique_predictions = list(dict.fromkeys(st.session_state.prediction_history[-5:]))
        for pred in unique_predictions:
            formatted_pred = pred.replace("_", " ").title()
            st.write(f"• {formatted_pred}")

    # Tips
    st.subheader("💡 Tips for Best Results")
    tips = [
        "Use clear, well-lit photos",
        "Focus on the affected leaves",
        "Include a plain, non-distracting background",
        "Take multiple angles if you're unsure",
        "Monitor your plant regularly for changes"
    ]
    for tip in tips:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem;
                    margin: 0.6rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    border-left: 3px solid #3CB371;">
            <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
        </div>
        """, unsafe_allow_html=True)

# ----------------------- FOOTER ---------------------------- #
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
