# streamlit_app.py – FINAL CLEAN VERSION (EfficientNetB2 + head)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper
from io import BytesIO

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

# ----------------------- CONFIG --------------------------- #
MODEL_PATH = "plant_disease_final_model.keras"
CLASS_NAMES_PATH = "class_names_final.json"
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_MB = 200

# ----------------------- HELPERS --------------------------- #
def check_openai_setup():
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

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
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
    if not os.path.exists(CLASS_NAMES_PATH):
        st.sidebar.error(f"❌ Class names file not found: {CLASS_NAMES_PATH}")
        return None
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        st.sidebar.success(f"✅ Loaded {len(class_names)} plant classes")
        return class_names
    except Exception as e:
        st.sidebar.error(f"❌ Error loading class names: {e}")
        return None

def preprocess_image(image, img_size):
    """Resize + normalize exactly like training (x / 255.0)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    img = image.resize(img_size)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(image, model, class_names, img_size):
    try:
        batch = preprocess_image(image, img_size)
        preds = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        if idx >= len(class_names):
            return None, None, "Prediction index out of range", None
        predicted_class = class_names[idx]
        confidence = float(np.max(preds))
        return predicted_class, confidence, None, preds
    except Exception as e:
        return None, None, str(e), None

def get_plant_advice(plant_name, disease):
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        if "rate_limit" in str(e).lower() or "429" in str(e):
            return "AI service rate limit reached. Using standard care advice instead."
        return "AI advice currently unavailable. Using standard care advice instead."

def display_fallback_advice(plant_name, disease):
    formatted_disease = disease.replace("_", " ").replace("___", " - ").title()
    formatted_plant = plant_name.replace("_", " ").title()
    st.info(f"""
    **🌱 Recommended Treatment for {formatted_disease} on {formatted_plant}**

    - Remove affected leaves to slow spread
    - Avoid wetting leaves when watering
    - Improve ventilation and avoid overcrowding
    - Monitor daily and adjust sunlight and soil moisture
    """)

# ----------------------- LOAD RESOURCES -------------------- #
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

# Infer image size
if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = (224, 224)

# ----------------------- SIDEBAR DEBUG --------------------- #
with st.sidebar:
    st.header("🔧 System Info")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Classes loaded: {len(class_names) if class_names else 0}")
    st.write(f"OpenAI ready: {openai_ready}")
    st.write(f"Image size: {img_size}")
    if model is not None:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")

    if st.button("Clear Uploaded Image"):
        st.session_state.uploaded_file_data = None
        st.session_state.uploaded_file_name = None
        st.session_state.prediction_history = []
        st.success("Cleared!")
        st.rerun()

# ----------------------- MAIN HEADER ----------------------- #
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    'Upload a plant leaf photo for instant AI-powered diagnosis and care advice.'
    '</p>',
    unsafe_allow_html=True
)

if model is None or class_names is None:
    st.error("⚠️ App is not fully configured. Please ensure the model and class names files are present.")
    st.stop()

# ----------------------- MAIN LAYOUT ----------------------- #
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)} • Max {MAX_FILE_SIZE_MB}MB",
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.session_state.uploaded_file_data = uploaded_file.getvalue()
        st.session_state.uploaded_file_name = uploaded_file.name

    if st.session_state.uploaded_file_data is None:
        st.markdown(f"""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">
                JPG, PNG, JPEG • Max {MAX_FILE_SIZE_MB}MB
            </p>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.uploaded_file_data is not None:
        try:
            img_bytes = BytesIO(st.session_state.uploaded_file_data)
            image = Image.open(img_bytes)

            st.success("✅ File uploaded successfully!")
            st.write(f"**Filename:** {st.session_state.uploaded_file_name}")
            st.image(image, caption="📷 Your Plant Leaf", width=400)

            file_size_mb = len(st.session_state.uploaded_file_data) / (1024 * 1024)
            st.write(f"**Image Details:** {image.size[0]} × {image.size[1]} pixels • {file_size_mb:.1f} MB")

            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant..."):
                    # use a fresh copy for prediction
                    pred_img = Image.open(BytesIO(st.session_state.uploaded_file_data))
                    disease, confidence, error, raw_preds = predict_image(pred_img, model, class_names, img_size)

                if error:
                    st.error(f"❌ Analysis failed: {error}")
                else:
                    st.session_state.prediction_history.append(disease)
                    if len(st.session_state.prediction_history) > 10:
                        st.session_state.prediction_history.pop(0)

                    # ---- Diagnosis card ----
                    st.subheader("📋 Diagnosis Results")
                    nice_name = (
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
                            {nice_name}
                        </h3>
                        <div style="text-align: center;">
                            <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                            <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                                {confidence:.1%}
                            </h2>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Confidence warning
                    if confidence < 0.4:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Low Confidence Warning</h4>
                            <p>Try a clearer, well-lit image focusing on the affected area.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence < 0.75:
                        st.markdown("""
                        <div class="warning-box">
                            <h4>⚠️ Moderate Confidence</h4>
                            <p>You may want to upload more images or consult a plant expert.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.success("✅ High Confidence – prediction is likely reliable.")

                    # Debug info
                    with st.expander("🔧 Technical Details"):
                        st.markdown('<div class="debug-box"><h4>🛠️ Preprocessing & Prediction Debug</h4></div>',
                                    unsafe_allow_html=True)
                        batch = preprocess_image(image, img_size)
                        st.write("Preprocessing:")
                        st.write({
                            "original_size": image.size,
                            "array_shape": batch.shape[1:],
                            "array_min": float(batch.min()),
                            "array_max": float(batch.max()),
                            "array_mean": float(batch.mean())
                        })
                        if raw_preds is not None:
                            st.write("Prediction:")
                            st.write(f"Predicted class: {disease}")
                            st.write(f"Raw confidence: {confidence:.4f}")
                            st.write(f"Raw preds shape: {raw_preds.shape}")
                            st.write(f"Max / Min / Mean: {raw_preds.max():.6f} / {raw_preds.min():.6f} / {raw_preds.mean():.6f}")
                            st.write("Top 5 classes:")
                            top5 = np.argsort(raw_preds)[-5:][::-1]
                            for idx in top5:
                                st.write(f"{idx}: {class_names[idx]} - {raw_preds[idx]:.4f} ({raw_preds[idx]*100:.2f}%)")

                    # Care instructions
                    st.markdown("---")
                    st.subheader("💡 Care Instructions")
                    plant_name = disease.split("_")[0] if "_" in disease else "plant"

                    if openai_ready:
                        with st.spinner("🤖 Generating personalized care advice..."):
                            advice = get_plant_advice(plant_name, disease)
                        if any(x in advice for x in ["OpenAI", "API key", "rate limit"]):
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

    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #2E8B57, #228B22);
                color: white; border-radius: 10px; padding: 1.2rem;
                text-align: center; margin: 0.8rem 0;">
        <h3 style="margin: 0; font-size: 1.8rem;">{len(class_names)}</h3>
        <p style="margin: 0; opacity: 0.9; font-size: 0.9rem;">Plant Types Supported</p>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.prediction_history:
        st.subheader("📊 Recent Predictions")
        for pred in st.session_state.prediction_history[-5:]:
            st.write("•", pred.replace("_", " ").title())

    st.subheader("💡 Tips for Best Results")
    for tip in [
        "Use clear, well-lit photos",
        "Focus on the affected leaves",
        "Avoid blurry or dark images",
        "Use a simple, non-distracting background",
    ]:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 1rem;
                    margin: 0.6rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    border-left: 3px solid #3CB371;">
            <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
</div>
""", unsafe_allow_html=True)
