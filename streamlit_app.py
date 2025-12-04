# streamlit_app.py – robust version with auto-detected preprocessing
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
import chatbot_helper

# ---------------- CONFIG ----------------
MODEL_PATH = "plant_disease_final_model.keras"
CLASS_NAMES_PATH = "class_names_final.json"
IMG_SIZE = 224
SUPPORTED_TYPES = ["jpg", "jpeg", "png"]

st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------- SESSION STATE ------------
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

# ------------- STYLES -------------------
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
    .warning-box {
        background: #FFF3CD;
        border: 1px solid #FFA500;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ------------- HELPERS ------------------
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
        return None
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        st.sidebar.success("✅ Model loaded")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = json.load(f)
        st.sidebar.info(f"✅ Loaded {len(class_names)} classes")
        return class_names
    except Exception as e:
        st.sidebar.error(f"❌ Error loading class names: {e}")
        return []

def model_expects_raw_0_255(model: tf.keras.Model) -> bool:
    """
    Detect if the model already has preprocessing near the input:
    - Rescaling(1./255) layer
    - OR a Lambda layer that wraps EfficientNet preprocess_input
      (we named it 'eff_preprocess' in training).
    If yes → we should feed raw 0–255.
    If no  → we should scale to 0–1 in Streamlit.
    """
    try:
        for layer in model.layers[:6]:
            # 1) Rescaling(1./255)
            if isinstance(layer, tf.keras.layers.Rescaling):
                scale = getattr(layer, "scale", None)
                if scale is not None and abs(scale - 1.0/255.0) < 1e-6:
                    return True

            # 2) Lambda(preprocess_input, name="eff_preprocess")
            if isinstance(layer, tf.keras.layers.Lambda):
                # Name check (we set name="eff_preprocess" in the notebook)
                if layer.name == "eff_preprocess":
                    return True
                # Fallback: try to see if function name looks like preprocess_input
                fn = getattr(layer, "function", None)
                if fn is not None and getattr(fn, "__name__", "") == "preprocess_input":
                    return True

        return False
    except Exception:
        return False

def preprocess_image(image: Image.Image, model: tf.keras.Model) -> np.ndarray:
    """
    Resize to 224x224 RGB.
    If model has internal preprocessing (Rescaling or Lambda(preprocess_input)):
        → DO NOT divide by 255 here (keep 0–255).
    Otherwise:
        → scale to 0–1 here.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))

    expects_raw = model_expects_raw_0_255(model)
    arr = np.array(image).astype("float32")

    if expects_raw:
        # Model will do /255 or preprocess_input inside
        scale_info = "raw 0–255 → model has internal preprocessing (Rescaling/Lambda)"
    else:
        # We need to scale here
        arr = arr / 255.0
        scale_info = "scaled 0–1 in Streamlit (no internal preprocessing detected)"

    arr = np.expand_dims(arr, axis=0)

    # Debug line so you can SEE what is happening
    st.write(
        "🛠️ Preprocessing debug:",
        f"shape={arr.shape}, min={arr.min():.3f}, max={arr.max():.3f}, "
        f"mean={arr.mean():.3f}, mode={scale_info}"
    )
    return arr

def predict_image(image, model, class_names):
    arr = preprocess_image(image, model)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    if idx >= len(class_names):
        return None, None, preds
    return class_names[idx], float(preds[idx]), preds

def get_plant_advice(plant_name, disease):
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception:
        return "AI advice currently unavailable. Please follow standard plant care practices."

def display_fallback_advice(plant_name, disease):
    formatted_disease = disease.replace("_", " ").replace("___", " - ").title()
    st.info(f"""
**🌱 Basic Care Tips for {formatted_disease}**

- Remove visibly infected leaves
- Avoid overhead watering (water soil, not leaves)
- Improve air circulation around the plant
- Disinfect tools after pruning
- Monitor the plant over the next 7–10 days
""")

# ------------- LOAD RESOURCES -----------
model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

# ------------- SIDEBAR ------------------
with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Number of classes: {len(class_names)}")
    st.write(f"OpenAI ready: {openai_ready}")
    st.write(f"Model path: {os.path.abspath(MODEL_PATH)}")
    st.write(f"Class names path: {os.path.abspath(CLASS_NAMES_PATH)}")
    if model is not None:
        st.write(f"Model input shape: {model.input_shape}")
        st.write(f"Model output shape: {model.output_shape}")
        st.write(f"Has internal preprocessing (Rescaling/Lambda): {model_expects_raw_0_255(model)}")

# stop if no model
if model is None or not class_names:
    st.error("Model or class names missing. Please upload both to the app folder.")
    st.stop()

# ------------- MAIN HEADER --------------
st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    'Upload a plant leaf photo for AI-based disease prediction and care advice.'
    '</p>',
    unsafe_allow_html=True
)

# ------------- LAYOUT -------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=SUPPORTED_TYPES,
        help=f"Supported formats: {', '.join(SUPPORTED_TYPES)} • Max 200MB",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">JPG, PNG, JPEG • Max 200MB</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.success("✅ File uploaded!")
        st.write(f"**Filename:** {uploaded_file.name}")
        st.image(image, caption="📷 Your Plant Leaf", width=400)

        if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
            with st.spinner("🔬 Analyzing your plant..."):
                disease, confidence, preds = predict_image(image, model, class_names)

            if disease is None:
                st.error("Prediction index out of range. Check class_names_final.json.")
            else:
                st.session_state.prediction_history.append(disease)
                formatted = (disease.replace('___', ' - ')
                                   .replace('__', ' - ')
                                   .replace('_', ' '))

                # Diagnosis card
                st.subheader("📋 Diagnosis Results")
                healthy = "healthy" in disease.lower()
                emoji = "✅" if healthy else "⚠️"
                status = "Healthy Plant" if healthy else "Needs Attention"
                color = "#2E8B57" if healthy else "#FFA500"

                st.markdown(f"""
                <div class="diagnosis-card">
                    <div style="text-align: center; margin-bottom: 1.2rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{emoji}</div>
                        <span style="background: {color}; color: white; padding: 0.4rem 0.8rem;
                                     border-radius: 15px; font-weight: 600;">
                            {status}
                        </span>
                    </div>
                    <h3 style="color: {color}; text-align: center; margin-bottom: 0.8rem;">
                        {formatted}
                    </h3>
                    <div style="text-align: center;">
                        <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence</p>
                        <h2 style="color: {color}; font-size: 2rem; margin: 0.3rem 0;">
                            {confidence:.1%}
                        </h2>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Confidence warning
                if confidence < 0.4:
                    st.markdown("""
                    <div class="warning-box">
                        <h4>⚠️ Low Confidence</h4>
                        <p>The model is not very confident. Try:</p>
                        <ul>
                            <li>Using a clearer, well-lit photo</li>
                            <li>Focusing on a single leaf</li>
                            <li>Uploading multiple images</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)

                # Debug expander
                with st.expander("🔧 Prediction Debug Info"):
                    st.write("Predicted class:", disease)
                    st.write("Raw confidence:", confidence)
                    if preds is not None:
                        top5 = np.argsort(preds)[-5:][::-1]
                        st.write("Top 5 predictions:")
                        for i in top5:
                            st.write(f"- {class_names[i]}: {preds[i]:.3f} ({preds[i]*100:.1f}%)")

                # Care advice
                st.markdown("---")
                st.subheader("💡 Care Instructions")
                plant_name = disease.split("_")[0] if "_" in disease else "plant"

                if openai_ready:
                    with st.spinner("🤖 Generating AI care advice..."):
                        advice = get_plant_advice(plant_name, disease)
                    st.info(advice)
                else:
                    display_fallback_advice(plant_name, disease)

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

    # Plant types metric
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
            st.write("•", pred.replace("_", " ").replace("___", " - ").title())

    st.subheader("💡 Tips for Best Results")
    tips = [
        "Use clear, well-lit photos",
        "Focus on the leaf, not the whole plant",
        "Avoid very dark or blurry images",
        "Try multiple leaves if symptoms differ",
    ]
    for tip in tips:
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
