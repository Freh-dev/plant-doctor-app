# streamlit_app.py - CLEAN FINAL VERSION (MATCHES TRAINING PIPELINE)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os
from io import BytesIO
import chatbot_helper

# ----------------------- CONSTANTS ----------------------- #

MODEL_PATH = "plant_disease_final_model.keras"
CLASS_NAMES_PATH = "class_names_final.json"
SUPPORTED_FORMATS = ["jpg", "jpeg", "png"]
MAX_FILE_SIZE_MB = 200
DEFAULT_IMG_SIZE = (224, 224)  # must match training

# ----------------------- PAGE CONFIG --------------------- #

st.set_page_config(
    page_title="Plant Doctor 🌿",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------- SESSION STATE ------------------- #

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes = None

if "uploaded_name" not in st.session_state:
    st.session_state.uploaded_name = None

# ----------------------- STYLING ------------------------- #

st.markdown(
    """
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
""",
    unsafe_allow_html=True,
)

# ----------------------- HELPERS ------------------------- #


def check_openai_setup() -> bool:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return False
    try:
        txt = chatbot_helper.generate_advice("tomato", "healthy")
        if "OpenAI" in txt or "API key" in txt:
            return False
        return True
    except Exception:
        return False


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"❌ Model file not found: {os.path.abspath(MODEL_PATH)}")
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
        st.sidebar.success(f"✅ Loaded {len(class_names)} classes from JSON")
        return class_names
    except Exception as e:
        # IMPORTANT: no tomato-only fallback in production
        st.sidebar.error(f"❌ Error loading {CLASS_NAMES_PATH}: {e}")
        return []


def preprocess_image_for_model(image: Image.Image, img_size):
    """
    EXACTLY match training pipeline:
      - RGB
      - resize to img_size
      - cast to float32
      - divide by 255.0  (values in [0,1])
    """
    # make sure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(img_size)
    arr = np.array(image).astype("float32") / 255.0  # <--- CRITICAL
    debug_info = {
        "original_size": image.size,
        "array_shape": arr.shape,
        "array_min": float(arr.min()),
        "array_max": float(arr.max()),
        "array_mean": float(arr.mean()),
    }
    arr = np.expand_dims(arr, axis=0)
    return arr, debug_info


def predict_image(image: Image.Image, model, class_names, img_size):
    try:
        batch, dbg = preprocess_image_for_model(image, img_size)
        preds = model.predict(batch, verbose=0)[0]
        idx = int(np.argmax(preds))
        if idx >= len(class_names):
            return None, None, "Prediction index out of range", dbg, preds
        return class_names[idx], float(preds[idx]), None, dbg, preds
    except Exception as e:
        return None, None, str(e), None, None


def get_plant_advice(plant_name, disease):
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        if "429" in str(e) or "rate limit" in str(e).lower():
            return "AI service rate limit reached. Using standard care advice instead."
        return "AI advice currently unavailable. Using standard care advice instead."


def display_fallback_advice(plant_name, disease):
    formatted_d = disease.replace("_", " ").replace("___", " - ").title()
    formatted_p = plant_name.replace("_", " ").title()
    st.info(
        f"""
**🌱 Recommended Treatment for {formatted_d} on {formatted_p}**

- Remove heavily affected leaves
- Avoid overhead watering; keep leaves dry
- Improve air circulation around plants
- Use appropriate fungicide / treatment if available
- Monitor plant daily for improvement or spread
"""
    )


# ----------------------- LOAD RESOURCES ------------------- #

model = load_model()
class_names = load_class_names()
openai_ready = check_openai_setup()

if model is not None and hasattr(model, "input_shape") and len(model.input_shape) == 4:
    img_size = (model.input_shape[1], model.input_shape[2])
else:
    img_size = DEFAULT_IMG_SIZE

# ----------------------- SIDEBAR DEBUG -------------------- #

with st.sidebar:
    st.header("🔧 Debug Info")
    st.write(f"Model loaded: {model is not None}")
    st.write(f"Classes loaded: {len(class_names)}")
    st.write(f"OpenAI ready: {openai_ready}")
    st.write(f"Model input shape: {getattr(model, 'input_shape', 'N/A')}")
    if class_names:
        st.write("First 5 classes:")
        for i, c in enumerate(class_names[:5]):
            st.write(f"{i}: {c}")

    if st.button("Clear uploaded image"):
        st.session_state.uploaded_bytes = None
        st.session_state.uploaded_name = None
        st.session_state.prediction_history = []
        st.success("Cleared stored image and history")
        st.rerun()

# If core resources missing, stop early
if model is None or not class_names:
    st.error(
        f"""
## 🔧 Service Temporarily Unavailable

Model or class names are not correctly loaded.

- Model loaded: **{model is not None}**
- Classes loaded: **{len(class_names)}**

Please ensure **{MODEL_PATH}** and **{CLASS_NAMES_PATH}** are present
in the same folder as `streamlit_app.py`.
"""
    )
    st.stop()

# ----------------------- MAIN HEADER ---------------------- #

st.markdown('<h1 class="main-header">🌿 Plant Doctor</h1>', unsafe_allow_html=True)
st.markdown(
    '<p style="text-align: center; color: #666; margin-bottom: 2rem;">'
    "Upload a plant leaf photo for instant AI-powered diagnosis and care advice."
    "</p>",
    unsafe_allow_html=True,
)

# ----------------------- LAYOUT --------------------------- #

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📸 Upload Plant Image")
    st.write("**Choose a plant leaf image**")

    uploaded_file = st.file_uploader(
        "Drag and drop your file here or click to browse",
        type=SUPPORTED_FORMATS,
        help=f"Supported formats: {', '.join(SUPPORTED_FORMATS)} • Max {MAX_FILE_SIZE_MB}MB",
        label_visibility="collapsed",
    )

    # Store bytes in session_state so we can re-open safely
    if uploaded_file is not None:
        raw = uploaded_file.read()
        st.session_state.uploaded_bytes = raw
        st.session_state.uploaded_name = uploaded_file.name

    if st.session_state.uploaded_bytes is None:
        st.markdown(
            f"""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
            <p style="color: #888; font-size: 0.9rem; margin: 0;">
                JPG, PNG, JPEG • Max {MAX_FILE_SIZE_MB}MB
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    if st.session_state.uploaded_bytes is not None:
        # Display image
        img_bytes = BytesIO(st.session_state.uploaded_bytes)
        img = Image.open(img_bytes)
        st.success("✅ **File uploaded successfully!**")
        st.write(f"**Filename:** {st.session_state.uploaded_name}")
        st.image(img, caption="📷 Your Plant Leaf", width=400)

        file_size_mb = len(st.session_state.uploaded_bytes) / (1024 * 1024)
        st.write(
            f"**Image Details:** {img.size[0]} × {img.size[1]} pixels • {file_size_mb:.1f} MB"
        )

        if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
            with st.spinner("🔬 Analyzing your plant..."):
                pred_img = Image.open(BytesIO(st.session_state.uploaded_bytes))
                disease, conf, err, dbg, raw_pred = predict_image(
                    pred_img, model, class_names, img_size
                )

            if err:
                st.error(
                    f"""
                ## ❌ Analysis Failed
                **Error:** {err}
                """
                )
            else:
                st.session_state.prediction_history.append(disease)
                if len(st.session_state.prediction_history) > 10:
                    st.session_state.prediction_history.pop(0)

                # Diagnosis card
                st.subheader("📋 Diagnosis Results")

                formatted = (
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

                st.markdown(
                    f"""
                <div class="diagnosis-card">
                    <div style="text-align: center; margin-bottom: 1.2rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.8rem;">{status_emoji}</div>
                        <span style="background: {status_color}; color: white; padding: 0.4rem 0.8rem;
                                     border-radius: 15px; font-weight: 600;">
                            {status_text}
                        </span>
                    </div>
                    <h3 style="color: {status_color}; text-align: center; margin-bottom: 0.8rem;">
                        {formatted}
                    </h3>
                    <div style="text-align: center;">
                        <p style="font-size: 1.1rem; color: #666; margin-bottom: 0.5rem;">Confidence Level</p>
                        <h2 style="color: {status_color}; font-size: 2rem; margin: 0.3rem 0;">
                            {conf:.1%}
                        </h2>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

                # Confidence hints
                if conf < 0.4:
                    st.markdown(
                        """
                    <div class="warning-box">
                        <h4>⚠️ Low Confidence</h4>
                        <p>The model is not very sure about this leaf. Try a clearer, well-lit close-up of the leaf.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                elif conf < 0.75:
                    st.markdown(
                        """
                    <div class="warning-box">
                        <h4>⚠️ Moderate Confidence</h4>
                        <p>You may want to upload more images or consult a plant expert to confirm.</p>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )

                # Technical debug
                with st.expander("🔧 Technical Details"):
                    st.markdown(
                        """
                    <div class="debug-box">
                        <h4>🛠️ Preprocessing & Prediction Debug</h4>
                    </div>
                    """,
                        unsafe_allow_html=True,
                    )
                    if dbg:
                        st.write("**Preprocessing:**")
                        for k, v in dbg.items():
                            st.write(f"- {k}: {v}")
                    st.write("**Prediction:**")
                    st.write(f"- Predicted class: {disease}")
                    st.write(f"- Raw confidence: {conf:.4f}")
                    if raw_pred is not None:
                        st.write(f"- Raw preds shape: {raw_pred.shape}")
                        st.write(
                            f"- Max / Min / Mean: "
                            f"{raw_pred.max():.6f} / {raw_pred.min():.6f} / {raw_pred.mean():.6f}"
                        )
                        top5 = np.argsort(raw_pred)[-5:][::-1]
                        st.write("**Top 5 classes:**")
                        for i_idx in top5:
                            st.write(
                                f"{i_idx}: {class_names[i_idx]} "
                                f"- {raw_pred[i_idx]:.4f} ({raw_pred[i_idx]*100:.2f}%)"
                            )

                # Feedback + care
                st.markdown("---")
                st.subheader("🤔 Prediction Accuracy")
                fb = st.radio(
                    "Does this prediction seem correct?",
                    ["Yes, looks accurate", "No, seems wrong", "Unsure"],
                    index=0,
                )
                if fb == "No, seems wrong":
                    st.warning(
                        "Thanks for the feedback. It helps identify where the model is weaker."
                    )

                st.markdown("---")
                st.subheader("💡 Care Instructions")

                plant_name = disease.split("_")[0] if "_" in disease else "plant"
                if openai_ready:
                    with st.spinner("🤖 Generating personalized care advice..."):
                        advice = get_plant_advice(plant_name, disease)
                    if any(
                        key in advice
                        for key in ["OpenAI", "API key", "rate limit", "unavailable"]
                    ):
                        st.warning("⚠️ Using fallback advice.")
                        display_fallback_advice(plant_name, disease)
                    else:
                        st.success("✅ AI-Generated Advice")
                        st.info(advice)
                else:
                    st.warning("⚠️ Using standard care advice (AI not configured).")
                    display_fallback_advice(plant_name, disease)

with col2:
    st.subheader("System Status")
    st.write("Real-time service overview")

    def status_card(title, desc, color):
        st.markdown(
            f"""
        <div class="status-card">
            <h4 style="color: {color}; margin-bottom: 0.3rem;">{title}</h4>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">{desc}</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    status_card("✅ Model Active", "Plant diagnosis ready", "#2E8B57")
    if openai_ready:
        status_card("✅ AI Advice Ready", "Personalized care tips enabled", "#2E8B57")
    else:
        status_card("⚠️ Basic Advice Mode", "Standard care tips only", "#FFA500")

    status_card(
        f"✅ Classes Loaded: {len(class_names)}",
        "Class order matches training labels",
        "#2E8B57",
    )

    if st.session_state.prediction_history:
        st.subheader("📊 Recent Predictions")
        for p in st.session_state.prediction_history[-5:]:
            st.write("• " + p.replace("_", " ").title())

    st.subheader("💡 Tips for Best Results")
    for tip in [
        "Use clear, bright, non-blurry images",
        "Focus on a single leaf if possible",
        "Avoid very dark or noisy backgrounds",
        "Take multiple photos if you’re unsure",
    ]:
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

st.markdown("---")
st.markdown(
    """
<div style="text-align: center; color: #666; padding: 1.5rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • EfficientNetB2 backbone • 38 classes
    </p>
</div>
""",
    unsafe_allow_html=True,
)
