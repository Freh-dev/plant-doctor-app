# streamlit_app.py – fixed version for Lambda layer issue
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
    """Load model with proper handling of Lambda layer"""
    if not os.path.exists(MODEL_PATH):
        st.sidebar.error(f"❌ Model file not found: {MODEL_PATH}")
        st.sidebar.write(f"Looking for: {os.path.abspath(MODEL_PATH)}")
        return None
    
    try:
        # Define the preprocessing function that the Lambda layer uses
        # This MUST be defined here for proper serialization
        from tensorflow.keras.applications.efficientnet import preprocess_input
        
        # Register the function with keras
        tf.keras.utils.get_custom_objects()['preprocess_input'] = preprocess_input
        
        # Try loading with custom objects
        try:
            model = tf.keras.models.load_model(
                MODEL_PATH,
                custom_objects={'preprocess_input': preprocess_input},
                compile=False
            )
            st.sidebar.success("✅ Model loaded successfully")
            return model
        except Exception as e1:
            st.sidebar.warning(f"First attempt failed: {str(e1)[:150]}")
            
            # Try alternative: load as weights only and rebuild architecture
            try:
                import keras
                # Create custom Lambda layer that can be serialized
                @keras.saving.register_keras_serializable()
                class EfficientNetPreprocessor(keras.layers.Layer):
                    def __init__(self, **kwargs):
                        super().__init__(**kwargs)
                        self.preprocess_fn = preprocess_input
                    
                    def call(self, inputs):
                        return self.preprocess_fn(inputs)
                    
                    def get_config(self):
                        config = super().get_config()
                        return config
                
                # Load with the custom layer
                model = tf.keras.models.load_model(
                    MODEL_PATH,
                    custom_objects={'eff_preprocess': EfficientNetPreprocessor},
                    compile=False
                )
                st.sidebar.success("✅ Model loaded with custom layer")
                return model
            except Exception as e2:
                st.sidebar.error(f"All loading attempts failed: {str(e2)[:150]}")
                return None
                
    except Exception as e:
        st.sidebar.error(f"❌ Critical error loading model: {e}")
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

def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for EfficientNet model
    We apply preprocessing here instead of in the model
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    
    arr = np.array(image).astype("float32")
    
    # Apply EfficientNet preprocessing manually
    # This replaces the Lambda layer in the model
    from tensorflow.keras.applications.efficientnet import preprocess_input
    arr = preprocess_input(arr)
    
    arr = np.expand_dims(arr, axis=0)
    return arr

def predict_image(image, model, class_names):
    """Predict with manual preprocessing"""
    arr = preprocess_image(image)
    try:
        preds = model.predict(arr, verbose=0)[0]
        idx = int(np.argmax(preds))
        if idx >= len(class_names):
            return None, None, preds
        return class_names[idx], float(preds[idx]), preds
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None, None, None

def get_plant_advice(plant_name, disease):
    try:
        return chatbot_helper.generate_advice(plant_name, disease)
    except Exception as e:
        st.warning(f"AI advice unavailable: {e}")
        return None

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

# Stop if critical resources missing
if model is None:
    st.error("""
    ## ❌ Model Failed to Load
    
    The model file exists but cannot be loaded due to serialization issues.
    
    **Quick Fix Options:**
    1. **Re-save the model** from your notebook without the Lambda layer
    2. **Use this workaround:** Update the model loading code to handle the Lambda layer
    
    **Temporary Workaround:**
    ```python
    # In your training notebook, replace Lambda layer with:
    # x = keras.layers.Rescaling(1./255)(input_layer)  # Instead of preprocess_input
    # Then re-save the model
    ```
    """)
    st.stop()

if not class_names:
    st.error("Class names could not be loaded. Please check class_names_final.json")
    st.stop()

# ------------- SIDEBAR ------------------
with st.sidebar:
    st.header("🔧 System Status")
    
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        st.metric("Model Status", "✅ Active" if model else "❌ Failed")
    with status_col2:
        st.metric("Classes", len(class_names))
    
    st.divider()
    
    st.subheader("Debug Info")
    st.write(f"Model: {os.path.basename(MODEL_PATH)}")
    st.write(f"Input shape: {model.input_shape}")
    st.write(f"Output shape: {model.output_shape}")
    st.write(f"OpenAI: {'✅ Ready' if openai_ready else '⚠️ Basic Mode'}")
    
    if st.checkbox("Show Model Layers"):
        for i, layer in enumerate(model.layers[:5]):  # Show first 5 layers
            st.write(f"{i}: {layer.name} - {layer.__class__.__name__}")

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
        help=f"Supported formats: {', '.join(SUPPORTED_TYPES)}",
        label_visibility="collapsed",
    )

    if uploaded_file is None:
        st.markdown("""
        <div class="upload-area">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🌿</div>
            <h3 style="color: #2E8B57; margin-bottom: 0.5rem;">Drag & Drop Your Plant Leaf Here</h3>
            <p style="color: #666; margin-bottom: 0.5rem;">or click the area above to browse files</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            st.success("✅ File uploaded!")
            st.image(image, caption="📷 Your Plant Leaf", width=350)
            
            # Show image info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"**Size:** {image.size[0]}×{image.size[1]}")
            with col_info2:
                st.write(f"**Mode:** {image.mode}")
            
            if st.button("🔍 Analyze Plant Health", type="primary", use_container_width=True):
                with st.spinner("🔬 Analyzing your plant..."):
                    disease, confidence, preds = predict_image(image, model, class_names)
                
                if disease is None or confidence is None:
                    st.error("Failed to make prediction. Please try another image.")
                else:
                    # Store in history
                    st.session_state.prediction_history.append({
                        'disease': disease,
                        'confidence': confidence,
                        'timestamp': st.session_state.get('timestamp', 'now')
                    })
                    
                    # Format disease name
                    formatted = disease.replace('___', ' - ').replace('__', ' - ').replace('_', ' ')
                    
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
                        st.warning("""
                        ⚠️ **Low Confidence Warning**
                        The model is not very confident in this prediction. For best results:
                        - Use a clear, well-lit photo
                        - Focus on a single leaf
                        - Avoid blurry or dark images
                        - Try multiple angles
                        """)
                    
                    # Top predictions
                    if preds is not None:
                        with st.expander("🔍 See Top Predictions"):
                            top_indices = np.argsort(preds)[-5:][::-1]
                            for i, idx in enumerate(top_indices):
                                prob = preds[idx]
                                bar_width = int(prob * 200)
                                st.markdown(f"""
                                **{i+1}. {class_names[idx].replace('_', ' ')}**
                                <div style="background: #e0e0e0; width: 200px; height: 10px; border-radius: 5px; margin: 2px 0 10px 0;">
                                    <div style="background: {'#2E8B57' if i==0 else '#4CAF50'}; width: {bar_width}px; height: 10px; border-radius: 5px;"></div>
                                </div>
                                {prob:.1%}
                                """, unsafe_allow_html=True)
                    
                    # Care advice
                    st.markdown("---")
                    st.subheader("💡 Care Instructions")
                    
                    # Extract plant name
                    if "___" in disease:
                        plant_name = disease.split("___")[0]
                    elif "__" in disease:
                        plant_name = disease.split("__")[0]
                    else:
                        plant_name = "Plant"
                    
                    if openai_ready:
                        with st.spinner("🤖 Generating AI care advice..."):
                            advice = get_plant_advice(plant_name, disease)
                        if advice:
                            st.success(advice)
                        else:
                            display_fallback_advice(plant_name, disease)
                    else:
                        display_fallback_advice(plant_name, disease)
                        
        except Exception as e:
            st.error(f"Error processing image: {e}")

with col2:
    st.subheader("📊 System Status")
    
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
        st.subheader("📋 Recent Predictions")
        for i, pred in enumerate(st.session_state.prediction_history[-3:]):
            disease_name = pred['disease'].replace("_", " ").replace("___", " - ").title()
            confidence = pred.get('confidence', 0)
            st.write(f"{i+1}. **{disease_name}** ({confidence:.0%})")
    
    # Tips
    st.subheader("💡 Tips for Best Results")
    tips = [
        "Use clear, well-lit photos",
        "Focus on the leaf, not the whole plant",
        "Avoid very dark or blurry images",
        "Try multiple leaves if symptoms differ",
    ]
    for tip in tips:
        st.markdown(f"""
        <div style="background: white; border-radius: 10px; padding: 0.8rem;
                    margin: 0.5rem 0; box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                    border-left: 3px solid #3CB371;">
            <p style="margin: 0; color: #555; font-size: 0.9rem;">• {tip}</p>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <p style="margin: 0; font-size: 0.9rem;">
        <strong>AI-powered plant health analysis</strong> • Keep your plants thriving 🌱
    </p>
    <p style="margin: 0; font-size: 0.8rem; color: #999;">
        Model: EfficientNetB2 • 38 plant disease classes
    </p>
</div>
""", unsafe_allow_html=True)
