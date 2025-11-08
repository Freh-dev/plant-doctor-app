# streamlit_app.py - UPDATED MODEL LOADING
@st.cache_resource
def load_model():
    try:
        # Use the fixed MobileNetV2 model
        model = tf.keras.models.load_model("plantvillage_mobilenetv2_fixed.h5")
        st.sidebar.success("✅ Advanced AI Model Loaded!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        return None

@st.cache_data
def load_class_names():
    try:
        with open("class_names_improved.json", "r") as f:
            return json.load(f)
    except Exception as e:
        st.error(f"❌ Error loading class names: {e}")
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
            "Tomato_Spider_mites", "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", 
            "Tomato_Mosaic_virus"
        ]

# Load resources
model = load_model()
class_names = load_class_names()
img_size = (128, 128)  # MobileNetV2 0.50_128 expects 128x128 input

def predict_image(image):
    """Predict plant disease from image"""
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
