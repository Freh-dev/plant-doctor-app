# inference.py 
import tensorflow as tf
import numpy as np
import json 
from tensorflow.keras.preprocessing import image  # ✅ This works fine!

# Load model and class names
model = tf.keras.models.load_model("improved_model.keras")  
img_size = (150, 150)  

with open("class_names_improved.json", "r") as f:
    class_names = json.load(f)

def predict_image(img_path, top_k=3):
    img = image.load_img(img_path, target_size=img_size)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, 0)  # batch
    preds = model.predict(arr)[0]
    top_idx = preds.argsort()[-top_k:][::-1]
    return [(class_names[i], float(preds[i])) for i in top_idx]

if __name__ == "__main__":
    result = predict_image("sample_leaf.jpg")
    print("Prediction:", result)