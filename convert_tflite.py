# convert_tflite.py
import tensorflow as tf

model = tf.keras.models.load_model("plant_doctor_efnb0.keras")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# Optional: quantization
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("plant_doctor.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Saved TFLite model -> plant_doctor.tflite")
