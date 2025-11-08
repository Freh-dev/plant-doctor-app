# fix_model.py
import tensorflow as tf

print("🔧 Fixing MobileNetV4 model...")

# Load the problematic model
try:
    model = tf.keras.models.load_model("plantvillage_finetuned_mobilenetv4.h5")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# Debug: Show model structure
print("Model outputs:", len(model.outputs))
for i, output in enumerate(model.outputs):
    print(f"Output {i} shape:", output.shape)
    print(f"Output {i} name:", output.name)

# Try to identify which output is for classification
classification_output = None
for i, output in enumerate(model.outputs):
    if len(output.shape) == 2:  # 2D output (batch_size, num_classes)
        classification_output = output
        print(f"🎯 Found classification output at index {i}")
        break

if classification_output is None:
    print("⚠️ No clear classification output found, using first output")
    classification_output = model.outputs[0]

# Create fixed model
fixed_model = tf.keras.Model(inputs=model.inputs, outputs=classification_output)

# Save the fixed model
fixed_model.save("plantvillage_finetuned_mobilenetv4_fixed.h5")
print("✅ Fixed model saved as 'plantvillage_finetuned_mobilenetv4_fixed.h5'")

# Test the fixed model
try:
    test_model = tf.keras.models.load_model("plantvillage_finetuned_mobilenetv4_fixed.h5")
    print("✅ Fixed model loads successfully!")
    print(f"Fixed model inputs: {len(test_model.inputs)}")
    print(f"Fixed model outputs: {len(test_model.outputs)}")
except Exception as e:
    print(f"❌ Fixed model test failed: {e}")
