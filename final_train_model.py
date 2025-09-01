# improved_train_FIXED.py
import tensorflow as tf
from tensorflow.keras import layers, models
import pathlib
import json

print("🚀 Starting IMPROVED plant disease model training...")

# Settings - Balanced approach
data_dir = pathlib.Path(r"C:\Users\Frita\Desktop\Python\Project\ML_AI_Project\data")
img_size = (150, 150)  # Slightly larger
batch_size = 12         # Balanced size
epochs = 15             # More epochs

print("📂 Loading data...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / "train",
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical',
    shuffle=True,
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir / "val", 
    image_size=img_size,
    batch_size=batch_size,
    label_mode='categorical'
)

class_names = train_ds.class_names
print(f"✅ Found {len(class_names)} classes: {class_names}")

# Better model architecture
model = models.Sequential([
    layers.Rescaling(1./255),
    
    # First conv block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Second conv block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Third conv block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    
    # Classifier
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("🧠 Model summary:")
model.summary()

# Callbacks for better training
callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
]

print("🚀 Training improved model...")
history = model.fit(
    train_ds,
    epochs=epochs,
    validation_data=val_ds,
    verbose=1,
    callbacks=callbacks
    # REMOVED: workers=2, use_multiprocessing=True
)

# Save results
model.save("improved_model.keras")
with open("class_names_improved.json", "w") as f:
    json.dump(class_names, f)




# create visual before the final print

import matplotlib.pyplot as plt
# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.close()  # Close the plot to free memory

# Then continue with your existing code...
print("✅ Training history plot saved as training_history.png")




# Final evaluation
print("📊 Evaluating...")
loss, accuracy = model.evaluate(val_ds, verbose=0)
print(f"🎯 Final Validation Accuracy: {accuracy:.2%}")

print("=" * 50)
print("✅ IMPROVED TRAINING COMPLETED!")
print("=" * 50)