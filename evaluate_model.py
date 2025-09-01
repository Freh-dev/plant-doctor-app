# evaluate_model.py
import tensorflow as tf
import numpy as np
import json 
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load model & dataset
model = tf.keras.models.load_model("improved_model.keras") 
img_size = (150, 150)  

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "data/val", image_size=img_size, batch_size=32, label_mode="categorical", shuffle=False
)
with open("class_names_improved.json", "r") as f:
    class_names = json.load(f)

# Collect predictions
y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

# Report
print(classification_report(y_true, y_pred, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
