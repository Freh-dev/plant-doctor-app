import os
import shutil
from sklearn.model_selection import train_test_split
import numpy as np

# Set paths
source_dir = r'C:\Users\Frita\Desktop\Python\PlantVillage-Dataset\raw\color'
train_dir = r'C:\Users\Frita\Desktop\Python\Project\ML_AI_Project\data\train'
val_dir = r'C:\Users\Frita\Desktop\Python\Project\ML_AI_Project\data\val'

# Create directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Get all class folders
class_folders = [f for f in os.listdir(source_dir) 
                if os.path.isdir(os.path.join(source_dir, f))]

for class_folder in class_folders:
    class_path = os.path.join(source_dir, class_folder)
    
    # Get all images in this class
    images = [f for f in os.listdir(class_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Split images: 80% train, 20% validation
    train_images, val_images = train_test_split(
        images, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Create class directories in train and validation folders
    os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
    
    # Copy train images
    for img in train_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_dir, class_folder, img)
        shutil.copy2(src, dst)
    
    # Copy validation images
    for img in val_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_dir, class_folder, img)
        shutil.copy2(src, dst)
    
    print(f"Class {class_folder}: {len(train_images)} train, {len(val_images)} validation")