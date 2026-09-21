import os
import shutil
import random

# --- CONFIGURATION ---
# Your original dataset path
SOURCE_DIR = r"C:\Users\HP\Desktop\plant-disease-classification\PlantVillage Dataset (Labeled)\Color Images"
# The new folder where we will create train/val split
BASE_DIR = r"C:\Users\HP\Desktop\plant-disease-classification\dataset"

TRAIN_DIR = os.path.join(BASE_DIR, "train")
VAL_DIR = os.path.join(BASE_DIR, "val")

# Split ratio: 80% train, 20% validation
SPLIT_RATIO = 0.8

# Create directories
os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# Get all class folders
classes = [d for d in os.listdir(SOURCE_DIR) if os.path.isdir(os.path.join(SOURCE_DIR, d))]
print(f"Found {len(classes)} classes: {classes}")

for class_name in classes:
    class_path = os.path.join(SOURCE_DIR, class_name)
    
    # Get all images in this class
    images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle images for random split
    random.shuffle(images)
    
    # Calculate split index
    split_idx = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Create class subfolders in train and val
    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)
    
    # Copy files (copying is safer than moving, so we keep the original)
    for img in train_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(TRAIN_DIR, class_name, img))
        
    for img in val_images:
        shutil.copy(os.path.join(class_path, img), os.path.join(VAL_DIR, class_name, img))
        
    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val images")

print("\n✅ Data split complete!")
print(f"Train directory: {TRAIN_DIR}")
print(f"Val directory: {VAL_DIR}")