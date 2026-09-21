import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
VAL_DIR = "dataset/val"
MODEL_PATH = "models/plant_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"

# Load model and class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

# Load validation data WITHOUT shuffling
val_datagen = ImageDataGenerator(rescale=1./255)
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # CRITICAL
)

# Predict
print("\nGenerating predictions...")
predictions = model.predict(val_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator.classes

# Report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix - Plant Disease Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
print("[OK] Confusion matrix saved to plots/confusion_matrix.png")

# Overall accuracy
accuracy = np.mean(y_pred == y_true)
print(f"\n[RESULT] Overall Validation Accuracy: {accuracy * 100:.2f}%")