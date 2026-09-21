import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Prevents GUI issues on Windows
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURATION ---
IMG_HEIGHT = 128   # Reduced from 224 for speed on CPU
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 5         # Start small; increase later
TRAIN_DIR = "dataset/train"
VAL_DIR = "dataset/val"
MODEL_SAVE_PATH = "models/plant_disease_model.h5"

# --- 2. DATA PREPROCESSING ---
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

class_names = list(train_generator.class_indices.keys())
print(f"\nFound {len(class_names)} classes.")
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}\n")

# --- 3. BUILD THE CNN MODEL ---
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 4. TRAIN THE MODEL ---
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator
)

# --- 5. SAVE THE MODEL ---
os.makedirs("models", exist_ok=True)
model.save(MODEL_SAVE_PATH)
print(f"\n[OK] Model saved to {MODEL_SAVE_PATH}")

with open("models/class_names.json", "w") as f:
    json.dump(class_names, f)
print("[OK] Class names saved to models/class_names.json")

# --- 6. PLOT TRAINING RESULTS ---
os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.tight_layout()
plt.savefig('plots/training_history.png')
print("[OK] Training plot saved to plots/training_history.png")

# --- 7. EVALUATE ON VALIDATION SET ---
print("\nEvaluating on validation set...")

# CRITICAL FIX: Do NOT use val_generator.classes with shuffle.
# Instead, collect predictions and true labels in order.
val_datagen_eval = ImageDataGenerator(rescale=1./255)

val_generator_eval = val_datagen_eval.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # <-- THE FIX
)

predictions = model.predict(val_generator_eval)
y_pred = np.argmax(predictions, axis=1)
y_true = val_generator_eval.classes

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('plots/confusion_matrix.png')
print("[OK] Confusion matrix saved to plots/confusion_matrix.png")