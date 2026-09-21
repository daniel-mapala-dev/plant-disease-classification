import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import json
import sys
import os

MODEL_PATH = "models/plant_disease_model.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_HEIGHT = 128
IMG_WIDTH = 128

def predict_disease(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"[ERROR] Model not found at {MODEL_PATH}. Run train.py first.")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)

    img = image.load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100

    print(f"\nImage: {img_path}")
    print(f"Predicted: {class_names[predicted_class]}")
    print(f"Confidence: {confidence:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_image>")
    else:
        predict_disease(sys.argv[1])