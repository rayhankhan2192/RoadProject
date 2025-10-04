
import os
from threading import Lock
from typing import Tuple

# IMPORTANT: set TF flags BEFORE importing tensorflow
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import tensorflow as tf
from PIL import Image
from decouple import config
from django.conf import settings
#from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess

IMAGE_SIZE = 224
CLASS_NAMES = ['Crack', 'Pothole', 'Surface Erosion']
MODEL_PATH = config("MODEL_PATH")  # keep it in .env
MODEL_COMPILE = False  # inference only

_model = None
_model_lock = Lock()

def get_model():
    """Thread-safe singleton loader for the Keras model."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                if not MODEL_PATH:
                    raise RuntimeError("MODEL_PATH is not set in your environment/.env")
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")
                _model = tf.keras.models.load_model(MODEL_PATH, compile=MODEL_COMPILE)
    return _model

# Image saving
def save_uploaded_file_exact(file_obj) -> str:
    """
    Save the uploaded file under media/uploads/ with the ORIGINAL filename.
    Returns the full filesystem path.
    Overwrites if a file with the same name exists.
    """
    upload_dir = os.path.join(settings.MEDIA_ROOT, "uploads")
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, file_obj.name)

    with open(file_path, 'wb+') as destination:
        for chunk in file_obj.chunks():
            destination.write(chunk)
    return file_path

# Lighting / brightness
def get_lighting_condition(arr_0_255: np.ndarray) -> Tuple[str, float]:
    """Compute brightness (0..255) and map to a lighting label."""
    tensor = tf.convert_to_tensor(arr_0_255, dtype=tf.float32)
    grayscale = tf.image.rgb_to_grayscale(tensor)
    brightness = float(tf.reduce_mean(grayscale).numpy())  # 0..255

    if brightness < 50:
        lighting = "Very Low Light"
    elif brightness < 100:
        lighting = "Low Light"
    elif brightness < 170:
        lighting = "Moderate Light"
    elif brightness < 220:
        lighting = "Bright Light"
    else:
        lighting = "Very Bright Light"

    return lighting, round(brightness, 2)


def predict_image(img: Image.Image):
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    #img_array = preprocess_input(img_array)
    img_array_expanded = tf.expand_dims(img_array, 0)
    model = get_model()
    prediction = model.predict(img_array_expanded, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
    confidence = round(100 * np.max(prediction[0]), 2)

    lighting, brightness = get_lighting_condition(img_array)
    return predicted_class, confidence, lighting, brightness
