# repair_and_resave.py
import os
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import tensorflow as tf

OLD_H5 = r"E:\Python\Research\2_Road Damage\RoadProject\TrainedModel\MOBILENETv1.o.h5"
NEW_KERAS = r"E:\Python\Research\2_Road Damage\RoadProject\TrainedModel\new.keras"

NUM_CLASSES = 3
INPUT_SHAPE = (224, 224, 3)

def build_clean_model():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights=None, input_shape=INPUT_SHAPE, name="efficientnetb0"
    )
    inputs = tf.keras.Input(shape=INPUT_SHAPE, name="input_image")
    x = base(inputs, training=False)                      # (None, 7, 7, 1280)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)       # (None, 1280)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="pred")(x)
    model = tf.keras.Model(inputs, outputs, name="pothole_efficientnet_clean")
    return model

model = build_clean_model()

# Try to salvage weights by name (will skip mismatches from the broken graph)
try:
    model.load_weights(OLD_H5, by_name=True, skip_mismatch=True)
    print("Loaded weights by name (skipped mismatches).")
except Exception as e:
    print(f"Weight load skipped (not fatal): {e}")

# Save as Keras v3 format
os.makedirs(os.path.dirname(NEW_KERAS), exist_ok=True)
model.save(NEW_KERAS)
print(f"Saved: {NEW_KERAS}")
