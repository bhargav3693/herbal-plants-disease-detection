import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

class MED117Predictor:
    def __init__(self, model_path='models/med117_model.h5'):
        print("Loading model...")

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model not found at {model_path}. "
                "Please train the model first."
            )

        self.model = tf.keras.models.load_model(model_path)

        # Load class names
        class_names_path = 'models/class_names.json'
        if not os.path.exists(class_names_path):
            raise FileNotFoundError(f"Class names not found at {class_names_path}")

        with open(class_names_path, 'r') as f:
            self.class_names = json.load(f)

        print(f"✅ Model loaded successfully!")
        print(f"📊 Total classes: {len(self.class_names)}")

    def preprocess_image(self, image, target_size=(224, 224)):
        if isinstance(image, str):
            img = Image.open(image)
        elif hasattr(image, 'mode'):  # PIL Image object
            img = image
        else:  # File-like object (Streamlit UploadedFile)
            img = Image.open(image)

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img = img.resize(target_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict(self, image, top_k=5):
        processed_img = self.preprocess_image(image)
        predictions = self.model.predict(processed_img, verbose=0)
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]

        results = []
        for idx in top_indices:
            plant_name = self.class_names[idx]
            confidence = float(predictions[0][idx] * 100)

            result = {
                'plant_name': plant_name,
                'confidence': confidence,
                'botanical_name': plant_name.replace('_', ' ')
            }
            results.append(result)

        return results
