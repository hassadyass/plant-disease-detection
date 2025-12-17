from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import io
from PIL import Image
from class_names import class_names  # Liste des classes

app = Flask(__name__)

# Chargement du modèle
model = load_model("models/plant_disease_model_mobilenet.h5", compile=False)

def preprocess_image_from_bytes(image_bytes, target_size=(224, 224)):
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize(target_size)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ✅ Route d'accueil pour éviter les erreurs GET /
@app.route('/')
def index():
    return "✅ API de détection des maladies de plantes est active !"

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    img_bytes = file.read()

    try:
        img_preprocessed = preprocess_image_from_bytes(img_bytes)
        preds = model.predict(img_preprocessed)
        predicted_index = np.argmax(preds, axis=1)[0]
        confidence = preds[0][predicted_index]
        predicted_class = class_names[predicted_index]

        seuil_confiance = 0.7
        if confidence < seuil_confiance:
            return jsonify({
                'result': 'unknown',
                'confidence': float(confidence)
            })

        return jsonify({
            'result': predicted_class,
            'confidence': float(confidence)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5050)




#D:\model\env\Scripts\pythonw.exe -m idlelib.idle









'''#!/usr/bin/env python
# coding: utf-8

import sys
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# === Config ===
MODEL_DIR = "D:/model"
MODEL_IMAGE_PATH = os.path.join(MODEL_DIR, "models/plant_disease_model_mobilenet.h5") 
IMG_PATH = os.path.join(MODEL_DIR, "images/strawbery.jpg")

# Ajouter le dossier MODEL_DIR au path pour imports
sys.path.append(MODEL_DIR)

# Import class_names généré par entraînement
from class_names import class_names

# === Fonction de prétraitement ===
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array
    except Exception as e:
        print(f"❌ Erreur lors du chargement de l'image : {e}")
        sys.exit(1)

# === Chargement du modèle ===
print("🔍 Chargement du modèle CNN (.h5)...")
try:
    model = load_model(MODEL_IMAGE_PATH, compile=False)
except Exception as e:
    print(f"❌ Erreur de chargement du modèle : {e}")
    sys.exit(1)

# === Prédiction ===
img_prepared = preprocess_image(IMG_PATH)
preds = model.predict(img_prepared)

predicted_index = np.argmax(preds, axis=1)[0]
predicted_class = class_names[predicted_index]
confidence = preds[0][predicted_index]

# === Résultat ===
print("\n📷 Image analysée :", IMG_PATH)

seuil_confiance = 0.7  
if confidence >= seuil_confiance:
    print(f"➡️ Classe prédite : {predicted_class} (indice {predicted_index})")
    print(f"🎯 Confiance : {confidence:.4f}")
else:
    print(f"⚠️ Confiance trop faible ({confidence:.4f}),Culture inconnue ou image non reconnue par le modèle.")

'''



