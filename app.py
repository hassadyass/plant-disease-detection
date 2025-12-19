import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ======================
# Config
# ======================
IMG_SIZE = 224
MODEL_PATH = "models/plant_disease_modell_resnet50.keras"

# ======================
# Charger le modèle
# ======================
st.set_page_config(
    page_title="Plant Disease Detection",
    page_icon="🌿",
    layout="centered"
)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()
class_names = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Background_without_leaves', 'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy', 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight', 'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy']

st.title("🌱 Détection des maladies des plantes ")
st.write("Téléverse une image de feuille pour détecter la maladie.")

uploaded_file = st.file_uploader(
    "📸 Choisir une image",
    type=["jpg", "jpeg", "png"]
)
def clean_class_name(name):
    name = name.replace("___", " ").replace("__", " ").replace("_", " ")
    
    words = name.split()
    cleaned_words = []
    for w in words:
        if len(cleaned_words) == 0 or cleaned_words[-1].lower() != w.lower():
            cleaned_words.append(w)
    return " ".join(cleaned_words)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image sélectionnée", use_column_width=True)

    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)

    # Normalisation (IMPORTANT pour ResNet50)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    # ======================
    # Prédiction
    # ======================
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = predictions[0][predicted_index] * 100
    predicted_class = class_names[predicted_index]
    

    # ======================
    # Résultat
    # ======================
    st.success(f"🌿 Maladie détectée : *{clean_class_name(predicted_class)}*")
    st.info(f"🎯 Confiance : *{confidence:.2f}%*")

    # ======================
    # Probabilités (optionnel)
    # ======================
    st.subheader("📊 Probabilités par classe")
    prob_dict = {
        class_names[i]: float(predictions[0][i])
        for i in range(len(class_names))
    }
    st.bar_chart(prob_dict)