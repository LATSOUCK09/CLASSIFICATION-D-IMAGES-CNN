
import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F

# Charge les modèles
@st.cache_resource
def load_cats_dogs_model():
    return load_model("models/CAT_DOG.h5")

@st.cache_resource
def load_malaria_model():
    return load_model("models/Cell_img.h5")

@st.cache_resource

def load_cifar10_model():
    return load_model("models/CIFAR10_CNN.h5")

#def load_cifar10_model():
    #model = torch.load("CIFAR10_CNN.ph", map_location=torch.device('cpu'))
    #model.eval()
    #return model

# Prédictions
from PIL import Image
import numpy as np

def predict_cats_dogs(image):
    # 1. Redimensionner l’image à 128x128
    image = image.resize((128, 128))  # ⚠️ adapter à la taille du modèle

    # 2. Convertir en tableau numpy et normaliser les pixels entre 0 et 1
    img_array = np.array(image) / 255.0

    # 3. Ajouter une dimension pour simuler un batch (forme : (1, 128, 128, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Prédire avec le modèle
    prediction = cats_dogs_model.predict(img_array)[0][0]

    # 5. Retourner le résultat lisible
    label = "Chien" if prediction >= 0.5 else "Chat"
    return label, float(prediction)

#def predict_cats_dogs(image):
    #img = image.resize((150, 150))
    #img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    #prediction = cats_dogs_model.predict(img_array)[0][0]
    #label = "Chien" if prediction >= 0.5 else "Chat"
    #return label, float(prediction)

def predict_malaria(image):
    #img = image.resize((64, 64))
    img = image.resize((50, 50)) 
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)
    prediction = malaria_model.predict(img_array)[0][0]
    label = "Parasitée" if prediction >= 0.5 else "Non parasitée"
    return label, float(prediction)
def predict_cifar10(image):
    # 1. Redimensionner l’image à 32x32 (taille CIFAR-10)
    image = image.resize((32, 32))

    # 2. Convertir en tableau numpy et normaliser pixels entre 0 et 1
    img_array = np.array(image) / 255.0

    # 3. Ajouter une dimension batch (forme : (1, 32, 32, 3))
    img_array = np.expand_dims(img_array, axis=0)

    # 4. Prédire avec le modèle
    predictions = cifar10_model.predict(img_array)[0]  # tableau de probas sur 10 classes

    # 5. Trouver la classe prédite (indice max)
    class_index = np.argmax(predictions)

    # 6. Labels CIFAR-10
    cifar10_labels = ['avion', 'auto', 'oiseau', 'chat', 'cerf', 
                      'chien', 'grenouille', 'cheval', 'bateau', 'camion']

    # 7. Retourner le label et la probabilité associée
    predicted_label = cifar10_labels[class_index]
    predicted_prob = float(predictions[class_index])

    return predicted_label, predicted_prob


# Interface Streamlit
st.title("🧠 Application de classification d'images par Deep Learning")

model_choice = st.selectbox("Choisissez un modèle :", [
    "Chat vs Chien",
    "Cellules Parasitaires",
    "CIFAR-10"
])

uploaded_file = st.file_uploader("Uploadez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Image chargée", use_column_width=True)

    with st.spinner("Analyse en cours..."):
        if model_choice == "Chat vs Chien":
            cats_dogs_model = load_cats_dogs_model()
            label, proba = predict_cats_dogs(image)
        elif model_choice == "Cellules Parasitaires":
            malaria_model = load_malaria_model()
            label, proba = predict_malaria(image)
        elif model_choice == "CIFAR-10":
            cifar10_model = load_cifar10_model()
            label, proba = predict_cifar10(image)

    st.success(f"✅ Résultat : {label} ({proba*100:.2f}%)")
else:
    st.info("Veuillez uploader une image pour commencer.")
