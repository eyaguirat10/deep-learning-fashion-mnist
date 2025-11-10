import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("model.h5")

# Class names
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

st.title("Fashion MNIST Classifier")

uploaded_file = st.file_uploader("Upload a 28x28 grayscale image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("L").resize((28, 28))
    st.image(img, caption="Input Image", width=150)

    img_array = np.array(img).reshape(1, -1).astype(float)

    # Optionnel : standardiser comme à l'entraînement
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    img_array = scaler.fit_transform(img_array)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]

    st.subheader(f"Prediction: {predicted_class}")
