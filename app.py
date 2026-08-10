import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load both models
small_model = load_model("small_cnn_model.h5")
big_model = load_model("big_cnn_model.h5")

st.title("Leaf Disease Classifier — Small vs Big CNN")

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "png"])

def predict(model, img):
    img = image.load_img(img, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    prediction = model.predict(img_array)
    return prediction

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    small_pred = predict(small_model, uploaded_file)
    big_pred = predict(big_model, uploaded_file)

    small_class = np.argmax(small_pred)
    big_class = np.argmax(big_pred)

    st.subheader("Small CNN Prediction")
    st.write(f"Class: {small_class}")
    st.write(f"Confidence: {np.max(small_pred):.4f}")

    st.subheader("Big CNN Prediction")
    st.write(f"Class: {big_class}")
    st.write(f"Confidence: {np.max(big_pred):.4f}")

    st.subheader("Comparison")
    if np.max(big_pred) > np.max(small_pred):
        st.write("**Big CNN is more confident**")
    else:
        st.write("**Small CNN is more confident**")
