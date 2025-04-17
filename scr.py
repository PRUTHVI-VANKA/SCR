import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.sidebar.write("🔄 Loading model...")
model = tf.keras.models.load_model("pv.keras")

skin_type_classes = ['Dry', 'Normal', 'Oily']
acne_classes = ['Low', 'Moderate', 'Severe']

def preprocess_image(img: Image.Image, size=(224,224)):
    img = img.resize(size)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)

st.set_page_config(page_title="Skin & Acne Predictor", layout="centered")
st.title("🌟 Skin Type & Acne Severity Predictor")
st.write("Upload a face photo; the model will tell you your skin type **and** acne severity.")

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg","jpeg","png"])
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    x = preprocess_image(img)
    preds = model.predict(x)
    
    if isinstance(preds, list) and len(preds) == 2:
        skin_pred, acne_pred = preds
    else:
        skin_pred, acne_pred = np.split(preds, 2, axis=1)
    
    skin_idx = np.argmax(skin_pred[0])
    acne_idx = np.argmax(acne_pred[0])
    
    skin_label = skin_type_classes[skin_idx]
    acne_label = acne_classes[acne_idx]
    
    st.subheader("🧴 Skin Type")
    st.success(f"**{skin_label}** ({skin_pred[0][skin_idx]*100:.1f}%)")
    
    st.subheader("💥 Acne Severity")
    st.success(f"**{acne_label}** ({acne_pred[0][acne_idx]*100:.1f}%)")
    
    st.balloons()
