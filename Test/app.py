import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CanTrack AI Diagnostics",
    page_icon="🔬",
    layout="centered"
)

# --- MODEL CONFIGURATION ---
MODELS_CONFIG = {
    "Leukemia Classifier (ALL)": {
        "model_path": "best_all_model.keras",   # same dir as app.py
        "image_size": (456, 456),
        "class_names": ['all_benign', 'all_early', 'all_pre', 'all_pro'],
        "description": "This model classifies Acute Lymphoblastic Leukemia subtypes from blood smear images."
    },
    "Brain Cancer Classifier": {
        "model_path": "best_brain_model.keras",  # same dir as app.py
        "image_size": (380, 380),
        "class_names": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "description": "This model classifies brain tumor types (Glioma, Meningioma, Pituitary) from MRI scans."
    }
    # You can add more models here if needed
}

# --- HELPER FUNCTIONS ---
@st.cache_resource
def load_keras_model(model_path: str):
    """Loads a Keras model from the specified path, with caching."""
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"⚠️ Error loading model from {model_path}: {e}")
        return None

def predict(model, image_to_predict, image_size, class_names):
    """Preprocesses an image and returns the model's prediction."""
    img = image_to_predict.resize(image_size)
    img_array = np.array(img)

    # Convert grayscale to RGB if needed
    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = np.max(predictions[0]) * 100
    return predicted_class, confidence

# --- SIDEBAR ---
st.sidebar.title("🧠 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a diagnostic model:",
    list(MODELS_CONFIG.keys())
)

# --- MODEL CONFIG ---
config = MODELS_CONFIG[model_choice]
model_path = config["model_path"]
image_size = config["image_size"]
class_names = config["class_names"]
description = config["description"]

# --- MAIN UI ---
st.title(f"⚕️ {model_choice}")
st.write(description)

# Load the selected model
model = load_keras_model(model_path)

if model:
    uploaded_file = st.file_uploader(
        "📤 Upload an image for analysis...", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption='Uploaded Image', use_column_width=True)

        with col2:
            with st.spinner('🤖 AI is analyzing the image...'):
                predicted_class, confidence = predict(model, image, image_size, class_names)
            st.success("✅ Analysis Complete")
            st.metric(label="Predicted Class", value=predicted_class)
            st.metric(label="Confidence", value=f"{confidence:.2f}%")
else:
    st.warning("⚠️ Model not loaded. Please ensure the .keras files are in the same folder as this app.")
