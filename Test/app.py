import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="CanTrack AI Diagnostics",
    page_icon="🔬",
    layout="centered"
)

# --- BASE DIR ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# --- REMOTE MODEL URLs ---
REMOTE_MODELS = {
    "best_all_model.keras": "https://raw.githubusercontent.com/krishna7124/CanTrack/main/Test/best_all_model.keras",
    "best_brain_model.keras": "https://raw.githubusercontent.com/krishna7124/CanTrack/main/Test/best_brain_model.keras"
}

# --- MODEL CONFIGURATION ---
MODELS_CONFIG = {
    "Leukemia Classifier (ALL)": {
        "model_path": os.path.join(MODELS_DIR, "best_all_model.keras"),
        "image_size": (456, 456),
        "class_names": ['all_benign', 'all_early', 'all_pre', 'all_pro'],
        "description": "This model classifies Acute Lymphoblastic Leukemia subtypes from blood smear images."
    },
    "Brain Cancer Classifier": {
        "model_path": os.path.join(MODELS_DIR, "best_brain_model.keras"),
        "image_size": (380, 380),
        "class_names": ['brain_glioma', 'brain_menin', 'brain_tumor'],
        "description": "This model classifies brain tumor types (Glioma, Meningioma, Pituitary) from MRI scans."
    }
}

# --- DOWNLOAD FUNCTION ---
def download_model_if_needed(local_path, remote_url):
    if not os.path.exists(local_path):
        st.info(f"📥 Downloading model: {os.path.basename(local_path)}")
        try:
            r = requests.get(remote_url, stream=True)
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            st.success(f"✅ Downloaded {os.path.basename(local_path)}")
        except Exception as e:
            st.error(f"❌ Failed to download {os.path.basename(local_path)}: {e}")

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
    img = image_to_predict.resize(image_size)
    img_array = np.array(img)

    if img_array.ndim == 2:
        img_array = np.stack((img_array,) * 3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0) / 255.0
    predictions = model.predict(img_array)
    idx = np.argmax(predictions[0])
    return class_names[idx], np.max(predictions[0]) * 100

# --- SIDEBAR ---
st.sidebar.title("🧠 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a diagnostic model:",
    list(MODELS_CONFIG.keys())
)

# --- MODEL CONFIG ---
config = MODELS_CONFIG[model_choice]
model_path = config["model_path"]
remote_url = REMOTE_MODELS[os.path.basename(model_path)]
image_size = config["image_size"]
class_names = config["class_names"]
description = config["description"]

# --- DOWNLOAD MODEL IF NEEDED ---
download_model_if_needed(model_path, remote_url)

# --- MAIN UI ---
st.title(f"⚕️ {model_choice}")
st.write(description)

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
    st.warning("⚠️ Model not loaded. Please check the remote links or local folder.")
