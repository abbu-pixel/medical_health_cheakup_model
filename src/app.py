import streamlit as st
import pandas as pd
import joblib
import os
import json
from glob import glob

# -----------------------------
# 📁 Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

METADATA_FILENAME = "metadata.json"
METADATA_PATH = os.path.join(MODELS_DIR, METADATA_FILENAME)

# -----------------------------
# ⚠️ Load Latest Model
# -----------------------------
def get_latest_model(models_dir):
    model_files = glob(os.path.join(models_dir, "*_model.pkl"))
    if not model_files:
        return None
    # Sort by modified time, latest first
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def load_model(model_path):
    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

MODEL_PATH = get_latest_model(MODELS_DIR)
model = load_model(MODEL_PATH)

# -----------------------------
# ⚙️ Streamlit UI
# -----------------------------
st.title("🩺 Medical Checkup Prediction")
st.markdown("This app predicts the health checkup status based on patient data.")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=["Male", "Female"])
systolic_bp = st.number_input("Systolic BP", min_value=50, max_value=250, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=30, max_value=150, value=80)
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)
temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=45.0, value=37.0)

# Encode gender
gender_map = {"Male": 1, "Female": 0}
gender_encoded = gender_map[gender]

# Prepare feature dataframe
input_df = pd.DataFrame(
    [[age, gender_encoded, systolic_bp, diastolic_bp, heart_rate, temperature]],
    columns=["age", "gender", "systolic_bp", "diastolic_bp", "heart_rate", "temperature"]
)

# Prediction
if st.button("Predict"):
    if model:
        try:
            prediction = model.predict(input_df)[0]
            st.success(f"✅ Predicted Status: {prediction}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
    else:
        st.warning("⚠️ Model not loaded. Cannot make prediction.")

# Show metadata
if os.path.exists(METADATA_PATH):
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    st.markdown(f"**Model Metadata:**\n```\n{json.dumps(metadata, indent=4)}\n```")
else:
    st.info("ℹ️ Metadata not found")
