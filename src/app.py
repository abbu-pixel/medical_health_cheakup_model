import streamlit as st
import pandas as pd
import joblib
import json
import os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Load metadata
with open(os.path.join(MODELS_DIR, "metadata.json"), "r") as f:
    metadata = json.load(f)

best_model_file = f"{metadata['best_model']}_model.pkl"
model_path = os.path.join(MODELS_DIR, best_model_file)
model = joblib.load(model_path)

st.title("Medical Checkup Prediction 🚑")

# Example input
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", [0, 1])
# ... (other inputs)

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "age": age,
        "gender": gender,
        # add other fields here
    }])
    pred = model.predict(input_df)[0]
    result = "🟢 Healthy" if pred == 0 else "🔴 Needs Medical Attention"
    st.success(f"Prediction: {result}")
