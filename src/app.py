import streamlit as st
import mlflow.pyfunc
import pandas as pd

# Load the latest production model from MLflow
MODEL_PATH = "./model_local"
st.sidebar.info(f"🔍 Loading model from MLflow: {MODEL_PATH} ...")
model = mlflow.pyfunc.load_model(MODEL_PATH)

st.title("Medical Checkup Prediction 🚑")

st.header("Enter Patient Details:")

# Input fields
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", options=[0, 1], help="0 = Male, 1 = Female")
heart_rate = st.number_input("Heart Rate (bpm)", min_value=30, max_value=200, value=70)
temperature = st.number_input("Temperature (°C)", min_value=35.0, max_value=42.0, value=37.0)
oxygen_level = st.number_input("Oxygen Level (%)", min_value=50, max_value=100, value=98)
glucose_level = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=400, value=90)
cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=180)
systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=150, value=80)

# Predict button
if st.button("Predict"):
    try:
        input_df = pd.DataFrame([{
            "age": age,
            "gender": gender,
            "heart_rate": heart_rate,
            "temperature": temperature,
            "oxygen_level": oxygen_level,
            "glucose_level": glucose_level,
            "cholesterol": cholesterol,
            "systolic_bp": systolic_bp,
            "diastolic_bp": diastolic_bp
        }])

        pred = int(model.predict(input_df)[0])
        result = "🟢 Healthy" if pred == 0 else "🔴 Needs Medical Attention"

        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {e}")
