from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os
import json
from glob import glob

app = Flask(__name__)

# -----------------------------
# 📁 Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS_DIR = os.path.join(BASE_DIR, "models")
METADATA_FILENAME = "metadata.json"
METADATA_PATH = os.path.join(MODELS_DIR, METADATA_FILENAME)

# -----------------------------
# ⚙️ Helper Functions
# -----------------------------
def get_latest_model(models_dir):
    """Get the most recently modified model file."""
    model_files = glob(os.path.join(models_dir, "*_model.pkl"))
    if not model_files:
        print("[ERROR] No model files found in 'models' directory.")
        return None
    model_files.sort(key=os.path.getmtime, reverse=True)
    return model_files[0]

def load_model(model_path):
    """Load model safely."""
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None
    try:
        model = joblib.load(model_path)
        print(f"[INFO] Loaded model: {model_path}")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None

# Load the latest model at startup
MODEL_PATH = get_latest_model(MODELS_DIR)
model = load_model(MODEL_PATH)

# -----------------------------
# 🌐 Flask Routes
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    """Render the web interface."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    data = request.form
    try:
        gender_map = {"Male": 1, "Female": 0}
        gender_encoded = gender_map.get(data.get("gender"), 0)

        # 🔹 Match training feature order
        input_df = pd.DataFrame([[
            float(data.get("age")),
            gender_encoded,
            float(data.get("heart_rate")),
            float(data.get("temperature")),
            float(data.get("oxygen_level")),
            float(data.get("glucose_level")),
            float(data.get("cholesterol")),
            float(data.get("systolic_bp")),
            float(data.get("diastolic_bp"))
        ]], columns=[
            "age", "gender", "heart_rate", "temperature",
            "oxygen_level", "glucose_level", "cholesterol",
            "systolic_bp", "diastolic_bp"
        ])

        # 🔹 Predict
        prediction = model.predict(input_df)[0]
        result_text = "Healthy ✅" if prediction == 1 else "Needs Attention ⚠️"

        return jsonify({"prediction": result_text})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route("/monitor", methods=["GET"])
def monitor_report():
    """Serve the Evidently data drift report as HTML."""
    report_path = os.path.join(os.path.dirname(__file__), "..", "reports", "data_drift_report.html")

    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
        # Optional auto-refresh every 10 seconds
        html = html.replace(
            "<head>",
            "<head><meta http-equiv='refresh' content='10'>"
        )
        return html
    else:
        return "<h3 style='color:red;'>❌ Drift report not found. Please check your CI/CD pipeline or run monitor.py manually.</h3>"


# -----------------------------
# 🚀 Run Server
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
