from flask import Flask, render_template, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

# Load the latest production model from MLflow
MODEL_NAME = "Medical_Checkup_XGBoost"
print(f"🔍 Loading model from MLflow: {MODEL_NAME}/Production ...")
model = mlflow.pyfunc.load_model("./model_local")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form

        # Convert input form data to DataFrame
        df = pd.DataFrame([{
            "age": float(data["age"]),
            "gender": int(data["gender"]),
            "heart_rate": float(data["heart_rate"]),
            "temperature": float(data["temperature"]),
            "oxygen_level": float(data["oxygen_level"]),
            "glucose_level": float(data["glucose_level"]),
            "cholesterol": float(data["cholesterol"]),
            "systolic_bp": float(data["systolic_bp"]),
            "diastolic_bp": float(data["diastolic_bp"])
        }])

        pred = int(model.predict(df)[0])

        # Map prediction to human-readable result
        if pred == 0:
            result = "🟢 Healthy"
        else:
            result = "🔴 Needs Medical Attention"

        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
