from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model_path = r"F:\models\model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html", prediction=None)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect form inputs
        data = {
            "Heart Rate (bpm)": float(request.form["HeartRate"]),
            "Temperature (°C)": float(request.form["Temperature"]),
            "Blood Pressure (mmHg)": request.form["BloodPressure"],
            "Device ID": request.form["DeviceID"],
            "Access Type": request.form["AccessType"],
            "Action": request.form["Action"]
        }

        df = pd.DataFrame([data])
        prediction = int(model.predict(df)[0])

        # Dynamic message + color
        if prediction == 1:
            message = "⚠️ Warning! Possible abnormal condition detected."
            color = "danger"
        else:
            message = "✅ All vitals appear normal."
            color = "success"

        return render_template("index.html", prediction=message, color=color)

    except Exception as e:
        return render_template("index.html", prediction=f"❌ Error: {e}", color="danger")


if __name__ == "__main__":
    app.run(debug=True)
