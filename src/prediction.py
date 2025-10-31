import pickle
import pandas as pd

# Load model
model_path = r"F:\models\model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Correct sample input (matching training types)
sample_input = {
    "Heart Rate (bpm)": 78,                # numeric
    "Temperature (°C)": 36.6,              # numeric
    "Blood Pressure (mmHg)": "120",        # categorical, must be string
    "Device ID": "Device_A1",              # categorical
    "Access Type": "Mobile",               # categorical
    "Action": "Login"                      # categorical
}

def predict(data):
    df = pd.DataFrame([data])
    print("\n🧾 Input DataFrame:\n", df)
    prediction = model.predict(df)
    return prediction[0]

if __name__ == "__main__":
    print("\n✅ Prediction:", predict(sample_input))
