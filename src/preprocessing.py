import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_data(input_file: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_file)
    print(f"✅ Loaded {len(df)} records from {input_file}")

    # Split blood pressure into systolic/diastolic
    bp_split = df["blood_pressure"].astype(str).str.split("/", expand=True)
    df["systolic_bp"] = pd.to_numeric(bp_split[0], errors="coerce")
    df["diastolic_bp"] = pd.to_numeric(bp_split[1], errors="coerce")
    df.drop(columns=["blood_pressure"], inplace=True)

    # Encode gender
    le = LabelEncoder()
    df["gender"] = le.fit_transform(df["gender"])

    # Drop non-numeric / irrelevant columns
    df.drop(columns=["device_id", "checkup_date"], inplace=True, errors="ignore")

    # Drop any rows with missing values
    df.dropna(inplace=True)

    # Split features and target
    X = df.drop(columns=["status"])
    y = df["status"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Save processed data
    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)

    print(f"✅ Preprocessed data saved in {output_dir}")

if __name__ == "__main__":
    # Fix: correct relative paths for DVC run location
    preprocess_data("data/raw/medical_data.csv", "data/processed")
