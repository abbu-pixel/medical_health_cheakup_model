import mlflow
from train_with_mlflow import train_models

def auto_retrain():
    print("🔁 Auto retraining triggered...")
    train_models("../data/processed")
    print("✅ Retraining completed and logged to MLflow.")

if __name__ == "__main__":
    auto_retrain()
