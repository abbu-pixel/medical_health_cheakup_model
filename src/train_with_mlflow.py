  import os
import pandas as pd
import joblib
import json
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# -----------------------------
# 📁 Project Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# -----------------------------
# 🚫 RESET DAGSHUB / REMOTE MLflow CONFIG
# -----------------------------
# This removes any previously cached remote MLflow configuration
mlflow.set_tracking_uri(None)
os.environ["MLFLOW_TRACKING_URI"] = ""
os.environ["MLFLOW_REGISTRY_URI"] = ""

# -----------------------------
# ⚙️ LOCAL MLflow Setup (CI/CD Safe)
# -----------------------------
# Use only local folder for MLflow tracking on GitHub Actions
mlflow.set_tracking_uri(MLRUNS_DIR)

mlflow.set_experiment("medical_checkup_models")
print(f"✅ MLflow Tracking Directory: {MLRUNS_DIR}")

# -----------------------------
# 🚀 Train and Log Models
# -----------------------------
def train_models(data_dir: str):
    print(f"📂 Loading data from: {data_dir}")

    # Load processed data
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()

    results = {}
    trained_models = {}

    # 1️⃣ RandomForest
    with mlflow.start_run(run_name="RandomForest") as run:
        params = {"n_estimators": 150, "max_depth": 10, "random_state": 42}
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # SAFE artifact path
        mlflow.sklearn.log_model(model, artifact_path="logged_model")

        results["RandomForest"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["RandomForest"] = model

        print(f"✅ RandomForest Accuracy: {acc:.4f}")

    # 2️⃣ XGBoost
    with mlflow.start_run(run_name="XGBoost") as run:
        params = {
            "n_estimators": 200,
            "learning_rate": 0.1,
            "max_depth": 8,
            "eval_metric": "logloss",
            "random_state": 42
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)

        # SAFE artifact path
        mlflow.xgboost.log_model(model, artifact_path="logged_model")

        results["XGBoost"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["XGBoost"] = model

        print(f"✅ XGBoost Accuracy: {acc:.4f}")

    # 🏆 Best Model
    best_model_name, best_info = max(results.items(), key=lambda x: x[1]["acc"])
    best_model = trained_models[best_model_name]

    print(f"\n🏆 Best Model: {best_model_name} ({best_info['acc']:.4f})")

    # Save model
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_model.pkl")
    joblib.dump(best_model, model_path)

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(
            {"best_model": best_model_name, "accuracy": best_info["acc"]},
            f,
            indent=4
        )

    print(f"✅ Saved best model at: {model_path}")
    print(f"🧾 Metadata stored at: {metadata_path}")

    # Register in mlflow local registry
    try:
        mlflow.register_model(
            model_uri=f"runs:/{best_info['run_id']}/logged_model",
            name=f"MedicalCheckup_{best_model_name}"
        )
        print(f"✅ Registered model: MedicalCheckup_{best_model_name}")
    except Exception as e:
        print(f"⚠️ Model registry skipped: {e}")

# -----------------------------
# 🏁 Entry Point
# -----------------------------
if __name__ == "__main__":
    train_models(DATA_DIR)
