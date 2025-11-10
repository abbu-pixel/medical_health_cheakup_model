import os
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.tensorflow
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import json

# -------------------------------------------------------------------
# 📍 Project Paths (always relative to repo root)
# -------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "mlruns")

os.makedirs(MLRUNS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------------
# ⚙️ Configure MLflow for Local + Render (portable)
# -------------------------------------------------------------------
mlflow.set_tracking_uri("file:///" + MLRUNS_DIR.replace("\\", "/"))
mlflow.set_experiment("medical_checkup_models")

print(f"✅ MLflow Tracking Directory: {MLRUNS_DIR}")

# -------------------------------------------------------------------
# 🚀 Train and Log Models
# -------------------------------------------------------------------
def train_models(processed_data_dir: str):
    print(f"📂 Loading data from: {processed_data_dir}")

    X_train = pd.read_csv(os.path.join(processed_data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(processed_data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(processed_data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_data_dir, "y_test.csv")).values.ravel()

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
        mlflow.sklearn.log_model(model, "model")

        results["RandomForest"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["RandomForest"] = model
        print(f"✅ RandomForest: {acc:.4f}")

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
        mlflow.xgboost.log_model(model, "model")

        results["XGBoost"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["XGBoost"] = model
        print(f"✅ XGBoost: {acc:.4f}")

    # 3️⃣ ANN
    with mlflow.start_run(run_name="ANN") as run:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = Sequential([
            Input(shape=(X_train.shape[1],)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        es = EarlyStopping(patience=3, restore_best_weights=True)

        model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=10, batch_size=32, verbose=1, callbacks=[es])
        _, acc = model.evaluate(X_test_scaled, y_test, verbose=0)

        mlflow.log_params({"layers": [64, 32], "activation": "relu", "epochs": 10, "batch_size": 32})
        mlflow.log_metric("accuracy", acc)
        mlflow.tensorflow.log_model(model, "model")

        results["ANN"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["ANN"] = model
        print(f"✅ ANN: {acc:.4f}")

    # 🏆 Choose best model
    best_model_name, best_info = max(results.items(), key=lambda x: x[1]["acc"])
    best_model = trained_models[best_model_name]
    print(f"\n🏆 Best Model: {best_model_name} ({best_info['acc']:.4f})")

    # 🧾 Save model + metadata
    model_path = os.path.join(MODELS_DIR, f"{best_model_name}_model.pkl")
    joblib.dump(best_model, model_path)

    metadata_path = os.path.join(MODELS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump({"best_model": best_model_name, "accuracy": best_info["acc"]}, f, indent=4)

    print(f"✅ Saved model at: {model_path}")
    print(f"🧾 Metadata stored at: {metadata_path}")

    # 🧠 Try model registry safely
    try:
        mlflow.register_model(model_uri=f"runs:/{best_info['run_id']}/model", name=f"MedicalCheckup_{best_model_name}")
        print(f"✅ Registered model: MedicalCheckup_{best_model_name}")
    except Exception as e:
        print(f"⚠️ Model registry skipped (likely local MLflow): {e}")

# -------------------------------------------------------------------
# 🏁 Entry Point
# -------------------------------------------------------------------
if __name__ == "__main__":
    train_models(DATA_DIR)
