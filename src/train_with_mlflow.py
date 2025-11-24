"""
src/train_with_mlflow.py

Train multiple models, log runs to MLflow, save best model and metadata.

Usage:
    python src/train_with_mlflow.py
"""

import os
import json
import argparse
from glob import glob

import pandas as pd
import joblib
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

# -----------------------------
# Configuration & paths
# -----------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data", "processed")
MODELS_DIR = os.path.join(ROOT, "models")
MLRUNS_DIR = os.path.join(ROOT, "mlruns")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(MLRUNS_DIR, exist_ok=True)

# MLflow tracking: local filesystem
mlflow.set_tracking_uri("file:///" + MLRUNS_DIR.replace("\\", "/"))
mlflow.set_experiment("medical_checkup_models")

# -----------------------------
# Helper functions
# -----------------------------
def load_data(data_dir):
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, X_test, y_train, y_test

def save_best_model(model_obj, model_name, accuracy):
    path = os.path.join(MODELS_DIR, f"{model_name}_model.pkl")
    joblib.dump(model_obj, path)
    meta = {"best_model": model_name, "accuracy": float(accuracy)}
    with open(os.path.join(MODELS_DIR, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=4)
    print(f"[INFO] Saved {model_name} -> {path} and metadata.json")

# Simple Keras wrapper for logging (so mlflow.tensorflow.log_model works)
def build_ann(input_dim):
    model = Sequential(
        [
            Input(shape=(input_dim,)),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

# -----------------------------
# Main training flow
# -----------------------------
def train_all(data_dir):
    X_train, X_test, y_train, y_test = load_data(data_dir)
    results = {}
    trained_models = {}

    # 1) RandomForest
    with mlflow.start_run(run_name="RandomForest") as run:
        params = {"n_estimators": 150, "max_depth": 10, "random_state": 42}
        rf = RandomForestClassifier(**params)
        rf.fit(X_train, y_train)
        preds = rf.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")

        results["RandomForest"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["RandomForest"] = rf
        print(f"[RUN] RandomForest accuracy={acc:.4f}")

    # 2) XGBoost
    with mlflow.start_run(run_name="XGBoost") as run:
        params = {"n_estimators": 200, "learning_rate": 0.1, "max_depth": 8, "random_state": 42}
        xgb = XGBClassifier(**params, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_train, y_train)
        preds = xgb.predict(X_test)
        acc = accuracy_score(y_test, preds)

        mlflow.log_params(params)
        mlflow.log_metric("accuracy", acc)
        mlflow.xgboost.log_model(xgb, "model")

        results["XGBoost"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["XGBoost"] = xgb
        print(f"[RUN] XGBoost accuracy={acc:.4f}")

    # 3) ANN (Keras/TensorFlow)
    with mlflow.start_run(run_name="ANN") as run:
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        ann = build_ann(X_train.shape[1])
        es = EarlyStopping(patience=3, restore_best_weights=True)

        ann.fit(X_train_s, y_train, validation_split=0.2, epochs=15, batch_size=32, callbacks=[es], verbose=0)
        loss, acc = ann.evaluate(X_test_s, y_test, verbose=0)

        mlflow.log_params({"layers": [64, 32], "activation": "relu", "epochs": 15, "batch_size": 32})
        mlflow.log_metric("accuracy", float(acc))
        # log scaler as artifact and ANN model
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        joblib.dump(scaler, scaler_path)
        mlflow.log_artifact(scaler_path, artifact_path="preprocessing")
        mlflow.tensorflow.log_model(ann, "model")

        results["ANN"] = {"acc": acc, "run_id": run.info.run_id}
        trained_models["ANN"] = (ann, scaler)  # store scaler with model
        print(f"[RUN] ANN accuracy={acc:.4f}")

    # Choose best model
    best_name, best_info = max(results.items(), key=lambda x: x[1]["acc"])
    best_model = trained_models[best_name]
    best_acc = best_info["acc"]
    print(f"\n[BEST] {best_name} with accuracy={best_acc:.4f}")

    # Save best model (if ANN, save the Keras model using joblib can't, so for ANN save via mlflow artifact or in models dir using mlflow)
    if best_name == "ANN":
        # save via mlflow as final artifact and also save Keras to HDF5 in models folder
        model_path = os.path.join(MODELS_DIR, f"{best_name}_model.h5")
        best_model[0].save(model_path)
        # save scaler already saved earlier
        mlflow.log_artifact(model_path, artifact_path="final_model")
        save_best_model({"keras_model": model_path, "scaler": scaler_path}, best_name, best_acc)
    else:
        # RandomForest/XGBoost stored as objects; save with joblib
        save_best_model(best_model, best_name, best_acc)

    # Attempt optional MLflow model registry registration (best-effort)
    try:
        run_id = best_info["run_id"]
        model_uri = f"runs:/{run_id}/model"
        registered_name = f"MedicalCheckup_{best_name}"
        mlflow.register_model(model_uri=model_uri, name=registered_name)
        print(f"[INFO] Registered model in MLflow Model Registry: {registered_name}")
    except Exception as e:
        print(f"[WARN] Could not register model in registry: {e}")

    return results

# -----------------------------
# CLI + DVC optional pull
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Processed data directory")
    parser.add_argument("--dvc-pull", action="store_true", help="Run `dvc pull` before training (if configured)")
    args = parser.parse_args()

    if args.dvc_pull:
        # optional - try to run dvc pull, skip if not available
        if os.system("dvc pull") != 0:
            print("[WARN] dvc pull failed or dvc not configured - continuing without DVC.")

    print(f"[INFO] MLflow tracking at: {mlflow.get_tracking_uri()}")
    results = train_all(args.data_dir)

    # Save a summary JSON with best metrics for quick access
    summary_path = os.path.join(MODELS_DIR, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Training summary written to {summary_path}")

if __name__ == "__main__":
    main()
