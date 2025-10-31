import os
import mlflow
import mlflow.sklearn
import mlflow.tensorflow
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from preprocessing import load_data, preprocess_data

# 🧭 Paths
DATA_PATH = r"F:\data\raw\final_dataset.csv"  # update filename if needed
MODELS_DIR = r"F:\models"
os.makedirs(MODELS_DIR, exist_ok=True)

# 🧩 Load & preprocess data
df = load_data(DATA_PATH)
X_train, X_test, y_train, y_test, preprocessor = preprocess_data(df)

# Save preprocessor for future inference
joblib.dump(preprocessor, os.path.join(MODELS_DIR, "preprocessor.pkl"))

# 🧠 Initialize MLflow
mlflow.set_tracking_uri("file:///F:/mlruns")
mlflow.set_experiment("medical_checkup_experiments")

def evaluate_and_log(y_true, y_pred):
    """Compute metrics and log to MLflow"""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    mlflow.log_metrics({
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    })
    return acc, prec, rec, f1

# 🚀 Model 1: Logistic Regression
with mlflow.start_run(run_name="Logistic_Regression"):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc, prec, rec, f1 = evaluate_and_log(y_test, preds)
    print(f"✅ Logistic Regression: acc={acc:.3f}, f1={f1:.3f}")

    mlflow.sklearn.log_model(model, "logistic_regression_model")
    joblib.dump(model, os.path.join(MODELS_DIR, "logistic_model.pkl"))

# 🚀 Model 2: XGBoost
with mlflow.start_run(run_name="XGBoost"):
    model = XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc, prec, rec, f1 = evaluate_and_log(y_test, preds)
    print(f"✅ XGBoost: acc={acc:.3f}, f1={f1:.3f}")

    mlflow.sklearn.log_model(model, "xgboost_model")
    joblib.dump(model, os.path.join(MODELS_DIR, "xgboost_model.pkl"))

# 🚀 Model 3: ANN
with mlflow.start_run(run_name="ANN_Model"):
    input_dim = X_train.shape[1]
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop],
        verbose=1
    )

    preds = (model.predict(X_test) > 0.5).astype(int).flatten()
    acc, prec, rec, f1 = evaluate_and_log(y_test, preds)
    print(f"✅ ANN: acc={acc:.3f}, f1={f1:.3f}")

    mlflow.tensorflow.log_model(model, "ann_model")
    model.save(os.path.join(MODELS_DIR, "ann_model.h5"))

print("\n🎯 All models trained and logged successfully!")
print(f"Models saved in: {MODELS_DIR}")
print("To view MLflow UI, run: mlflow ui --backend-store-uri file:///F:/mlruns")
