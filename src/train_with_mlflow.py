import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from tensorflow import keras
import scipy.sparse

# ✅ Load dataset
data_path = r"F:\data\processed\final_dataset.csv"
df = pd.read_csv(data_path)

print("✅ Dataset loaded:", df.shape)
print("Columns:", df.columns.tolist())

# ✅ Clean dataset
drop_cols = [c for c in df.columns if 'Patient ID' in c or 'Timestamp' in c or 'IP' in c]
df = df.drop(columns=drop_cols, errors='ignore')

# Ensure target column exists
if 'Target' not in df.columns:
    raise ValueError("❌ Target column not found. Check your CSV file.")

# Drop rows where Target is missing
df = df.dropna(subset=['Target'])

# ✅ Separate features and target
X = df.drop('Target', axis=1)
y = df['Target']

# ✅ Identify numeric/categorical columns
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

# ✅ Define preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, num_cols),
    ('cat', categorical_transformer, cat_cols)
])

# ✅ Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlflow.set_experiment("medical_checkup_models")

# -----------------------------
# RANDOM FOREST MODELS
# -----------------------------
rf_params = [
    {"n_estimators": 100, "max_depth": 5},
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 300, "max_depth": 15},
]

for p in rf_params:
    with mlflow.start_run(run_name=f"RandomForest_{p['n_estimators']}"):
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(**p, random_state=42))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_params(p)
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ RF {p} → acc={acc:.3f}, f1={f1:.3f}")

# -----------------------------
# XGBOOST MODELS
# -----------------------------
xgb_params = [
    {"n_estimators": 100, "max_depth": 3, "learning_rate": 0.1},
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.05},
    {"n_estimators": 300, "max_depth": 7, "learning_rate": 0.01},
]

for p in xgb_params:
    with mlflow.start_run(run_name=f"XGBoost_{p['n_estimators']}"):
        model = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(**p, use_label_encoder=False, eval_metric='logloss', random_state=42))
        ])
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_params(p)
        mlflow.log_metrics({"accuracy": acc, "f1": f1})
        mlflow.sklearn.log_model(model, "model")
        print(f"✅ XGB {p} → acc={acc:.3f}, f1={f1:.3f}")

# -----------------------------
# ANN MODELS (Fixed)
# -----------------------------
def create_ann(input_dim, hidden_units=32, lr=0.001):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dense(hidden_units // 2, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Preprocess data
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# ✅ Convert sparse → dense for TensorFlow
if scipy.sparse.issparse(X_train_prep):
    X_train_prep = X_train_prep.toarray()
if scipy.sparse.issparse(X_test_prep):
    X_test_prep = X_test_prep.toarray()

input_dim = X_train_prep.shape[1]

ann_params = [
    {"hidden_units": 32, "lr": 0.001},
    {"hidden_units": 64, "lr": 0.0005},
    {"hidden_units": 128, "lr": 0.0001},
]

for p in ann_params:
    with mlflow.start_run(run_name=f"ANN_{p['hidden_units']}"):
        model = create_ann(input_dim, **p)
        model.fit(X_train_prep, y_train, epochs=10, batch_size=32, verbose=0)
        loss, acc = model.evaluate(X_test_prep, y_test, verbose=0)
        mlflow.log_params(p)
        mlflow.log_metrics({"accuracy": acc})
        mlflow.keras.log_model(model, "model")
        print(f"✅ ANN {p} → acc={acc:.3f}")

print("\n🎯 Training completed successfully! Run `mlflow ui --port 5000` to view logs.")
