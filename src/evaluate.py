import joblib
import json
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd

# =====================================================
# 📦 Load model and test data
# =====================================================
MODEL_PATH = "models/model.pkl"
DATA_PATH = r"F:\mlops_project\data\processed\final_dataset.csv"

model = joblib.load(MODEL_PATH)
df = pd.read_csv(DATA_PATH)

# drop unnecessary columns
drop_cols = [c for c in df.columns if 'Patient ID' in c or 'Timestamp' in c or 'IP' in c]
df = df.drop(columns=drop_cols, errors='ignore')

if 'Target' not in df.columns:
    raise ValueError("❌ Target column not found in dataset.")

X = df.drop('Target', axis=1)
y = df['Target']

# simple split for evaluation (same as training)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 🎯 Evaluate
# =====================================================
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='weighted')

metrics = {"accuracy": acc, "f1_score": f1}

# =====================================================
# 💾 Save metrics for DVC tracking
# =====================================================
import os
os.makedirs("metrics", exist_ok=True)
with open("metrics/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
# =====================================================
# 📊 Save Confusion Matrix
# =====================================================
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

os.makedirs("reports", exist_ok=True)

disp = ConfusionMatrixDisplay.from_predictions(y_test, preds)
plt.title("Confusion Matrix")
plt.savefig("reports/confusion_matrix.png")
plt.close()

# Also copy metrics to reports for DVC output tracking
import shutil
shutil.copy("metrics/metrics.json", "reports/metrics.json")

print("✅ Evaluation complete!")
print(json.dumps(metrics, indent=4))


print("✅ Evaluation complete!")
print(json.dumps(metrics, indent=4))
