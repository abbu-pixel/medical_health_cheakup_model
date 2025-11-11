import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import json
import os

# -----------------------------
# 📁 Paths
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# -----------------------------
# 📊 Load Data
# -----------------------------
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

# -----------------------------
# 🧾 Generate Evidently Report
# -----------------------------
report = Report(metrics=[
    DataQualityPreset(),
    DataDriftPreset(),
])

report.run(reference_data=X_train, current_data=X_test)

# Save report as HTML
report_path = os.path.join(REPORTS_DIR, "data_drift_report.html")
report.save_html(report_path)

# Save key metrics as JSON
summary_path = os.path.join(REPORTS_DIR, "data_drift_summary.json")
with open(summary_path, "w") as f:
    json.dump(report.as_dict(), f, indent=4)

print(f"✅ Evidently report saved at: {report_path}")
print(f"📈 Summary JSON saved at: {summary_path}")
