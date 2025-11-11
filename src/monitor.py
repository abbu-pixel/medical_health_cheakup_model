import os
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd

# Paths
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Load data
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

# Generate drift report
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=X_train, current_data=X_test)

# Save HTML report
report_path = os.path.join(REPORTS_DIR, "evidently_drift_report.html")
report.save_html(report_path)
print(f"✅ Evidently report saved at: {report_path}")
