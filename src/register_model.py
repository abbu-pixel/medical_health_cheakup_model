# src/register_model.py
import os
import json
import shutil

# Paths
model_path = "models/model.pkl"
metrics_path = "reports/metrics.json"
registry_dir = "registry"
registry_info_path = os.path.join(registry_dir, "model_info.json")

os.makedirs(registry_dir, exist_ok=True)

# Load metrics
with open(metrics_path, "r") as f:
    metrics = json.load(f)

# Save model info
model_info = {
    "model_path": model_path,
    "metrics": metrics,
    "status": "registered",
    "version": 1
}

with open(registry_info_path, "w") as f:
    json.dump(model_info, f, indent=4)

# Copy model to registry folder for versioning
shutil.copy(model_path, os.path.join(registry_dir, "model_v1.pkl"))

print("✅ Model registered successfully!")
print(json.dumps(model_info, indent=4))
