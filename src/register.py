import mlflow
from mlflow.tracking import MlflowClient

# Track experiment
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # or your MLflow server URI
mlflow.set_experiment("Medical_Checkup_Experiment")

client = MlflowClient()

model_name = "medical_checkup_model"

# Register the model
result = mlflow.register_model(
    model_uri="runs:/<your_run_id>/model",  # Replace with your actual run_id
    name=model_name
)

print(f"✅ Auto-registered model version {result.version} from run {result.run_id}")

# --- Promote model to Staging automatically ---
client.transition_model_version_stage(
    name=model_name,
    version=result.version,
    stage="Staging"
)

print(f"🚀 Model {model_name} version {result.version} promoted to 'Staging'")
