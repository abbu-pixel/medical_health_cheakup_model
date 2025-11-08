import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

experiment_name = "medical_checkup_models"
experiment = client.get_experiment_by_name(experiment_name)

# Search for latest run where the run name is 'XGBoost'
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'XGBoost'",
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise ValueError("❌ No runs found with name 'XGBoost'. Check MLflow for available run names.")

best_run = runs[0]
run_id = best_run.info.run_id
model_uri = f"runs:/{run_id}/model"

# Register the model
registered_model = mlflow.register_model(model_uri, "Medical_Checkup_XGBoost")

# Promote to Production
client.transition_model_version_stage(
    name="Medical_Checkup_XGBoost",
    version=registered_model.version,
    stage="Production",
    archive_existing_versions=True
)

print(f"✅ XGBoost model (run_id={run_id}) promoted to Production as version {registered_model.version}")
