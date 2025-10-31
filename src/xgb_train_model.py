import mlflow
import mlflow.xgboost
import xgboost as xgb
from utils.data_loader import load_data
from utils.metrics import get_classification_metrics, get_regression_metrics

def train_xgb(task_type="classification", n_estimators=100, max_depth=5, learning_rate=0.1):
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment("XGBoost_Medical_Checkup")

    with mlflow.start_run():
        if task_type == "classification":
            model = xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                use_label_encoder=False,
                eval_metric="logloss"
            )
        else:
            model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate
            )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics_dict = (get_classification_metrics(y_test, y_pred)
                        if task_type == "classification"
                        else get_regression_metrics(y_test, y_pred))

        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "task_type": task_type
        })
        for k, v in metrics_dict.items():
            mlflow.log_metric(k, v)

        mlflow.xgboost.log_model(model, "model")

        print(f"✅ XGBoost training complete | Task: {task_type} | Metrics: {metrics_dict}")

if __name__ == "__main__":
    train_xgb(task_type="classification")
