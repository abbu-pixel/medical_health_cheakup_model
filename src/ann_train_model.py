import mlflow
import mlflow.keras
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from utils.data_loader import load_data
from utils.metrics import get_classification_metrics, get_regression_metrics

def train_ann(task_type="classification", hidden_units=[128, 64], dropout_rate=0.3, epochs=20, batch_size=32):
    X_train, X_test, y_train, y_test = load_data()
    mlflow.set_experiment("ANN_Medical_Checkup")

    with mlflow.start_run():
        model = models.Sequential()
        model.add(layers.Input(shape=(X_train.shape[1],)))

        for units in hidden_units:
            model.add(layers.Dense(units, activation='relu'))
            model.add(layers.Dropout(dropout_rate))

        if task_type == "classification":
            model.add(layers.Dense(1, activation='sigmoid'))
            loss_fn = 'binary_crossentropy'
            metrics = ['accuracy']
        else:
            model.add(layers.Dense(1))
            loss_fn = 'mse'
            metrics = ['mse']

        model.compile(optimizer='adam', loss=loss_fn, metrics=metrics)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                            validation_data=(X_test, y_test), verbose=1)

        y_pred = model.predict(X_test)

        if task_type == "classification":
            y_pred = (y_pred > 0.5).astype(int)
            metrics_dict = get_classification_metrics(y_test, y_pred)
        else:
            metrics_dict = get_regression_metrics(y_test, y_pred)

        # Log to MLflow
        mlflow.log_params({
            "hidden_units": hidden_units,
            "dropout_rate": dropout_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "task_type": task_type
        })
        for k, v in metrics_dict.items():
            mlflow.log_metric(k, v)

        mlflow.keras.log_model(model, "model")

        print(f"✅ ANN training complete | Task: {task_type} | Metrics: {metrics_dict}")

if __name__ == "__main__":
    train_ann(task_type="classification")
