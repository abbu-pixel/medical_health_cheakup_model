from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def get_classification_metrics(y_true, y_pred):
    return {"accuracy": accuracy_score(y_true, y_pred)}

def get_regression_metrics(y_true, y_pred):
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred)
    }
