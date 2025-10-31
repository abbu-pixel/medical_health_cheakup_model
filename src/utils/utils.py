from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, r2_score, mean_squared_error

def get_classification_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0)
    }

def get_regression_metrics(y_true, y_pred):
    return {
        "r2_score": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred)
    }
