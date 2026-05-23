from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(estimator, x, y):
    y_pred = estimator.predict(x)
    return {
        "mse": mean_squared_error(y, y_pred),
        "mae": mean_absolute_error(y, y_pred),
        "r2": r2_score(y, y_pred),
    }


def print_metrics(metrics, split_name):
    print(f"\nMétricas de {split_name}:")
    print(f"  MSE: {metrics['mse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  R2:  {metrics['r2']:.4f}")
