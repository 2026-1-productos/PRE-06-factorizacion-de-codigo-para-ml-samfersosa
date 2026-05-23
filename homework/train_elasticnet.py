from sklearn.linear_model import ElasticNet

from src.data import load_data, prepare_data, split_data
from src.metrics import compute_metrics, print_metrics

HYPERPARAMS = [
    (0.5, 0.5),
    (0.2, 0.2),
    (0.1, 0.1),
    (0.1, 0.05),
    (0.3, 0.2),
]


def train_elasticnet():
    df = load_data()
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    best_estimator = None
    best_r2 = float("-inf")

    for alpha, l1_ratio in HYPERPARAMS:
        estimator = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=12345)
        estimator.fit(x_train, y_train)

        train_metrics = compute_metrics(estimator, x_train, y_train)
        test_metrics = compute_metrics(estimator, x_test, y_test)

        print(f"\n{estimator}:")
        print_metrics(train_metrics, "entrenamiento")
        print_metrics(test_metrics, "testing")

        if test_metrics["r2"] > best_r2:
            best_r2 = test_metrics["r2"]
            best_estimator = estimator

    return best_estimator, best_r2


if __name__ == "__main__":
    train_elasticnet()
