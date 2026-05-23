from sklearn.neighbors import KNeighborsRegressor

from src.data import load_data, prepare_data, split_data
from src.metrics import compute_metrics, print_metrics

N_NEIGHBORS_OPTIONS = [1, 3, 5, 7, 9, 11, 15]


def train_knn():
    df = load_data()
    x, y = prepare_data(df)
    x_train, x_test, y_train, y_test = split_data(x, y)

    best_estimator = None
    best_r2 = float("-inf")

    for n_neighbors in N_NEIGHBORS_OPTIONS:
        estimator = KNeighborsRegressor(n_neighbors=n_neighbors)
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
    train_knn()
