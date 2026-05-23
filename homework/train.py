import os

import joblib

from homework.train_elasticnet import train_elasticnet
from homework.train_knn import train_knn

if __name__ == "__main__":
    print("=" * 60)
    print("Entrenando ElasticNet...")
    print("=" * 60)
    elasticnet, r2_elasticnet = train_elasticnet()

    print("\n" + "=" * 60)
    print("Entrenando KNN...")
    print("=" * 60)
    knn, r2_knn = train_knn()

    print("\n" + "=" * 60)
    print("Resultados finales:")
    print(f"  Mejor ElasticNet R2: {r2_elasticnet:.4f}  -> {elasticnet}")
    print(f"  Mejor KNN R2:        {r2_knn:.4f}  -> {knn}")

    best = elasticnet if r2_elasticnet >= r2_knn else knn
    best_r2 = max(r2_elasticnet, r2_knn)
    print(f"\nMejor modelo: {best}  (R2={best_r2:.4f})")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best, "models/estimator.pkl")
    print("Modelo guardado en models/estimator.pkl")
