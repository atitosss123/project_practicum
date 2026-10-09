"""Перебор seed'ов для основного набора: чем обоснован STRUCTURED_SEED.

Для каждого seed генерируется набор, считается взвешенное смешанное расстояние
(0.6 количественные / 0.2 плазмида / 0.2 источник), иерархическая кластеризация
методом Уорда, подбор k по силуэту и совпадение с заложенными экотипами (ARI).

Отбираются seed'ы, где силуэт определяет k=3, а ARI попадает в коридор
0.65-0.93: восстановление хорошее, но не идеальное - один-два штамма относятся
не к своему экотипу, и это предмет интерпретации в отчёте. Идеальное
восстановление (ARI = 1) для учебного разбора бесполезно: обсуждать нечего.

Запуск:
    python data/seed_sweep.py                  # первые 200 seed'ов
    python data/seed_sweep.py --n-seeds 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_data import QUANT_FEATURES, STRUCTURED_SEED, generate_structured

WEIGHTS = {"quant": 0.6, "binary": 0.2, "categorical": 0.2}
K_RANGE = range(2, 11)


def _normalized(matrix):
    """Делим матрицу на её максимум; константный признак даёт нулевой максимум."""
    largest = matrix.max()
    return matrix / largest if largest > 0 else matrix


def mixed_distance(df):
    """Взвешенная смесь трёх расстояний по схеме из задания."""
    quant = _normalized(
        squareform(pdist(StandardScaler().fit_transform(df[QUANT_FEATURES]), "euclidean"))
    )
    binary = _normalized(
        squareform(pdist(df[["has_plasmid"]].to_numpy(float), "hamming"))
    )
    categorical = _normalized(
        squareform(pdist(pd.get_dummies(df["source"]).to_numpy(float), "euclidean"))
    )
    return (
        WEIGHTS["quant"] * quant
        + WEIGHTS["binary"] * binary
        + WEIGHTS["categorical"] * categorical
    )


def evaluate(seed):
    """Прогон на одном seed: оптимальное k, силуэт и совпадение с экотипами."""
    features, truth = generate_structured(seed=seed)
    distance = mixed_distance(features)
    tree = linkage(squareform(distance, checks=False), method="ward")

    scores = {
        k: silhouette_score(distance, fcluster(tree, k, "maxclust"), metric="precomputed")
        for k in K_RANGE
    }
    best_k = max(scores, key=scores.get)
    labels = fcluster(tree, best_k, "maxclust")
    return {
        "seed": seed,
        "best_k": best_k,
        "silhouette": scores[best_k],
        "ari": adjusted_rand_score(truth["ecotype"], labels),
    }


def main():
    parser = argparse.ArgumentParser(description="Перебор seed'ов основного набора.")
    parser.add_argument("--n-seeds", type=int, default=200, help="сколько seed'ов перебрать")
    parser.add_argument("--target-k", type=int, default=3, help="ожидаемое число кластеров")
    parser.add_argument("--ari-min", type=float, default=0.65)
    parser.add_argument("--ari-max", type=float, default=0.93)
    args = parser.parse_args()

    results = pd.DataFrame(evaluate(seed) for seed in range(1, args.n_seeds + 1))

    print(f"Перебрано seed'ов: {len(results)}")
    print("Распределение найденного k: "
          f"{results.best_k.value_counts().sort_index().to_dict()}")
    print(f"Силуэт: медиана {results.silhouette.median():.3f}, "
          f"разброс {results.silhouette.min():.3f}-{results.silhouette.max():.3f}")
    print(f"Совпадение с экотипами (ARI): медиана {results.ari.median():.2f}, "
          f"идеальных прогонов {int((results.ari > 0.999).sum())}")

    fits = results[
        (results.best_k == args.target_k)
        & results.ari.between(args.ari_min, args.ari_max)
    ]
    print(f"\nПодходящих (k={args.target_k} и ARI {args.ari_min}-{args.ari_max}): "
          f"{len(fits)} из {len(results)}")
    print(fits.head(12).to_string(index=False,
                                  formatters={"silhouette": "{:.3f}".format,
                                              "ari": "{:.2f}".format}))

    chosen = results[results.seed == STRUCTURED_SEED]
    if not chosen.empty:
        row = chosen.iloc[0]
        mark = "входит в подходящие" if STRUCTURED_SEED in set(fits.seed) else "НЕ ПОДХОДИТ"
        print(f"\nПринятый STRUCTURED_SEED={STRUCTURED_SEED}: k={row.best_k}, "
              f"силуэт={row.silhouette:.3f}, ARI={row.ari:.2f} - {mark}")


if __name__ == "__main__":
    main()
