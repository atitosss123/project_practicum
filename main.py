"""Сквозной прогон: от CSV до таблиц и графиков в каталоге out/.

Запуск:
    python main.py                                  # основной набор
    python main.py --data data/strains_uniform.csv --no-truth
    python main.py --n-permutations 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.clustering import cluster
from src.distances import mixed_distance
from src.loader import load_strains, load_truth
from src.preprocess import build_blocks
from src.profiling import (
    agreement_with_truth,
    assignments,
    categorical_echo,
    cluster_summary,
    describe_clusters,
    mismatched_strains,
)
from src.sensitivity import separating_gap, sweep, tipping_point
from src.significance import assess
from src.viz import (
    coordinates_table,
    describe_projection,
    mds_coordinates,
    pca_coordinates,
    plot_dendrogram,
    plot_projection,
    plot_silhouette_profile,
)

ROOT = Path(__file__).resolve().parent


def section(title: str):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main():
    parser = argparse.ArgumentParser(description="Кластеризация штаммов бактерий.")
    parser.add_argument("--data", default="data/strains_features.csv")
    parser.add_argument("--truth", default="data/strains_truth.csv")
    parser.add_argument("--no-truth", action="store_true",
                        help="не сверяться со скрытой истиной (её нет для контрольного набора)")
    parser.add_argument("--out", default="out")
    parser.add_argument("--n-permutations", type=int, default=200)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)

    # 1. Данные
    section("1. Данные")
    df = load_strains(ROOT / args.data)
    truth = None if args.no_truth else load_truth(ROOT / args.truth)
    print(f"Загружено {len(df)} штаммов из {args.data}")
    print(f"Источники: {df.source.value_counts().to_dict()}, "
          f"плазмида у {df.has_plasmid.sum()} штаммов")

    # 2. Предобработка и расстояния
    section("2. Предобработка и матрица расстояний")
    blocks = build_blocks(df)
    parts = mixed_distance(blocks)
    print(f"Веса: {parts.weights}")
    off_diagonal = ~np.eye(len(df), dtype=bool)
    for name in ("quantitative", "binary", "categorical"):
        values = np.unique(np.round(getattr(parts, name)[off_diagonal], 6))
        print(f"  {name:14s} различных значений на парах штаммов: {len(values)}"
              + ("  (ступенчатая компонента)" if len(values) <= 2 else ""))
    print(f"Итоговая матрица: {parts.mixed.shape}, "
          f"диапазон {parts.mixed[off_diagonal].min():.3f}-{parts.mixed.max():.3f}")
    pd.DataFrame(parts.mixed.round(4), index=df.strain_id, columns=df.strain_id).to_csv(
        out / "distance_matrix.csv"
    )

    # 3. Кластеризация
    section("3. Кластеризация и подбор числа кластеров")
    result = cluster(parts.mixed)
    print("Силуэт по k: " + "  ".join(f"{k}:{v:.3f}" for k, v in result.scores.items()))
    print(f"Оптимальное k = {result.best_k}, силуэт {result.silhouette:.3f}, "
          f"размеры кластеров {result.cluster_sizes}")
    if result.is_conclusive:
        print("Максимум однозначен.")
    else:
        print(f"ВНИМАНИЕ: силуэт не различает k = {result.plateau} "
              f"(разница в пределах допуска). Число кластеров определено неустойчиво.")

    control = cluster(parts.mixed, method="average")
    from sklearn.metrics import adjusted_rand_score
    print(f"Контроль методом average: k = {control.best_k}, "
          f"силуэт {control.silhouette:.3f}, совпадение с Уордом "
          f"ARI = {adjusted_rand_score(result.labels, control.labels):.2f}")

    assignments(df, result.labels).to_csv(out / "clusters.csv", index=False)
    pd.Series(result.scores, name="silhouette").rename_axis("k").to_csv(
        out / "silhouette_profile.csv"
    )

    # 4. Значимость
    section("4. Отличается ли структура от случайной")
    significance = assess(df, result.silhouette, n_draws=args.n_permutations)
    print(significance.summary())

    # 5. Профили кластеров
    section("5. Профили кластеров")
    summary = cluster_summary(df, result.labels)
    print(summary.to_string(index=False))
    summary.to_csv(out / "cluster_summary.csv", index=False)
    print("\n" + describe_clusters(summary))

    print("\nНасколько разбиение пересказывает категориальные признаки (ARI):")
    for name, value in categorical_echo(df, result.labels).items():
        print(f"  {name:18s} {value:.2f}")

    if truth is not None:
        agreement = agreement_with_truth(truth, result.labels)
        print(f"\nСовпадение с заложенными экотипами: ARI = {agreement['ari']:.2f}")
        print(agreement["crosstab"].to_string())
        mismatched = mismatched_strains(df, truth, result.labels)
        if not mismatched.empty:
            followed = int(mismatched.followed_ecotype.sum())
            print(f"\nШтаммы с ложной меткой источника ({len(mismatched)}): "
                  f"за настоящим экотипом пошли {followed}")
            print(mismatched.to_string(index=False))
            mismatched.to_csv(out / "mismatched_strains.csv", index=False)

    # 6. Устойчивость к весам
    section("6. Устойчивость к выбору весов")
    gap = separating_gap(blocks, truth.ecotype if truth is not None else df.source)
    print(f"Разделяющий разрыв по количественным признакам: {gap['separating_gap']} "
          f"(медианы {gap['median_within_group']} внутри и "
          f"{gap['median_between_groups']} между группами)")
    print(f"Плоский штраф за одну несовпавшую категорию: "
          f"{parts.weights['categorical']:.2f} - это "
          f"{parts.weights['categorical'] / gap['separating_gap']:.0%} разрыва")
    table = sweep(blocks, df, truth)
    print()
    print(table.to_string(index=False))
    table.to_csv(out / "weight_sensitivity.csv", index=False)
    if truth is not None:
        print("\n" + tipping_point(table))

    # 7. Визуализация
    section("7. Визуализация")
    coordinates, variance = coordinates_table(blocks, parts.mixed)
    coordinates.to_csv(out / "pca_coordinates.csv", index=False)
    print(f"Первые две главные компоненты объясняют "
          f"{variance.sum() * 100:.1f}% дисперсии количественных признаков")

    pca, _ = pca_coordinates(blocks)
    mds, stress = mds_coordinates(parts.mixed)
    quality = ("отличная" if stress < 0.05 else "хорошая" if stress < 0.10
               else "приемлемая" if stress < 0.20 else "ненадёжная")
    print(f"Напряжение MDS (Краскал, stress-1): {stress:.3f} - проекция {quality}")
    print("\nОписание проекции PCA:")
    print(describe_projection(pca, result.labels, blocks.strain_ids, variance))

    if args.no_plots:
        print("\nГрафики отключены (--no-plots); координаты сохранены в pca_coordinates.csv")
    else:
        files = [
            plot_silhouette_profile(result.scores, result.best_k,
                                    out / "silhouette_profile.png", result.plateau),
            plot_dendrogram(result.tree, blocks.strain_ids, result.best_k,
                            out / "dendrogram.png", result.plateau),
            plot_projection(pca, result.labels, blocks.strain_ids,
                            out / "pca_scatter.png",
                            "Главные компоненты количественных признаков\n"
                            "(окраска по кластерам смешанного расстояния)",
                            (f"PC1 ({variance[0] * 100:.1f}% дисперсии)",
                             f"PC2 ({variance[1] * 100:.1f}%)")),
            plot_projection(mds, result.labels, blocks.strain_ids,
                            out / "mds_scatter.png",
                            "Многомерное шкалирование по смешанной матрице\n"
                            "(та же геометрия, по которой кластеризовали)",
                            ("ось 1", "ось 2")),
        ]
        print("\nГрафики: " + ", ".join(f.name for f in files))

    section("Готово")
    print(f"Все результаты в каталоге {out}/")


if __name__ == "__main__":
    main()
