"""Визуализация результата: дендрограмма, PCA и MDS.

Три представления отвечают на разные вопросы, поэтому строятся все три.

Дендрограмма показывает иерархию и то, насколько произволен выбранный разрез.
Для набора, где силуэт даёт плато, это главная картинка: видно, что разрез
между двумя и тремя кластерами держится на малой разнице высот.

Главные компоненты (PCA) требует задание. Важная оговорка: PCA работает с
признаками, а кластеры построены по взвешенной матрице расстояний. Это разные
геометрии, поэтому кластеры на PCA могут выглядеть перемешанными даже при
корректном разбиении, и принимать такую картинку за признак ошибки нельзя.

Многомерное шкалирование (MDS) по нашей матрице расхождения не даёт: оно
проецирует ровно то расстояние, по которому кластеризовали. Поэтому именно
MDS - честная картинка разбиения, а PCA - требуемая.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # файлы, без интерактивного окна
import matplotlib.pyplot as plt  # noqa: E402
from scipy.cluster.hierarchy import dendrogram  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.manifold import MDS  # noqa: E402

PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
           "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
FIGURE_SIZE = (9, 6)
DPI = 150


def _colors_for(labels):
    unique = sorted(set(labels))
    mapping = {label: PALETTE[i % len(PALETTE)] for i, label in enumerate(unique)}
    return [mapping[label] for label in labels], mapping


def pca_coordinates(blocks, n_components: int = 2):
    """Координаты в пространстве первых главных компонент и объяснённая дисперсия.

    PCA считается по стандартизованным количественным признакам: смешивать в
    одну матрицу признаки, для которых задание требует разных метрик, значило
    бы считать не ту геометрию.
    """
    model = PCA(n_components=n_components, random_state=0)
    coordinates = model.fit_transform(blocks.quantitative)
    return coordinates, model.explained_variance_ratio_


def mds_coordinates(distance: np.ndarray, n_components: int = 2):
    """Координаты MDS по готовой матрице расстояний и напряжение Краскала.

    Возвращается нормированное напряжение (stress-1), а не атрибут stress_
    модели: при метрическом MDS sklearn кладёт туда сырую сумму квадратов
    отклонений, которую нельзя ни с чем сравнить. Нормированный показатель
    читается по общепринятой шкале: ниже 0.05 - отличная проекция, до 0.10 -
    хорошая, выше 0.20 - проекции доверять нельзя.
    """
    from scipy.spatial.distance import pdist, squareform

    model = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=0,
        normalized_stress=False,
    )
    coordinates = model.fit_transform(distance)

    projected = squareform(pdist(coordinates, "euclidean"))
    upper = np.triu_indices(len(distance), 1)
    original = distance[upper]
    stress = float(
        np.sqrt(((original - projected[upper]) ** 2).sum() / (original ** 2).sum())
    )
    return coordinates, stress


def coordinates_table(blocks, distance: np.ndarray) -> pd.DataFrame:
    """Обе проекции в одной таблице - требуемый выходной артефакт."""
    pca, variance = pca_coordinates(blocks)
    mds, _ = mds_coordinates(distance)
    return pd.DataFrame(
        {
            "strain_id": blocks.strain_ids,
            "pc1": pca[:, 0].round(4),
            "pc2": pca[:, 1].round(4),
            "mds1": mds[:, 0].round(4),
            "mds2": mds[:, 1].round(4),
        }
    ), variance


def plot_dendrogram(tree, strain_ids, best_k: int, path, plateau=()) -> Path:
    """Дендрограмма с линией разреза при выбранном числе кластеров."""
    # Высота разреза - между k-м и (k-1)-м объединением, считая с конца:
    # ниже неё дерево распадается ровно на best_k ветвей.
    heights = np.sort(tree[:, 2])
    cut = (heights[-best_k] + heights[-best_k + 1]) / 2 if best_k > 1 else heights[-1]

    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    # Порог окраски совпадает с разрезом, иначе scipy красит по своему
    # умолчанию и цвета ветвей не отвечают найденным кластерам.
    dendrogram(tree, labels=list(strain_ids), ax=axes, color_threshold=cut,
               above_threshold_color="#8C8C8C")
    axes.axhline(cut, color="#C44E52", linestyle="--", linewidth=1.2,
                 label=f"разрез на {best_k} кластера")

    title = "Дендрограмма (метод Уорда по смешанной матрице расстояний)"
    if len(plateau) > 1:
        title += f"\nсилуэт не различает k = {', '.join(map(str, plateau))}"
    axes.set_title(title)
    axes.set_xlabel("штамм")
    axes.set_ylabel("расстояние объединения")
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return Path(path)


def plot_projection(coordinates, labels, strain_ids, path, title, axis_labels) -> Path:
    """Точечная диаграмма проекции с окраской по кластерам."""
    colors, mapping = _colors_for(labels)
    figure, axes = plt.subplots(figsize=FIGURE_SIZE)
    axes.scatter(coordinates[:, 0], coordinates[:, 1], c=colors, s=90,
                 edgecolor="white", linewidth=1.2, zorder=3)

    # Подпись ставится рядом с точкой, а не внутри: внутри маркера текст из
    # трёх знаков обрезается, и штаммы перестают различаться.
    for (x, y), name in zip(coordinates, strain_ids):
        axes.annotate(name, (x, y), textcoords="offset points", xytext=(7, 5),
                      fontsize=7.5, color="#333333", zorder=4)

    for label, color in mapping.items():
        axes.scatter([], [], c=color, s=100, label=f"кластер {label}")

    axes.set_title(title)
    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.axhline(0, color="#DDDDDD", linewidth=0.8, zorder=1)
    axes.axvline(0, color="#DDDDDD", linewidth=0.8, zorder=1)
    axes.legend(loc="best", framealpha=0.9)
    axes.margins(0.12)
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return Path(path)


def plot_silhouette_profile(scores: dict, best_k: int, path, plateau=()) -> Path:
    """Профиль силуэта по k - показывает плато, а не одну точку."""
    figure, axes = plt.subplots(figsize=(8, 4.5))
    ks = list(scores)
    values = [scores[k] for k in ks]
    colors = ["#C44E52" if k in plateau else "#4C72B0" for k in ks]
    axes.bar(ks, values, color=colors, width=0.65)
    axes.set_xticks(ks)
    axes.set_ylim(0, max(values) * 1.18)
    for k, value in zip(ks, values):
        axes.text(k, value + max(values) * 0.02, f"{value:.3f}",
                  ha="center", fontsize=8)

    subtitle = (f"выделены k, неотличимые от лучшего: {', '.join(map(str, plateau))}"
                if len(plateau) > 1 else f"максимум однозначен при k = {best_k}")
    axes.set_title(f"Индекс силуэта по числу кластеров\n{subtitle}")
    axes.set_xlabel("число кластеров k")
    axes.set_ylabel("силуэт")
    figure.tight_layout()
    figure.savefig(path, dpi=DPI)
    plt.close(figure)
    return Path(path)


def describe_projection(coordinates, labels, strain_ids, variance=None) -> str:
    """Текстовое описание проекции - фолбэк, когда графика недоступна."""
    lines = []
    if variance is not None:
        total = float(np.sum(variance)) * 100
        lines.append(f"Первые две компоненты объясняют {total:.1f}% дисперсии "
                     f"({variance[0] * 100:.1f}% и {variance[1] * 100:.1f}%).")
    for label in sorted(set(labels)):
        mask = np.asarray(labels) == label
        center = coordinates[mask].mean(axis=0)
        members = ", ".join(np.asarray(strain_ids)[mask])
        lines.append(
            f"Кластер {label}: центр ({center[0]:+.2f}, {center[1]:+.2f}), "
            f"штаммы {members}."
        )
    return "\n".join(lines)
