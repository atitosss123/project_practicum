"""Иерархическая кластеризация и подбор числа кластеров по силуэту.

Метод Уорда применяется к готовой матрице расстояний, как требует задание.
Формула Ларда-Уильямса, по которой scipy наращивает дерево для Уорда,
предполагает евклидову метрику, а взвешенная смесь трёх расстояний ею не
является: "прирост внутригрупповой дисперсии" перестаёт быть тем, чем
называется. scipy это молча примет и выдаст число.

Поэтому тот же подбор прогоняется на average linkage, для которого
произвольная матрица расстояний корректна. Расхождение двух разбиений - это
и есть измеренная цена допущения.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import silhouette_score

K_RANGE = tuple(range(2, 11))

# Разница силуэта, в пределах которой разбиения считаются неразличимыми.
# На 20 объектах третий знак после запятой - шум, а не сигнал.
PLATEAU_TOLERANCE = 0.01


@dataclass(frozen=True)
class ClusteringResult:
    """Результат кластеризации при выбранном числе кластеров."""

    method: str
    best_k: int
    silhouette: float
    labels: np.ndarray
    scores: dict                      # k -> силуэт
    plateau: tuple = field(default=())  # все k, неотличимые от лучшего
    tree: np.ndarray = None

    @property
    def is_conclusive(self) -> bool:
        """Определено ли число кластеров однозначно.

        Если в пределах допуска с лучшим k соседствуют другие, argmax выбран
        шумом в третьем знаке, и подавать его как найденный ответ нельзя.
        """
        return len(self.plateau) == 1

    @property
    def cluster_sizes(self) -> dict:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))


def build_tree(distance: np.ndarray, method: str = "ward") -> np.ndarray:
    """Дерево иерархической кластеризации по полной матрице расстояний."""
    from scipy.spatial.distance import squareform

    return linkage(squareform(distance, checks=False), method=method)


def silhouette_profile(distance: np.ndarray, tree: np.ndarray, k_range=K_RANGE) -> dict:
    """Силуэт для каждого числа кластеров из диапазона.

    Силуэт считается на той же матрице расстояний, по которой строилось
    дерево (metric="precomputed"). Считать его на признаках было бы
    подменой: оценивали бы не ту геометрию, по которой кластеризовали.

    k, при котором разрез даёт не k групп, пропускается: такое значение
    нечего оценивать. Это не редкость, а воспроизводимое свойство scipy:
    fcluster с критерием maxclust возвращает k-1 групп вместо k ровно при
    k = n-1 (проверено для n от 3 до 20). При 20 штаммах это k=19, то есть
    за пределами диапазона из задания, но на малых наборах встречается.
    """
    scores = {}
    n_objects = distance.shape[0]
    for k in k_range:
        # k = n-1 отсекаем сразу: см. про maxclust выше.
        if k >= n_objects - 1:
            continue
        labels = fcluster(tree, k, criterion="maxclust")
        if len(np.unique(labels)) != k:
            continue
        scores[k] = float(silhouette_score(distance, labels, metric="precomputed"))
    if not scores:
        raise ValueError("ни одно число кластеров из диапазона не дало корректного разреза")
    return scores


def find_plateau(scores: dict, tolerance: float = PLATEAU_TOLERANCE) -> tuple:
    """Все k, чей силуэт отличается от максимального не более чем на допуск."""
    best = max(scores.values())
    return tuple(sorted(k for k, score in scores.items() if best - score <= tolerance))


def cluster(distance: np.ndarray, method: str = "ward", k_range=K_RANGE) -> ClusteringResult:
    """Полный подбор: дерево, профиль силуэта, выбор k, разметка штаммов."""
    tree = build_tree(distance, method=method)
    scores = silhouette_profile(distance, tree, k_range=k_range)
    best_k = max(scores, key=scores.get)
    return ClusteringResult(
        method=method,
        best_k=best_k,
        silhouette=scores[best_k],
        labels=fcluster(tree, best_k, criterion="maxclust"),
        scores=scores,
        plateau=find_plateau(scores),
        tree=tree,
    )


def labels_at(result: ClusteringResult, k: int) -> np.ndarray:
    """Разметка при произвольном числе кластеров на том же дереве."""
    return fcluster(result.tree, k, criterion="maxclust")
