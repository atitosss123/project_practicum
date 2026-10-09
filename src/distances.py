"""Смешанная матрица расстояний между штаммами.

Схема из задания: для количественных признаков евклидово расстояние, для
бинарного - Хэмминга, для категориального (one-hot) - евклидово. Каждая
компонента нормируется на свой максимум, затем складывается с весами
0.6 / 0.2 / 0.2.

Важное свойство этой схемы: из трёх компонент две вырождены. Хэмминг по одному
бинарному признаку даёт только 0 или 1; евклид по one-hot из трёх категорий -
только 0 или корень из двух. После нормировки обе дают ровно два значения, 0 и
1. То есть 40% итогового расстояния - ступенчатая функция, и непрерывную
геометрию задают исключительно количественные признаки с весом 0.6.
Практическое следствие: кластеры тяготеют к разбиению по комбинации
"источник x плазмида", и это нужно проверять явно, а не обнаруживать потом.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist, squareform

from .preprocess import FeatureBlocks

DEFAULT_WEIGHTS = {"quantitative": 0.6, "binary": 0.2, "categorical": 0.2}


@dataclass(frozen=True)
class DistanceParts:
    """Итоговая матрица и три компоненты, из которых она сложена."""

    mixed: np.ndarray
    quantitative: np.ndarray
    binary: np.ndarray
    categorical: np.ndarray
    weights: dict

    def condensed(self) -> np.ndarray:
        """Сжатая форма - то, что принимает scipy.cluster.hierarchy.linkage."""
        return squareform(self.mixed, checks=False)


def normalize_by_max(matrix: np.ndarray) -> np.ndarray:
    """Делит матрицу на её максимум, приводя значения к отрезку [0, 1].

    Нулевой максимум возможен и законен: признак оказался одинаковым у всех
    штаммов. Делить в этом случае не на что - матрица уже нулевая.
    """
    matrix = np.asarray(matrix, dtype=float)
    largest = matrix.max() if matrix.size else 0.0
    return matrix / largest if largest > 0 else matrix


def pairwise(values: np.ndarray, metric: str) -> np.ndarray:
    """Полная (не сжатая) матрица попарных расстояний."""
    return squareform(pdist(np.asarray(values, dtype=float), metric=metric))


def validate_weights(weights: dict) -> dict:
    """Веса должны покрывать три компоненты и давать в сумме единицу."""
    expected = set(DEFAULT_WEIGHTS)
    if set(weights) != expected:
        raise ValueError(f"ожидались веса для {sorted(expected)}, получено {sorted(weights)}")
    if any(w < 0 for w in weights.values()):
        raise ValueError(f"веса не могут быть отрицательными: {weights}")
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"сумма весов должна равняться 1, получено {total}")
    return dict(weights)


def mixed_distance(blocks: FeatureBlocks, weights: dict = None) -> DistanceParts:
    """Взвешенная смесь трёх расстояний по схеме задания."""
    weights = validate_weights(weights or DEFAULT_WEIGHTS)

    quantitative = normalize_by_max(pairwise(blocks.quantitative, "euclidean"))
    binary = normalize_by_max(pairwise(blocks.binary, "hamming"))
    categorical = normalize_by_max(pairwise(blocks.categorical, "euclidean"))

    mixed = (
        weights["quantitative"] * quantitative
        + weights["binary"] * binary
        + weights["categorical"] * categorical
    )
    # Асимметрия и ненулевая диагональ здесь возможны только из-за ошибки
    # округления; приводим явно, иначе scipy.squareform откажется принимать.
    mixed = (mixed + mixed.T) / 2.0
    np.fill_diagonal(mixed, 0.0)

    return DistanceParts(
        mixed=mixed,
        quantitative=quantitative,
        binary=binary,
        categorical=categorical,
        weights=weights,
    )
