"""Предобработка: стандартизация количественных признаков и one-hot категории.

Признаки трёх разных типов дальше обрабатываются разными метриками, поэтому
здесь они не склеиваются в одну матрицу, а раскладываются по блокам.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loader import (
    BINARY_FEATURE,
    CATEGORICAL_FEATURE,
    QUANT_FEATURES,
    SOURCE_CATEGORIES,
    STRAIN_ID,
)


@dataclass(frozen=True)
class FeatureBlocks:
    """Признаки, разложенные по типам, плюс всё нужное для интерпретации."""

    strain_ids: np.ndarray
    quantitative: np.ndarray       # стандартизованные, (n, 5)
    binary: np.ndarray             # (n, 1)
    categorical: np.ndarray        # one-hot, (n, 3)
    quant_names: list
    categorical_names: list
    means: np.ndarray              # средние исходных признаков
    deviations: np.ndarray         # стандартные отклонения исходных признаков
    raw_quantitative: np.ndarray   # исходные единицы, для профилей кластеров

    @property
    def n_strains(self) -> int:
        return len(self.strain_ids)


def standardize(values: np.ndarray):
    """Z-преобразование: вычесть среднее, разделить на стандартное отклонение.

    Знаменатель n (ddof=0), как в sklearn.preprocessing.StandardScaler.
    Константный признак имеет нулевое отклонение - такой столбец обнуляется,
    а не превращается в бесконечность: признак, одинаковый у всех штаммов,
    не несёт информации о различиях и не должен влиять на расстояния.
    """
    values = np.asarray(values, dtype=float)
    means = values.mean(axis=0)
    deviations = values.std(axis=0, ddof=0)
    safe = np.where(deviations > 0, deviations, 1.0)
    standardized = (values - means) / safe
    standardized[:, deviations == 0] = 0.0
    return standardized, means, deviations


def one_hot(labels, categories=SOURCE_CATEGORIES):
    """One-hot кодировка по заранее заданному списку категорий.

    Первая категория НЕ отбрасывается: евклидово расстояние по полной кодировке
    одинаково для любой пары различных категорий (корень из двух), а усечённая
    кодировка делает одну категорию ближе к остальным без всякого основания.
    """
    labels = np.asarray(labels)
    encoded = np.zeros((len(labels), len(categories)), dtype=float)
    for position, category in enumerate(categories):
        encoded[labels == category, position] = 1.0
    return encoded, list(categories)


def build_blocks(df: pd.DataFrame) -> FeatureBlocks:
    """Раскладывает проверенную таблицу штаммов по блокам признаков."""
    raw = df[QUANT_FEATURES].to_numpy(dtype=float)
    standardized, means, deviations = standardize(raw)
    encoded, categorical_names = one_hot(df[CATEGORICAL_FEATURE])

    return FeatureBlocks(
        strain_ids=df[STRAIN_ID].to_numpy(),
        quantitative=standardized,
        binary=df[[BINARY_FEATURE]].to_numpy(dtype=float),
        categorical=encoded,
        quant_names=list(QUANT_FEATURES),
        categorical_names=categorical_names,
        means=means,
        deviations=deviations,
        raw_quantitative=raw,
    )
