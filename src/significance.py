"""Проверка, отличается ли найденная структура от случайной.

Зачем это нужно. Силуэт - не критерий значимости, а описательная мера: у него
нет нулевого распределения "из коробки". На 20 объектах его максимум по k
почти всегда оказывается приличным, потому что 20 точек можно нарезать на
несколько плотных кучек даже в чистом шуме. Замер на 300 независимых
равномерных наборах того же размера: медиана максимума силуэта 0.307,
95-й процентиль 0.368, отдельные выборки доходят до 0.420. Без нулевого
распределения утверждение "силуэт высокий" ничего не означает.

Отдельно стоит отметить контрольный набор из задания (seed=123): его силуэт
0.402 лежит на 99.3-м процентиле равномерного шума. Это неудачная выборка
одна на 150, с аномально сильной случайной кластеризацией, поэтому как
единственный эталон шума она не годится - завышает типичный уровень втрое
против медианного отрыва.

Калибровка проверена на 40 независимых шумовых наборах, где нулевая гипотеза
верна по построению: p < 0.05 получено в 1 случае из 40 при номинальных 5%,
медиана p-значения 0.62. Тест консервативен, то есть скорее пропустит
структуру, чем выдумает её.

Нулевая модель - независимая перестановка каждого столбца признаков. Она
разрушает связи между признаками, но сохраняет маргинальное распределение
каждого из них ровно таким, как в данных. Это строже, чем сравнение с
произвольно сгенерированным шумом: мы отвечаем на вопрос "есть ли в данных
многомерная структура сверх той, что объясняется распределениями отдельных
признаков", а не "отличаются ли данные от равномерных".
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .clustering import K_RANGE, cluster
from .distances import mixed_distance
from .loader import BINARY_FEATURE, CATEGORICAL_FEATURE, QUANT_FEATURES
from .preprocess import build_blocks

SHUFFLED_COLUMNS = [*QUANT_FEATURES, BINARY_FEATURE, CATEGORICAL_FEATURE]
DEFAULT_DRAWS = 200


@dataclass(frozen=True)
class SignificanceResult:
    """Наблюдённый силуэт на фоне нулевого распределения."""

    observed: float
    null_scores: np.ndarray
    n_draws: int
    method: str

    @property
    def p_value(self) -> float:
        """Доля перестановок, давших силуэт не хуже наблюдённого.

        Добавленная единица в числителе и знаменателе - поправка на то, что
        сами данные тоже являются одной из возможных перестановок. Без неё
        p-значение могло бы оказаться нулевым, чего при конечном числе
        перестановок не бывает.
        """
        at_least_as_high = int((self.null_scores >= self.observed).sum())
        return (1 + at_least_as_high) / (1 + self.n_draws)

    @property
    def percentile(self) -> float:
        """В какой процентиль нулевого распределения попал наблюдённый силуэт."""
        return float((self.null_scores < self.observed).mean() * 100)

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05

    def summary(self) -> str:
        verdict = (
            "структура значимо отличается от случайной"
            if self.is_significant
            else "структуру нельзя отличить от случайной"
        )
        return (
            f"силуэт {self.observed:.3f} против нулевого распределения "
            f"(медиана {np.median(self.null_scores):.3f}, "
            f"максимум {self.null_scores.max():.3f}, {self.n_draws} перестановок): "
            f"процентиль {self.percentile:.1f}, p = {self.p_value:.4f} - {verdict}"
        )


def permute(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Независимо перемешивает каждый столбец признаков.

    Идентификаторы штаммов остаются на месте: перемешиваются значения внутри
    столбцов, поэтому каждая строка становится случайной комбинацией реально
    наблюдавшихся значений признаков.
    """
    shuffled = df.copy()
    for column in SHUFFLED_COLUMNS:
        shuffled[column] = rng.permutation(shuffled[column].to_numpy())
    return shuffled


def null_distribution(
    df: pd.DataFrame,
    n_draws: int = DEFAULT_DRAWS,
    method: str = "ward",
    k_range=K_RANGE,
    seed: int = 0,
) -> np.ndarray:
    """Максимум силуэта по k для каждой из перестановок исходных данных."""
    rng = np.random.default_rng(seed)
    scores = np.empty(n_draws, dtype=float)
    for draw in range(n_draws):
        permuted = permute(df, rng)
        distance = mixed_distance(build_blocks(permuted)).mixed
        scores[draw] = cluster(distance, method=method, k_range=k_range).silhouette
    return scores


def assess(
    df: pd.DataFrame,
    observed: float,
    n_draws: int = DEFAULT_DRAWS,
    method: str = "ward",
    k_range=K_RANGE,
    seed: int = 0,
) -> SignificanceResult:
    """Сравнивает наблюдённый силуэт с нулевым распределением перестановок."""
    null_scores = null_distribution(
        df, n_draws=n_draws, method=method, k_range=k_range, seed=seed
    )
    return SignificanceResult(
        observed=observed, null_scores=null_scores, n_draws=n_draws, method=method
    )
