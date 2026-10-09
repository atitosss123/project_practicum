"""Устойчивость разбиения к выбору весов.

Веса 0.6 / 0.2 / 0.2 заданы в условии без обоснования, поэтому нужно знать,
насколько от них зависит ответ. Измерение показывает, что зависимость
решающая, а сами веса стоят на точке перелома.

Причина - в устройстве компонент. Разрыв между экотипами по пяти
количественным признакам составляет 0.265 от итогового расстояния (медиана
по парам разных экотипов минус медиана по парам одного экотипа). Плоский
штраф за несовпадение источника даёт 0.200, то есть 75% всей разделяющей
способности количественных признаков, и не зависит от того, насколько штаммы
на самом деле похожи. Два таких штрафа вместе дают 0.400 - в полтора раза
больше разрыва. Поэтому штамм с ложной меткой источника отъезжает почти
настолько же, насколько отъезжал бы представитель другого экотипа.

Отдельный вывод из этого же замера: силуэт не является мерой правильности.
При весах 0.5 / 0.25 / 0.25 силуэт выше (0.570 против 0.494), а совпадение с
настоящими экотипами хуже (0.56 против 0.70). Подбор весов по максимуму
силуэта ухудшил бы ответ.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .clustering import cluster
from .distances import mixed_distance
from .loader import CATEGORICAL_FEATURE
from .preprocess import FeatureBlocks

# Доли, которые по очереди отдаются бинарному и категориальному признакам
# (поровну), остаток достаётся количественным.
DEFAULT_SHARES = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)
ASSIGNMENT_SHARE = 0.20


def weights_for(share: float) -> dict:
    """Веса при заданной доле на каждый из двух категориальных признаков."""
    return {
        "quantitative": 1.0 - 2.0 * share,
        "binary": share,
        "categorical": share,
    }


def separating_gap(blocks: FeatureBlocks, group_labels, weight: float = 0.6) -> dict:
    """Насколько количественные признаки разделяют группы, в единицах расстояния.

    Сравнивается с плоским штрафом, который даёт одна категориальная
    компонента: он равен её весу, потому что после нормировки компонента
    принимает только значения 0 и 1.
    """
    quantitative = mixed_distance(
        blocks, {"quantitative": 1.0, "binary": 0.0, "categorical": 0.0}
    ).quantitative
    labels = np.asarray(group_labels)
    upper = np.triu_indices(len(labels), 1)
    same_group = (labels[:, None] == labels[None, :])[upper]
    values = quantitative[upper]

    within = weight * float(np.median(values[same_group]))
    between = weight * float(np.median(values[~same_group]))
    return {
        "median_within_group": round(within, 3),
        "median_between_groups": round(between, 3),
        "separating_gap": round(between - within, 3),
    }


def sweep(
    blocks: FeatureBlocks,
    df: pd.DataFrame,
    truth: pd.DataFrame = None,
    shares=DEFAULT_SHARES,
) -> pd.DataFrame:
    """Разбиение при разных весах: что находится и чему оно соответствует."""
    rows = []
    for share in shares:
        weights = weights_for(share)
        result = cluster(mixed_distance(blocks, weights).mixed)
        row = {
            "quantitative": round(weights["quantitative"], 2),
            "binary": round(share, 2),
            "categorical": round(share, 2),
            "best_k": result.best_k,
            "silhouette": round(result.silhouette, 3),
            "ari_vs_source": round(
                adjusted_rand_score(df[CATEGORICAL_FEATURE], result.labels), 2
            ),
            "is_assignment": bool(np.isclose(share, ASSIGNMENT_SHARE)),
        }
        if truth is not None:
            row["ari_vs_ecotype"] = round(
                adjusted_rand_score(truth["ecotype"], result.labels), 2
            )
        rows.append(row)

    columns = ["quantitative", "binary", "categorical", "best_k", "silhouette"]
    if truth is not None:
        columns.append("ari_vs_ecotype")
    columns += ["ari_vs_source", "is_assignment"]
    return pd.DataFrame(rows)[columns]


def tipping_point(table: pd.DataFrame) -> str:
    """Короткий вывод: где разбиение перестаёт следовать количественным признакам."""
    if "ari_vs_ecotype" not in table.columns:
        return "скрытая истина не задана, вывод о точке перелома невозможен"

    perfect = table.loc[table.ari_vs_ecotype >= 0.99, "categorical"]
    assignment = table.loc[table.is_assignment]
    if perfect.empty or assignment.empty:
        return "точка перелома в проверенном диапазоне не обнаружена"

    row = assignment.iloc[0]
    return (
        f"экотипы восстанавливаются полностью при весе категориальных признаков "
        f"до {perfect.max():.2f}; на весах из задания ({row.categorical:.2f}) "
        f"совпадение падает до {row.ari_vs_ecotype:.2f}, а совпадение с заявленным "
        f"источником составляет {row.ari_vs_source:.2f}"
    )
