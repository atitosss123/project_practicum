"""Профили кластеров и проверки, что разбиение означает.

Средние значения выводятся в ИСХОДНЫХ единицах, а не в стандартизованных:
проценты GC и мегабазы читаются осмысленно, z-оценки - нет.

Кроме описания кластеров здесь живут две проверки, без которых разбиение
легко перетолковать:

  - совпадение с комбинацией "источник x плазмида". Две из трёх компонент
    расстояния ступенчаты и в сумме дают 40% веса, поэтому кластеры могут
    оказаться простым пересказом этих двух признаков. Это надо измерить, а не
    предполагать;
  - судьба штаммов с несовпадающим источником. Если количественные признаки с
    весом 0.6 действительно перебивают ложную категорию, такой штамм попадёт
    в кластер своего экотипа, а не своего источника.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .loader import BINARY_FEATURE, CATEGORICAL_FEATURE, QUANT_FEATURES, STRAIN_ID

CLUSTER_COLUMN = "cluster"


def assignments(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Таблица (strain_id, cluster) - основной требуемый результат."""
    return pd.DataFrame({STRAIN_ID: df[STRAIN_ID].to_numpy(), CLUSTER_COLUMN: labels})


def cluster_summary(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Средние количественные признаки, мода источника и доля плазмид по кластерам."""
    table = df.copy()
    table[CLUSTER_COLUMN] = labels

    summary = table.groupby(CLUSTER_COLUMN)[QUANT_FEATURES].mean().round(3)
    summary.insert(0, "n_strains", table.groupby(CLUSTER_COLUMN).size())

    grouped = table.groupby(CLUSTER_COLUMN)
    summary["plasmid_share"] = grouped[BINARY_FEATURE].mean().round(2)
    summary["source_mode"] = grouped[CATEGORICAL_FEATURE].agg(
        lambda values: values.mode().iat[0]
    )
    # Насколько мода доминирует: без этого "преимущественно почва" может
    # означать и 7 из 7, и 3 из 7.
    summary["source_mode_share"] = grouped[CATEGORICAL_FEATURE].agg(
        lambda values: values.value_counts(normalize=True).iat[0]
    ).round(2)
    summary["source_mix"] = grouped[CATEGORICAL_FEATURE].agg(
        lambda values: ", ".join(
            f"{source} {count}" for source, count in values.value_counts().items()
        )
    )
    return summary.reset_index()


def categorical_echo(df: pd.DataFrame, labels: np.ndarray) -> dict:
    """Насколько разбиение пересказывает категориальные признаки.

    Высокое совпадение означает, что кластеры воспроизводят комбинацию
    "источник x плазмида", то есть ступенчатые 40% веса перетянули результат,
    а количественные признаки роли почти не играют.
    """
    source = df[CATEGORICAL_FEATURE].astype(str)
    plasmid = df[BINARY_FEATURE].astype(str)
    return {
        "source_only": adjusted_rand_score(source, labels),
        "plasmid_only": adjusted_rand_score(plasmid, labels),
        "source_x_plasmid": adjusted_rand_score(source + "/" + plasmid, labels),
    }


def agreement_with_truth(truth: pd.DataFrame, labels: np.ndarray) -> dict:
    """Совпадение разбиения со скрытой истиной (заложенными экотипами)."""
    return {
        "ari": adjusted_rand_score(truth["ecotype"], labels),
        "crosstab": pd.crosstab(
            truth["ecotype"], pd.Series(labels, name=CLUSTER_COLUMN)
        ),
    }


def mismatched_strains(df: pd.DataFrame, truth: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Штаммы, чей источник не совпадает с экотипом, и куда они попали.

    Колонка followed_ecotype отвечает на вопрос, перебили ли количественные
    признаки ложную категорию: True означает, что штамм оказался в кластере,
    где преобладает его настоящий экотип, а не заявленный источник.
    """
    if "source_mismatch" not in truth.columns:
        return pd.DataFrame()

    merged = df[[STRAIN_ID, CATEGORICAL_FEATURE]].merge(truth, on=STRAIN_ID)
    merged[CLUSTER_COLUMN] = labels

    # Для каждого кластера - преобладающий настоящий экотип.
    dominant = (
        merged.groupby(CLUSTER_COLUMN)["ecotype"]
        .agg(lambda values: values.mode().iat[0])
        .to_dict()
    )
    merged["cluster_ecotype"] = merged[CLUSTER_COLUMN].map(dominant)
    merged["followed_ecotype"] = merged["ecotype"] == merged["cluster_ecotype"]

    return merged.loc[
        merged["source_mismatch"],
        [STRAIN_ID, "ecotype", CATEGORICAL_FEATURE, CLUSTER_COLUMN,
         "cluster_ecotype", "followed_ecotype"],
    ].reset_index(drop=True)


def describe_clusters(summary: pd.DataFrame) -> str:
    """Текстовое описание кластеров - фолбэк, когда графика недоступна."""
    lines = []
    for row in summary.itertuples(index=False):
        traits = ", ".join(
            f"{name} {getattr(row, name):.2f}" for name in QUANT_FEATURES
        )
        lines.append(
            f"Кластер {getattr(row, CLUSTER_COLUMN)}: {row.n_strains} штаммов. "
            f"{traits}. Плазмида у {row.plasmid_share:.0%}. "
            f"Источник преимущественно {row.source_mode} "
            f"({row.source_mode_share:.0%}; состав: {row.source_mix})."
        )
    return "\n".join(lines)
