import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def three_strains():
    """Ручной пример: три штамма, все пять признаков равноотстоящие.

    Равный шаг внутри каждого признака даёт одинаковые z-оценки по всем пяти
    столбцам: -1.224745, 0, +1.224745. Поэтому евклидово расстояние по
    количественным признакам равно корню из пяти, умноженному на разность
    z-оценок, и весь расчёт проверяется на бумаге.
    """
    return pd.DataFrame(
        {
            "strain_id": ["A", "B", "C"],
            "growth_rate": [0.8, 1.0, 1.2],
            "product_yield": [2.0, 2.5, 3.0],
            "pH_tolerance": [6.0, 7.0, 8.0],
            "GC_content": [50.0, 55.0, 60.0],
            "genome_size_mbp": [3.0, 4.0, 5.0],
            "has_plasmid": [0, 1, 1],
            "source": ["soil", "water", "soil"],
        }
    )


@pytest.fixture
def three_strains_expected():
    """Расстояния для ручного примера, посчитанные вручную.

    Количественные: шаг z = 1.224745, по пяти признакам
        A-B = sqrt(5) * 1.224745 = 2.738613
        A-C = sqrt(5) * 2.449490 = 5.477226  <- максимум
        B-C = sqrt(5) * 1.224745 = 2.738613
      после нормировки: 0.5 / 1.0 / 0.5
    Бинарный (Хэмминг по одному признаку, значения 0/1/1): 1.0 / 1.0 / 0.0
    Категориальный (one-hot soil/water/soil, максимум sqrt(2)): 1.0 / 0.0 / 1.0
    Смесь 0.6 / 0.2 / 0.2:
        A-B = 0.6*0.5 + 0.2*1.0 + 0.2*1.0 = 0.7
        A-C = 0.6*1.0 + 0.2*1.0 + 0.2*0.0 = 0.8
        B-C = 0.6*0.5 + 0.2*0.0 + 0.2*1.0 = 0.5
    """
    return {
        "quantitative": np.array([[0.0, 0.5, 1.0], [0.5, 0.0, 0.5], [1.0, 0.5, 0.0]]),
        "binary": np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
        "categorical": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        "mixed": np.array([[0.0, 0.7, 0.8], [0.7, 0.0, 0.5], [0.8, 0.5, 0.0]]),
    }


@pytest.fixture
def three_blocks():
    """12 штаммов тремя заведомо разделёнными группами.

    Группы разнесены по количественным признакам настолько, что любой разумный
    метод обязан найти ровно три кластера. Нужны для проверки, что подбор k
    работает на однозначном случае - прежде чем доверять ему на спорном.
    """
    rng = np.random.default_rng(7)
    centers = {
        "gut": (0.7, 1.8, 5.5, 47.0, 3.2),
        "soil": (1.0, 2.8, 7.0, 55.0, 4.5),
        "water": (1.4, 3.8, 8.5, 63.0, 5.8),
    }
    rows = []
    for source, center in centers.items():
        for _ in range(4):
            rows.append(
                {
                    "growth_rate": center[0] + rng.normal(0, 0.01),
                    "product_yield": center[1] + rng.normal(0, 0.02),
                    "pH_tolerance": center[2] + rng.normal(0, 0.03),
                    "GC_content": center[3] + rng.normal(0, 0.1),
                    "genome_size_mbp": center[4] + rng.normal(0, 0.02),
                    "has_plasmid": 1 if source == "gut" else 0,
                    "source": source,
                }
            )
    df = pd.DataFrame(rows)
    df.insert(0, "strain_id", [f"T{i:02d}" for i in range(1, len(df) + 1)])
    return df
