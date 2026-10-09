"""Чтение и валидация таблицы штаммов.

Нарушение схемы - явная ошибка, а не тихое продолжение: дальше по пайплайну
некорректные данные превратятся в правдоподобные, но бессмысленные кластеры.
"""

from pathlib import Path

import pandas as pd

STRAIN_ID = "strain_id"

QUANT_FEATURES = [
    "growth_rate",
    "product_yield",
    "pH_tolerance",
    "GC_content",
    "genome_size_mbp",
]
BINARY_FEATURE = "has_plasmid"
CATEGORICAL_FEATURE = "source"

REQUIRED_COLUMNS = [STRAIN_ID, *QUANT_FEATURES, BINARY_FEATURE, CATEGORICAL_FEATURE]

# Полный список категорий задан явно, а не выводится из данных: иначе one-hot
# кодировка меняет размерность от набора к набору, и расстояния становятся
# несравнимыми между прогонами.
SOURCE_CATEGORIES = ["gut", "soil", "water"]

# Физически осмысленные границы. Выход за них означает ошибку в данных,
# а не экзотический штамм.
FEATURE_LIMITS = {
    "growth_rate": (0.0, 5.0),
    "product_yield": (0.0, 20.0),
    "pH_tolerance": (0.0, 14.0),
    "GC_content": (20.0, 80.0),
    "genome_size_mbp": (0.1, 15.0),
}


class SchemaError(ValueError):
    """Таблица не соответствует схеме задачи."""


def validate_strains(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет таблицу штаммов и возвращает её с приведёнными типами."""
    problems = []

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"отсутствуют обязательные колонки: {', '.join(missing)}")

    if df.empty:
        raise SchemaError("таблица пуста")

    duplicates = df[STRAIN_ID][df[STRAIN_ID].duplicated()].tolist()
    if duplicates:
        problems.append(f"повторяющиеся идентификаторы штаммов: {sorted(set(duplicates))}")

    for feature in QUANT_FEATURES:
        column = pd.to_numeric(df[feature], errors="coerce")
        if column.isna().any():
            bad = df.loc[column.isna(), STRAIN_ID].tolist()
            problems.append(f"{feature}: нечисловые или пустые значения у {bad}")
            continue
        low, high = FEATURE_LIMITS[feature]
        out_of_range = df.loc[~column.between(low, high), STRAIN_ID].tolist()
        if out_of_range:
            problems.append(
                f"{feature}: значения вне допустимого диапазона {low}-{high} у {out_of_range}"
            )
        df[feature] = column.astype(float)

    binary = pd.to_numeric(df[BINARY_FEATURE], errors="coerce")
    if not binary.isin([0, 1]).all():
        bad = df.loc[~binary.isin([0, 1]), STRAIN_ID].tolist()
        problems.append(f"{BINARY_FEATURE}: допустимы только 0 и 1, нарушено у {bad}")
    else:
        df[BINARY_FEATURE] = binary.astype(int)

    unknown = sorted(set(df[CATEGORICAL_FEATURE]) - set(SOURCE_CATEGORIES))
    if unknown:
        problems.append(
            f"{CATEGORICAL_FEATURE}: неизвестные источники {unknown}, "
            f"допустимы {SOURCE_CATEGORIES}"
        )

    if problems:
        raise SchemaError("таблица штаммов не прошла валидацию:\n  - " + "\n  - ".join(problems))

    return df.reset_index(drop=True)


def load_strains(path) -> pd.DataFrame:
    """Читает CSV со штаммами и проверяет схему."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"файл с данными не найден: {path}")
    return validate_strains(pd.read_csv(path))


def load_truth(path) -> pd.DataFrame:
    """Читает таблицу скрытой истины (настоящий экотип каждого штамма).

    Используется только для оценки качества разбиения, в кластеризацию не идёт.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"файл со скрытой истиной не найден: {path}")
    truth = pd.read_csv(path)
    missing = [c for c in (STRAIN_ID, "ecotype") if c not in truth.columns]
    if missing:
        raise SchemaError(f"в таблице истины нет колонок: {', '.join(missing)}")
    return truth
