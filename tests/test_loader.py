from pathlib import Path

import pytest

from src.loader import (
    REQUIRED_COLUMNS,
    SchemaError,
    load_strains,
    load_truth,
    validate_strains,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def test_real_datasets_pass_validation():
    for name in ("strains_features.csv", "strains_uniform.csv"):
        df = load_strains(DATA_DIR / name)
        assert len(df) == 20
        assert list(df.columns) == REQUIRED_COLUMNS


def test_truth_table_matches_the_main_dataset():
    features = load_strains(DATA_DIR / "strains_features.csv")
    truth = load_truth(DATA_DIR / "strains_truth.csv")
    assert list(truth.strain_id) == list(features.strain_id)
    assert set(truth.ecotype) == {"gut", "soil", "water"}


def test_clean_table_passes(three_strains):
    assert len(validate_strains(three_strains)) == 3


def test_missing_column_is_rejected(three_strains):
    with pytest.raises(SchemaError, match="отсутствуют обязательные колонки"):
        validate_strains(three_strains.drop(columns=["GC_content"]))


def test_duplicate_strain_id_is_rejected(three_strains):
    three_strains.loc[2, "strain_id"] = "A"
    with pytest.raises(SchemaError, match="повторяющиеся идентификаторы"):
        validate_strains(three_strains)


def test_non_binary_plasmid_flag_is_rejected(three_strains):
    three_strains.loc[1, "has_plasmid"] = 2
    with pytest.raises(SchemaError, match="допустимы только 0 и 1"):
        validate_strains(three_strains)


def test_unknown_source_is_rejected(three_strains):
    three_strains.loc[0, "source"] = "sediment"
    with pytest.raises(SchemaError, match="неизвестные источники"):
        validate_strains(three_strains)


def test_out_of_range_value_is_rejected(three_strains):
    """GC-содержание 150% - ошибка в данных, а не экзотический штамм."""
    three_strains.loc[1, "GC_content"] = 150.0
    with pytest.raises(SchemaError, match="вне допустимого диапазона"):
        validate_strains(three_strains)


def test_non_numeric_value_is_rejected(three_strains):
    three_strains["growth_rate"] = three_strains["growth_rate"].astype(object)
    three_strains.loc[1, "growth_rate"] = "быстрый"
    with pytest.raises(SchemaError, match="нечисловые или пустые"):
        validate_strains(three_strains)


def test_missing_value_is_rejected(three_strains):
    three_strains.loc[1, "product_yield"] = None
    with pytest.raises(SchemaError, match="нечисловые или пустые"):
        validate_strains(three_strains)


def test_empty_table_is_rejected(three_strains):
    with pytest.raises(SchemaError, match="таблица пуста"):
        validate_strains(three_strains.iloc[0:0])


def test_all_problems_are_reported_at_once(three_strains):
    """Валидация не останавливается на первой ошибке: исправлять по одной долго."""
    three_strains.loc[2, "strain_id"] = "A"
    three_strains.loc[0, "source"] = "sediment"
    three_strains.loc[1, "GC_content"] = 150.0
    with pytest.raises(SchemaError) as error:
        validate_strains(three_strains)
    message = str(error.value)
    assert "повторяющиеся идентификаторы" in message
    assert "неизвестные источники" in message
    assert "вне допустимого диапазона" in message


def test_missing_file_is_reported_clearly():
    with pytest.raises(FileNotFoundError, match="не найден"):
        load_strains(DATA_DIR / "нет-такого-файла.csv")
