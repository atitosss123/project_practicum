import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from src.loader import QUANT_FEATURES, SOURCE_CATEGORIES
from src.preprocess import build_blocks, one_hot, standardize


def test_standardize_gives_zero_mean_and_unit_deviation(three_strains):
    standardized, _, _ = standardize(three_strains[QUANT_FEATURES].to_numpy())
    assert np.allclose(standardized.mean(axis=0), 0.0)
    assert np.allclose(standardized.std(axis=0, ddof=0), 1.0)


def test_standardize_matches_sklearn(three_strains):
    """Своя реализация должна совпадать с StandardScaler: тот же знаменатель n."""
    values = three_strains[QUANT_FEATURES].to_numpy()
    assert np.allclose(standardize(values)[0], StandardScaler().fit_transform(values))


def test_standardize_keeps_known_z_values(three_strains):
    """Равный шаг внутри признака даёт z = -1.224745 / 0 / +1.224745."""
    standardized, _, _ = standardize(three_strains[QUANT_FEATURES].to_numpy())
    expected = np.array([-1.224745, 0.0, 1.224745])
    for column in range(standardized.shape[1]):
        assert np.allclose(standardized[:, column], expected, atol=1e-6)


def test_constant_feature_becomes_zero_not_infinity():
    """Признак, одинаковый у всех штаммов, обнуляется и не даёт NaN."""
    values = np.array([[1.0, 7.0], [2.0, 7.0], [3.0, 7.0]])
    standardized, _, deviations = standardize(values)
    assert np.isfinite(standardized).all()
    assert np.allclose(standardized[:, 1], 0.0)
    assert deviations[1] == 0.0


def test_one_hot_uses_fixed_categories_even_if_unused():
    """Отсутствующая в данных категория даёт нулевой столбец, а не исчезает."""
    encoded, names = one_hot(np.array(["soil", "water", "soil"]))
    assert names == SOURCE_CATEGORIES
    assert encoded.shape == (3, 3)
    assert np.allclose(encoded[:, names.index("gut")], 0.0)
    assert np.allclose(encoded.sum(axis=1), 1.0)


def test_one_hot_distance_is_equal_for_any_distinct_pair():
    """Полная кодировка: любые две разные категории равноудалены."""
    encoded, _ = one_hot(np.array(SOURCE_CATEGORIES))
    pairs = [(0, 1), (0, 2), (1, 2)]
    distances = [np.linalg.norm(encoded[i] - encoded[j]) for i, j in pairs]
    assert np.allclose(distances, np.sqrt(2))


def test_build_blocks_shapes_and_raw_values(three_strains):
    blocks = build_blocks(three_strains)
    assert blocks.n_strains == 3
    assert blocks.quantitative.shape == (3, 5)
    assert blocks.binary.shape == (3, 1)
    assert blocks.categorical.shape == (3, 3)
    # Исходные единицы сохранены для профилей кластеров.
    assert np.allclose(blocks.raw_quantitative[:, 0], [0.8, 1.0, 1.2])
    assert np.allclose(blocks.means[0], 1.0)
