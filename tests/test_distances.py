import numpy as np
import pytest
from scipy.spatial.distance import squareform

from src.distances import (
    DEFAULT_WEIGHTS,
    mixed_distance,
    normalize_by_max,
    validate_weights,
)
from src.preprocess import build_blocks


def test_matches_hand_calculation(three_strains, three_strains_expected):
    """Главная проверка: весь расчёт сверен с ручным."""
    parts = mixed_distance(build_blocks(three_strains))
    for component, expected in three_strains_expected.items():
        assert np.allclose(getattr(parts, component), expected, atol=1e-6), component


def test_matrix_is_a_valid_distance_matrix(three_strains):
    parts = mixed_distance(build_blocks(three_strains))
    assert np.allclose(parts.mixed, parts.mixed.T)
    assert np.allclose(np.diag(parts.mixed), 0.0)
    assert parts.mixed.min() >= 0.0
    assert parts.mixed.max() <= 1.0


def test_every_component_is_normalized_to_unit_maximum(three_strains):
    parts = mixed_distance(build_blocks(three_strains))
    for component in ("quantitative", "binary", "categorical"):
        assert np.isclose(getattr(parts, component).max(), 1.0), component


def test_degenerate_components_take_only_two_values(three_strains):
    """Бинарная и категориальная компоненты дают ровно 0 или 1.

    Это и есть та ступенчатость, из-за которой 40% итогового расстояния
    не несут непрерывной геометрии.
    """
    parts = mixed_distance(build_blocks(three_strains))
    for component in ("binary", "categorical"):
        values = np.unique(np.round(getattr(parts, component), 9))
        assert set(values) <= {0.0, 1.0}, (component, values)


def test_condensed_form_is_accepted_by_scipy(three_strains):
    parts = mixed_distance(build_blocks(three_strains))
    condensed = parts.condensed()
    assert condensed.shape == (3,)
    assert np.allclose(squareform(condensed), parts.mixed)


def test_normalize_by_max_leaves_zero_matrix_alone():
    """Константный признак даёт нулевую матрицу: делить не на что, NaN недопустим."""
    zeros = np.zeros((4, 4))
    result = normalize_by_max(zeros)
    assert np.isfinite(result).all()
    assert np.allclose(result, 0.0)


def test_constant_binary_feature_contributes_nothing(three_strains):
    """Если плазмида есть у всех, бинарная компонента нулевая, а не NaN."""
    df = three_strains.copy()
    df["has_plasmid"] = 1
    parts = mixed_distance(build_blocks(df))
    assert np.isfinite(parts.mixed).all()
    assert np.allclose(parts.binary, 0.0)
    # Остаётся 0.6 на количественные и 0.2 на категориальный.
    assert np.isclose(parts.mixed[0, 1], 0.6 * 0.5 + 0.2 * 1.0)


def test_full_weight_on_quantitative_reproduces_that_component(three_strains):
    blocks = build_blocks(three_strains)
    parts = mixed_distance(
        blocks, {"quantitative": 1.0, "binary": 0.0, "categorical": 0.0}
    )
    assert np.allclose(parts.mixed, parts.quantitative)


def test_weights_must_sum_to_one():
    with pytest.raises(ValueError, match="сумма весов"):
        validate_weights({"quantitative": 0.6, "binary": 0.2, "categorical": 0.5})


def test_weights_must_cover_all_three_components():
    with pytest.raises(ValueError, match="ожидались веса"):
        validate_weights({"quantitative": 1.0})


def test_weights_cannot_be_negative():
    with pytest.raises(ValueError, match="отрицательными"):
        validate_weights({"quantitative": 1.2, "binary": -0.2, "categorical": 0.0})


def test_default_weights_follow_the_assignment():
    assert DEFAULT_WEIGHTS == {"quantitative": 0.6, "binary": 0.2, "categorical": 0.2}
