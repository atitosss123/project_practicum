import numpy as np
import pytest

from src.clustering import cluster
from src.distances import mixed_distance
from src.loader import QUANT_FEATURES, load_strains
from src.preprocess import build_blocks
from src.significance import SHUFFLED_COLUMNS, SignificanceResult, assess, null_distribution, permute
from tests.test_loader import DATA_DIR


def silhouette_of(df):
    return cluster(mixed_distance(build_blocks(df)).mixed).silhouette


def test_permutation_preserves_every_column_as_a_multiset():
    """Нулевая модель сохраняет распределения признаков и рвёт только связи."""
    df = load_strains(DATA_DIR / "strains_features.csv")
    shuffled = permute(df, np.random.default_rng(0))
    for column in SHUFFLED_COLUMNS:
        assert sorted(shuffled[column]) == sorted(df[column]), column


def test_permutation_keeps_strain_ids_in_place():
    df = load_strains(DATA_DIR / "strains_features.csv")
    shuffled = permute(df, np.random.default_rng(0))
    assert list(shuffled.strain_id) == list(df.strain_id)


def test_permutation_actually_changes_the_arrangement():
    df = load_strains(DATA_DIR / "strains_features.csv")
    shuffled = permute(df, np.random.default_rng(0))
    assert not shuffled[QUANT_FEATURES].equals(df[QUANT_FEATURES])


def test_permutation_destroys_correlation_between_features():
    """Связь между признаками должна пропадать, иначе нулевая модель не нулевая."""
    df = load_strains(DATA_DIR / "strains_features.csv")
    observed = abs(df.growth_rate.corr(df.GC_content))
    rng = np.random.default_rng(0)
    permuted = [
        abs(permute(df, rng).growth_rate.corr(permute(df, rng).GC_content))
        for _ in range(50)
    ]
    assert observed > np.percentile(permuted, 90)


def test_p_value_arithmetic_is_explicit():
    """Поправка на единицу: p не бывает нулевым при конечном числе перестановок."""
    result = SignificanceResult(
        observed=0.5, null_scores=np.zeros(99), n_draws=99, method="ward"
    )
    assert result.p_value == pytest.approx(1 / 100)
    assert result.percentile == 100.0
    assert result.is_significant


def test_p_value_is_one_when_nothing_beats_the_null():
    result = SignificanceResult(
        observed=0.0, null_scores=np.ones(99), n_draws=99, method="ward"
    )
    assert result.p_value == pytest.approx(1.0)
    assert result.percentile == 0.0
    assert not result.is_significant


def test_null_distribution_has_the_requested_size():
    df = load_strains(DATA_DIR / "strains_features.csv")
    scores = null_distribution(df, n_draws=12, seed=1)
    assert scores.shape == (12,)
    assert np.isfinite(scores).all()


def test_null_distribution_is_reproducible():
    df = load_strains(DATA_DIR / "strains_features.csv")
    first = null_distribution(df, n_draws=8, seed=3)
    second = null_distribution(df, n_draws=8, seed=3)
    assert np.allclose(first, second)


def test_planted_structure_is_detected():
    """Основной набор: структура заложена, тест обязан её увидеть."""
    df = load_strains(DATA_DIR / "strains_features.csv")
    result = assess(df, silhouette_of(df), n_draws=100, seed=0)
    assert result.is_significant
    assert result.observed > result.null_scores.max()


def test_noise_dataset_sits_inside_its_own_null_distribution():
    """Набор без структуры не должен выделяться на фоне своих же перестановок.

    Данные - независимые равномерные признаки, поэтому наблюдённое
    расположение значений ничем не выделено среди перестановок, и силуэт
    обязан попасть внутрь нулевого распределения, а не выше него.
    """
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
    from generate_data import generate_uniform

    from src.loader import validate_strains

    df = validate_strains(generate_uniform(seed=1004))
    result = assess(df, silhouette_of(df), n_draws=100, seed=5)
    assert not result.is_significant
    assert result.observed < result.null_scores.max()


def test_summary_mentions_the_verdict():
    df = load_strains(DATA_DIR / "strains_features.csv")
    text = assess(df, silhouette_of(df), n_draws=20, seed=0).summary()
    assert "перестановок" in text
    assert "p =" in text
