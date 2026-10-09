import numpy as np
import pytest

from src.loader import load_strains, load_truth
from src.preprocess import build_blocks
from src.sensitivity import (
    ASSIGNMENT_SHARE,
    separating_gap,
    sweep,
    tipping_point,
    weights_for,
)
from tests.test_loader import DATA_DIR


@pytest.fixture
def inputs():
    df = load_strains(DATA_DIR / "strains_features.csv")
    truth = load_truth(DATA_DIR / "strains_truth.csv")
    return df, truth, build_blocks(df)


def test_weights_always_sum_to_one():
    for share in (0.0, 0.1, 0.2, 0.3, 0.5):
        assert sum(weights_for(share).values()) == pytest.approx(1.0)


def test_assignment_share_reproduces_the_required_weights():
    assert weights_for(ASSIGNMENT_SHARE) == pytest.approx(
        {"quantitative": 0.6, "binary": 0.2, "categorical": 0.2}
    )


def test_separating_gap_is_positive_for_real_groups(inputs):
    _, truth, blocks = inputs
    gap = separating_gap(blocks, truth.ecotype)
    assert gap["median_between_groups"] > gap["median_within_group"]
    assert gap["separating_gap"] > 0


def test_flat_penalty_is_comparable_to_the_whole_quantitative_gap(inputs):
    """Ключевой замер: один плоский штраф 0.2 съедает большую часть разрыва.

    Поэтому штамм с ложной меткой источника отъезжает почти настолько же,
    насколько отъезжал бы представитель другого экотипа.
    """
    _, truth, blocks = inputs
    gap = separating_gap(blocks, truth.ecotype)["separating_gap"]
    assert 0.5 < ASSIGNMENT_SHARE / gap < 1.0


def test_sweep_covers_every_requested_weighting(inputs):
    df, truth, blocks = inputs
    shares = (0.0, 0.1, 0.2)
    table = sweep(blocks, df, truth, shares=shares)
    assert len(table) == len(shares)
    assert table.is_assignment.sum() == 1
    assert set(table.columns) >= {"best_k", "silhouette", "ari_vs_ecotype", "ari_vs_source"}


def test_sweep_works_without_truth(inputs):
    df, _, blocks = inputs
    table = sweep(blocks, df, truth=None, shares=(0.0, 0.2))
    assert "ari_vs_ecotype" not in table.columns
    assert "ari_vs_source" in table.columns


def test_lighter_categorical_weight_recovers_the_ecotypes_better(inputs):
    """Измеренный перелом: при весе 0.15 и ниже экотипы восстанавливаются полностью."""
    df, truth, blocks = inputs
    table = sweep(blocks, df, truth, shares=(0.15, ASSIGNMENT_SHARE))
    lighter = table.loc[table.categorical == 0.15].iloc[0]
    assignment = table.loc[table.is_assignment].iloc[0]
    assert lighter.ari_vs_ecotype > assignment.ari_vs_ecotype
    assert lighter.ari_vs_source < assignment.ari_vs_source


def test_higher_silhouette_can_mean_worse_recovery(inputs):
    """Силуэт не мера правильности: при 0.25 он выше, а совпадение с истиной хуже."""
    df, truth, blocks = inputs
    table = sweep(blocks, df, truth, shares=(ASSIGNMENT_SHARE, 0.25))
    assignment = table.loc[table.is_assignment].iloc[0]
    heavier = table.loc[table.categorical == 0.25].iloc[0]
    assert heavier.silhouette > assignment.silhouette
    assert heavier.ari_vs_ecotype < assignment.ari_vs_ecotype


def test_tipping_point_names_both_numbers(inputs):
    df, truth, blocks = inputs
    text = tipping_point(sweep(blocks, df, truth))
    assert "0.15" in text and "0.20" in text


def test_tipping_point_refuses_without_truth(inputs):
    df, _, blocks = inputs
    assert "невозможен" in tipping_point(sweep(blocks, df, truth=None))
