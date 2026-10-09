import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster
from src.distances import mixed_distance
from src.loader import QUANT_FEATURES, load_strains, load_truth
from src.preprocess import build_blocks
from src.profiling import (
    agreement_with_truth,
    assignments,
    categorical_echo,
    cluster_summary,
    describe_clusters,
    mismatched_strains,
)
from tests.test_loader import DATA_DIR


@pytest.fixture
def main_run():
    df = load_strains(DATA_DIR / "strains_features.csv")
    truth = load_truth(DATA_DIR / "strains_truth.csv")
    result = cluster(mixed_distance(build_blocks(df)).mixed)
    return df, truth, result


def test_assignments_cover_every_strain_once(main_run):
    df, _, result = main_run
    table = assignments(df, result.labels)
    assert list(table.strain_id) == list(df.strain_id)
    assert table.cluster.nunique() == result.best_k


def test_summary_reports_means_in_original_units(main_run):
    """Средние должны читаться как проценты GC и мегабазы, а не как z-оценки."""
    df, _, result = main_run
    summary = cluster_summary(df, result.labels)
    assert len(summary) == result.best_k
    assert summary.n_strains.sum() == len(df)
    assert summary.GC_content.between(40, 70).all()
    assert summary.genome_size_mbp.between(3, 6).all()


def test_summary_means_match_a_direct_computation(main_run):
    df, _, result = main_run
    summary = cluster_summary(df, result.labels)
    first = summary.loc[summary.cluster == 1].iloc[0]
    members = df[result.labels == 1]
    for feature in QUANT_FEATURES:
        assert first[feature] == pytest.approx(members[feature].mean(), abs=1e-3)


def test_summary_records_how_dominant_the_source_mode_is(main_run):
    """Без доли моды «преимущественно почва» может значить и 7 из 7, и 3 из 7."""
    df, _, result = main_run
    summary = cluster_summary(df, result.labels)
    assert summary.source_mode_share.between(0, 1).all()
    assert (summary.source_mode_share >= 1 / 3).all()
    assert summary.source_mix.str.contains(r"\d").all()


def test_categorical_echo_detects_a_pure_retelling():
    """Если кластеры совпали с источником один в один, эхо должно быть единицей."""
    df = load_strains(DATA_DIR / "strains_features.csv")
    labels = pd.factorize(df.source)[0] + 1
    echo = categorical_echo(df, labels)
    assert echo["source_only"] == pytest.approx(1.0)


def test_categorical_echo_on_the_main_dataset_exceeds_truth_agreement(main_run):
    """Измеренный результат: разбиение ближе к заявленному источнику, чем к экотипу.

    Это следствие весов из задания, а не ошибка: плоский штраф 0.2 за
    несовпадение источника сопоставим с разделяющей способностью всех пяти
    количественных признаков.
    """
    df, truth, result = main_run
    echo = categorical_echo(df, result.labels)
    agreement = agreement_with_truth(truth, result.labels)
    assert echo["source_only"] > agreement["ari"]


def test_agreement_returns_a_crosstab_over_all_strains(main_run):
    _, truth, result = main_run
    agreement = agreement_with_truth(truth, result.labels)
    assert agreement["crosstab"].to_numpy().sum() == len(truth)
    assert 0.0 <= agreement["ari"] <= 1.0


def test_mismatched_strains_are_listed_with_their_destination(main_run):
    df, truth, result = main_run
    table = mismatched_strains(df, truth, result.labels)
    assert len(table) == int(truth.source_mismatch.sum())
    assert set(table.columns) >= {"strain_id", "ecotype", "source", "followed_ecotype"}
    # Источник по определению не совпадает с экотипом у каждого из них.
    assert (table.ecotype != table.source).all()


def test_mismatched_strains_empty_without_truth_column(main_run):
    df, truth, result = main_run
    assert mismatched_strains(df, truth.drop(columns=["source_mismatch"]),
                              result.labels).empty


def test_description_mentions_every_cluster(main_run):
    df, _, result = main_run
    text = describe_clusters(cluster_summary(df, result.labels))
    assert text.count("Кластер") == result.best_k
    assert "Плазмида" in text
