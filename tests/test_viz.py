import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from src.clustering import cluster
from src.distances import mixed_distance
from src.loader import load_strains
from src.preprocess import build_blocks
from src.viz import (
    coordinates_table,
    describe_projection,
    mds_coordinates,
    pca_coordinates,
    plot_dendrogram,
    plot_projection,
    plot_silhouette_profile,
)
from tests.test_loader import DATA_DIR


@pytest.fixture
def run():
    df = load_strains(DATA_DIR / "strains_features.csv")
    blocks = build_blocks(df)
    parts = mixed_distance(blocks)
    return blocks, parts, cluster(parts.mixed)


def test_pca_returns_two_components_and_ordered_variance(run):
    blocks, _, _ = run
    coordinates, variance = pca_coordinates(blocks)
    assert coordinates.shape == (blocks.n_strains, 2)
    assert variance[0] >= variance[1]
    assert 0 < variance.sum() <= 1


def test_mds_stress_is_the_normalized_measure(run):
    """stress_ модели при метрическом MDS - сырая сумма квадратов, её нельзя
    сравнивать со шкалой Краскала. Возвращаться должен нормированный показатель."""
    _, parts, _ = run
    coordinates, stress = mds_coordinates(parts.mixed)
    assert coordinates.shape == (len(parts.mixed), 2)
    assert 0.0 <= stress <= 1.0

    upper = np.triu_indices(len(parts.mixed), 1)
    projected = squareform(pdist(coordinates, "euclidean"))[upper]
    original = parts.mixed[upper]
    expected = np.sqrt(((original - projected) ** 2).sum() / (original ** 2).sum())
    assert stress == pytest.approx(expected)


def test_mds_preserves_the_distance_ordering_reasonably(run):
    """MDS проецирует ту же матрицу, по которой кластеризовали, поэтому
    расстояния на картинке должны сильно коррелировать с исходными."""
    _, parts, _ = run
    coordinates, _ = mds_coordinates(parts.mixed)
    upper = np.triu_indices(len(parts.mixed), 1)
    projected = squareform(pdist(coordinates, "euclidean"))[upper]
    correlation = np.corrcoef(parts.mixed[upper], projected)[0, 1]
    assert correlation > 0.9


def test_coordinates_table_holds_both_projections(run):
    blocks, parts, _ = run
    table, variance = coordinates_table(blocks, parts.mixed)
    assert list(table.columns) == ["strain_id", "pc1", "pc2", "mds1", "mds2"]
    assert len(table) == blocks.n_strains
    assert list(table.strain_id) == list(blocks.strain_ids)
    assert len(variance) == 2


def test_description_lists_every_cluster_and_the_variance(run):
    blocks, _, result = run
    coordinates, variance = pca_coordinates(blocks)
    text = describe_projection(coordinates, result.labels, blocks.strain_ids, variance)
    assert "дисперсии" in text
    assert text.count("Кластер") == result.best_k
    for strain in blocks.strain_ids:
        assert strain in text


def test_plots_are_written_to_disk(run, tmp_path):
    blocks, parts, result = run
    coordinates, variance = pca_coordinates(blocks)
    files = [
        plot_silhouette_profile(result.scores, result.best_k,
                                tmp_path / "profile.png", result.plateau),
        plot_dendrogram(result.tree, blocks.strain_ids, result.best_k,
                        tmp_path / "tree.png", result.plateau),
        plot_projection(coordinates, result.labels, blocks.strain_ids,
                        tmp_path / "scatter.png", "заголовок", ("x", "y")),
    ]
    for path in files:
        assert path.exists() and path.stat().st_size > 1000
