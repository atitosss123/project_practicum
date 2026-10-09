import numpy as np
import pytest
from sklearn.metrics import adjusted_rand_score

from src.clustering import (
    K_RANGE,
    build_tree,
    cluster,
    find_plateau,
    labels_at,
    silhouette_profile,
)
from src.distances import mixed_distance
from src.loader import load_strains, load_truth
from src.preprocess import build_blocks
from tests.test_loader import DATA_DIR


def distance_of(df):
    return mixed_distance(build_blocks(df)).mixed


def test_finds_three_obvious_clusters(three_blocks):
    """Контроль на однозначном случае: три разделённые группы обязаны найтись."""
    result = cluster(distance_of(three_blocks))
    assert result.best_k == 3
    assert result.cluster_sizes == {1: 4, 2: 4, 3: 4}
    assert result.silhouette > 0.8
    assert result.is_conclusive


def test_recovers_block_membership(three_blocks):
    result = cluster(distance_of(three_blocks))
    assert adjusted_rand_score(three_blocks["source"], result.labels) == 1.0


def test_profile_covers_the_whole_k_range(three_blocks):
    distance = distance_of(three_blocks)
    scores = silhouette_profile(distance, build_tree(distance))
    # 12 штаммов: k от 2 до 10 укладывается целиком.
    assert set(scores) == set(K_RANGE)
    assert all(-1.0 <= value <= 1.0 for value in scores.values())


def test_three_strains_have_no_usable_k(three_strains):
    """На трёх штаммах ни одно k из диапазона не годится, и это явная ошибка.

    k >= 3 недостижимо при n = 3, а k = 2 попадает в случай k = n-1, где
    scipy возвращает одну группу вместо двух. Молча вернуть профиль из
    одного сомнительного значения было бы хуже, чем отказать.
    """
    distance = distance_of(three_strains)
    with pytest.raises(ValueError, match="ни одно число кластеров"):
        silhouette_profile(distance, build_tree(distance))


@pytest.mark.parametrize("n_objects", [4, 5, 6, 8, 12, 20])
def test_maxclust_shortfall_at_k_equals_n_minus_one_is_excluded(n_objects):
    """Документирует свойство scipy, из-за которого нужен отсев k = n-1."""
    from scipy.cluster.hierarchy import fcluster, linkage

    rng = np.random.default_rng(0)
    tree = linkage(rng.normal(size=(n_objects, 5)), "ward")
    shortfall = len(np.unique(fcluster(tree, n_objects - 1, "maxclust")))
    assert shortfall == n_objects - 2, "ожидался недобор на единицу при k = n-1"

    # Профиль такое k не возвращает: единственное запрошенное значение отсеяно,
    # оценивать нечего, и это явная ошибка вместо молчаливого пропуска.
    distance = np.abs(rng.normal(size=(n_objects, n_objects)))
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    with pytest.raises(ValueError, match="ни одно число кластеров"):
        silhouette_profile(distance, build_tree(distance), k_range=(n_objects - 1,))


def test_plateau_collects_everything_within_tolerance():
    scores = {2: 0.490, 3: 0.494, 4: 0.470, 5: 0.492, 6: 0.400}
    assert find_plateau(scores, tolerance=0.01) == (2, 3, 5)
    assert find_plateau(scores, tolerance=0.001) == (3,)


def test_single_peak_is_conclusive():
    assert find_plateau({2: 0.3, 3: 0.8, 4: 0.3}, tolerance=0.01) == (3,)


def test_labels_at_reuses_the_same_tree(three_blocks):
    result = cluster(distance_of(three_blocks))
    for k in (2, 4, 5):
        assert len(np.unique(labels_at(result, k))) == k


def test_main_dataset_plateau_is_flagged_as_inconclusive():
    """На основном наборе k определено неустойчиво, и объект это сообщает.

    Профиль почти плоский: 0.490 при k=2, 0.494 при k=3, 0.492 при k=5.
    Разрыв 0.004 на 20 объектах - шум, а не сигнал.
    """
    result = cluster(distance_of(load_strains(DATA_DIR / "strains_features.csv")))
    assert result.best_k == 3
    assert not result.is_conclusive
    assert result.plateau == (2, 3, 5)


def test_main_dataset_recovers_the_planted_ecotypes():
    features = load_strains(DATA_DIR / "strains_features.csv")
    truth = load_truth(DATA_DIR / "strains_truth.csv")
    result = cluster(distance_of(features))
    assert adjusted_rand_score(truth["ecotype"], result.labels) > 0.65


def test_ward_and_average_agree_on_the_main_dataset():
    """Контроль допущения о метрике: Уорд на нашей матрице некорректен строго,
    average - корректен. Близкие разбиения означают, что результат не артефакт."""
    distance = distance_of(load_strains(DATA_DIR / "strains_features.csv"))
    ward = cluster(distance, method="ward")
    average = cluster(distance, method="average")
    assert adjusted_rand_score(ward.labels, average.labels) > 0.6


def test_unusable_k_range_is_an_error(three_strains):
    with pytest.raises(ValueError, match="ни одно число кластеров"):
        distance = distance_of(three_strains)
        silhouette_profile(distance, build_tree(distance), k_range=(5, 6))
