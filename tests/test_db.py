"""Проверки базы данных.

Часть проверок читает только файл схемы и работает без сервера. Остальные
требуют поднятой базы и пропускаются, если её нет: тесты не должны падать
на машине, где PostgreSQL ещё не установлен.
"""

import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = (ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
DSN = os.environ.get("DATABASE_URL", "postgresql://localhost/strains")

EXPECTED_TABLES = {
    "dataset", "strain", "strain_truth", "pair_distance", "clustering_run",
    "silhouette_score", "cluster_assignment", "cluster_profile", "projection",
    "weight_sensitivity",
}


# --- проверки файла схемы, сервер не нужен ---

def test_schema_declares_every_expected_table():
    declared = set(re.findall(r"CREATE TABLE (\w+)", SCHEMA))
    assert declared == EXPECTED_TABLES


def test_source_is_an_enum_not_free_text():
    """Опечатка «Soil» молча создала бы четвёртую категорию и изменила расстояния."""
    assert "CREATE TYPE isolation_source AS ENUM ('gut', 'soil', 'water')" in SCHEMA
    assert "source           isolation_source NOT NULL" in SCHEMA


def test_pair_is_stored_once():
    """Матрица симметрична; ограничение делает дубль пары невозможным."""
    assert "CHECK (strain_a < strain_b)" in SCHEMA


def test_weights_must_sum_to_one_at_the_table_level():
    assert "CHECK (weight_quantitative + weight_binary + weight_categorical = 1.0)" in SCHEMA


def test_feature_limits_match_the_python_validation():
    """Границы в базе и в загрузчике должны совпадать, иначе данные разойдутся."""
    sys.path.insert(0, str(ROOT))
    from src.loader import FEATURE_LIMITS

    for feature, (low, high) in FEATURE_LIMITS.items():
        column = "ph_tolerance" if feature == "pH_tolerance" else feature.lower()
        pattern = rf"CHECK \({column}\s+BETWEEN\s+{low:g}\s+AND\s+{high:g}\)"
        assert re.search(pattern, SCHEMA), f"{feature}: нет ограничения {low}-{high}"


def test_plateau_is_stored_so_uncertainty_survives():
    """Без массива плато из базы не узнать, что best_k выбран с перевесом 0.004."""
    assert "plateau         smallint[] NOT NULL" in SCHEMA
    assert "is_conclusive   boolean  NOT NULL" in SCHEMA


# --- проверки на живой базе ---

def _connection():
    psycopg = pytest.importorskip("psycopg", reason="драйвер psycopg не установлен")
    try:
        return psycopg.connect(DSN, connect_timeout=3)
    except Exception as error:  # noqa: BLE001 - любая ошибка означает «базы нет»
        pytest.skip(f"база недоступна ({DSN}): {error}")


@pytest.fixture(scope="module")
def db():
    connection = _connection()
    with connection.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM information_schema.tables "
                       "WHERE table_schema = 'public' AND table_name = 'strain'")
        if cursor.fetchone()[0] == 0:
            pytest.skip("схема не создана: запустите db/load.py --create-schema")
    yield connection
    connection.close()


def test_both_datasets_are_loaded(db):
    with db.cursor() as cursor:
        cursor.execute("SELECT name, has_truth FROM dataset ORDER BY name")
        rows = dict(cursor.fetchall())
    assert rows == {"structured": True, "uniform": False}


def test_every_dataset_holds_twenty_strains(db):
    with db.cursor() as cursor:
        cursor.execute("SELECT d.name, count(*) FROM strain s "
                       "JOIN dataset d ON d.id = s.dataset_id GROUP BY d.name")
        for name, count in cursor.fetchall():
            assert count == 20, name


def test_distance_matrix_is_complete(db):
    """20 штаммов дают 190 пар в верхнем треугольнике."""
    with db.cursor() as cursor:
        cursor.execute("SELECT d.name, count(*) FROM pair_distance p "
                       "JOIN dataset d ON d.id = p.dataset_id GROUP BY d.name")
        for name, count in cursor.fetchall():
            assert count == 190, name


def test_full_matrix_view_adds_both_halves_and_diagonal(db):
    with db.cursor() as cursor:
        cursor.execute("SELECT count(*) FROM distance_matrix_full m "
                       "JOIN dataset d ON d.id = m.dataset_id WHERE d.name = 'structured'")
        assert cursor.fetchone()[0] == 190 * 2 + 20


def test_degenerate_components_are_visible_in_the_data(db):
    """Ключевой замер проекта должен подниматься запросом."""
    with db.cursor() as cursor:
        cursor.execute("""SELECT count(DISTINCT d_quantitative), count(DISTINCT d_binary),
                                 count(DISTINCT d_categorical)
                          FROM pair_distance p JOIN dataset d ON d.id = p.dataset_id
                          WHERE d.name = 'structured'""")
        quantitative, binary, categorical = cursor.fetchone()
    assert quantitative == 190
    assert binary == 2
    assert categorical == 2


def test_main_run_matches_the_report(db):
    with db.cursor() as cursor:
        cursor.execute("""SELECT best_k, silhouette, is_conclusive, plateau, p_value, ari_vs_truth
                          FROM clustering_run r JOIN dataset d ON d.id = r.dataset_id
                          WHERE d.name = 'structured'""")
        best_k, silhouette, conclusive, plateau, p_value, ari = cursor.fetchone()
    assert best_k == 3
    assert float(silhouette) == pytest.approx(0.4941, abs=5e-4)
    assert conclusive is False
    assert sorted(plateau) == [2, 3, 5]
    assert float(p_value) < 0.05
    assert float(ari) == pytest.approx(0.70, abs=0.01)


def test_every_strain_got_a_cluster(db):
    with db.cursor() as cursor:
        cursor.execute("""SELECT count(*) FROM strain s
                          LEFT JOIN cluster_assignment c ON c.strain_id = s.id
                          WHERE c.strain_id IS NULL""")
        assert cursor.fetchone()[0] == 0


def test_weight_sensitivity_records_the_tipping_point(db):
    """При весе 0.15 группы восстанавливаются полностью, при 0.20 - уже нет."""
    with db.cursor() as cursor:
        cursor.execute("""SELECT weight_categorical, ari_vs_truth, ari_vs_source, is_assignment
                          FROM weight_sensitivity w JOIN dataset d ON d.id = w.dataset_id
                          WHERE d.name = 'structured' ORDER BY weight_categorical""")
        rows = {float(w): (float(t), float(s), a) for w, t, s, a in cursor.fetchall()}
    assert rows[0.15][0] == pytest.approx(1.0, abs=0.01)
    assert rows[0.20][0] < rows[0.15][0]
    assert rows[0.20][1] > rows[0.20][0], "разбиение ближе к метке, чем к настоящей группе"
    assert rows[0.20][2] is True


def test_enum_rejects_an_unknown_source(db):
    with db.cursor() as cursor:
        cursor.execute("SAVEPOINT probe")
        with pytest.raises(Exception):
            cursor.execute("SELECT 'sediment'::isolation_source")
        cursor.execute("ROLLBACK TO SAVEPOINT probe")


def _statements(path: Path):
    """Запросы из файла без мета-команд psql."""
    body = "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines()
        if not line.lstrip().startswith("\\")
    )
    return [chunk.strip() for chunk in body.split(";") if chunk.strip()]


def test_every_demo_query_runs(db):
    """Запросы из queries.sql должны выполняться, а не падать при показе.

    Этот тест поймал две реальные ошибки: mode() WITHIN GROUP использовался
    как оконная функция, чего PostgreSQL не допускает, и пустое имя колонки
    AS "" вызывало ошибку разбора.
    """
    statements = _statements(ROOT / "db" / "queries.sql")
    assert len(statements) == 8, f"ожидалось 8 запросов, найдено {len(statements)}"
    with db.cursor() as cursor:
        for number, statement in enumerate(statements, 1):
            cursor.execute("SAVEPOINT q")
            try:
                cursor.execute(statement)
                assert cursor.fetchall() is not None
            except Exception as error:  # noqa: BLE001
                pytest.fail(f"запрос {number} упал: {error}\n{statement[:200]}")
            finally:
                cursor.execute("ROLLBACK TO SAVEPOINT q")


def test_row_counts_match_the_documented_ones(db):
    """Числа в db/README.md должны соответствовать тому, что реально в базе."""
    documented = {
        "dataset": 2, "strain": 40, "strain_truth": 20, "pair_distance": 380,
        "clustering_run": 2, "silhouette_score": 18, "cluster_assignment": 40,
        "cluster_profile": 8, "projection": 40, "weight_sensitivity": 14,
    }
    with db.cursor() as cursor:
        for table, expected in documented.items():
            cursor.execute(f"SELECT count(*) FROM {table}")
            assert cursor.fetchone()[0] == expected, table
