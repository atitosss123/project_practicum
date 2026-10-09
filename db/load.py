"""Загрузка данных и результатов расчёта в PostgreSQL.

Скрипт прогоняет пайплайн и складывает в базу и исходные данные, и всё
посчитанное: матрицу непохожести, профиль силуэта, разбиение, профили
кластеров, координаты проекций и таблицу устойчивости к весам. После этого
любое число из отчёта поднимается запросом, а не пересчитывается.

Строка подключения берётся из переменной окружения DATABASE_URL, по умолчанию
postgresql://localhost/strains.

Запуск:
    python db/load.py --create-schema          # создать схему и загрузить всё
    python db/load.py                          # только перезалить данные
    DATABASE_URL=postgresql://... python db/load.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clustering import cluster  # noqa: E402
from src.distances import DEFAULT_WEIGHTS, mixed_distance  # noqa: E402
from src.loader import load_strains, load_truth  # noqa: E402
from src.preprocess import build_blocks  # noqa: E402
from src.profiling import cluster_summary  # noqa: E402
from src.sensitivity import sweep  # noqa: E402
from src.significance import assess  # noqa: E402
from src.viz import coordinates_table  # noqa: E402

DEFAULT_DSN = "postgresql://localhost/strains"

DATASETS = [
    {
        "name": "structured",
        "description": "Основной набор: три экотипа с перекрытием и шумом",
        "generator": "data/generate_data.py::generate_structured",
        "seed": 11,
        "csv": "data/strains_features.csv",
        "truth_csv": "data/strains_truth.csv",
    },
    {
        "name": "uniform",
        "description": "Контрольный набор: код генерации из условия задачи",
        "generator": "data/generate_data.py::generate_uniform",
        "seed": 123,
        "csv": "data/strains_uniform.csv",
        "truth_csv": None,
    },
]


def connect(dsn):
    return psycopg.connect(dsn, autocommit=False)


def create_schema(connection):
    """Пересоздаёт схему с нуля."""
    schema = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with connection.cursor() as cursor:
        cursor.execute(schema)
    connection.commit()


def insert_dataset(cursor, spec, has_truth):
    cursor.execute(
        """INSERT INTO dataset (name, description, generator, seed, has_truth)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (name) DO UPDATE
               SET description = EXCLUDED.description,
                   generator   = EXCLUDED.generator,
                   seed        = EXCLUDED.seed,
                   has_truth   = EXCLUDED.has_truth
           RETURNING id""",
        (spec["name"], spec["description"], spec["generator"], spec["seed"], has_truth),
    )
    return cursor.fetchone()[0]


def insert_strains(cursor, dataset_id, df):
    """Возвращает отображение «код штамма -> первичный ключ»."""
    rows = [
        (dataset_id, row.strain_id, row.growth_rate, row.product_yield,
         row.pH_tolerance, row.GC_content, row.genome_size_mbp,
         bool(row.has_plasmid), row.source)
        for row in df.itertuples(index=False)
    ]
    cursor.executemany(
        """INSERT INTO strain (dataset_id, strain_id, growth_rate, product_yield,
                               ph_tolerance, gc_content, genome_size_mbp,
                               has_plasmid, source)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    cursor.execute(
        "SELECT strain_id, id FROM strain WHERE dataset_id = %s", (dataset_id,)
    )
    return dict(cursor.fetchall())


def insert_truth(cursor, keys, truth):
    cursor.executemany(
        "INSERT INTO strain_truth (strain_id, ecotype, source_mismatch) VALUES (%s, %s, %s)",
        [(keys[row.strain_id], row.ecotype, bool(row.source_mismatch))
         for row in truth.itertuples(index=False)],
    )


def insert_distances(cursor, dataset_id, keys, codes, parts):
    """Только верхний треугольник: матрица симметрична, диагональ нулевая."""
    rows = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = keys[codes[i]], keys[codes[j]]
            if a > b:
                a, b = b, a
            rows.append((dataset_id, a, b,
                         float(parts.quantitative[i, j]), float(parts.binary[i, j]),
                         float(parts.categorical[i, j]), float(parts.mixed[i, j])))
    cursor.executemany(
        """INSERT INTO pair_distance (dataset_id, strain_a, strain_b,
                                      d_quantitative, d_binary, d_categorical, d_mixed)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    return len(rows)


def insert_run(cursor, dataset_id, result, weights, significance, ari_truth, ari_source):
    cursor.execute(
        """INSERT INTO clustering_run (dataset_id, linkage_method,
               weight_quantitative, weight_binary, weight_categorical,
               best_k, silhouette, is_conclusive, plateau,
               p_value, n_permutations, ari_vs_truth, ari_vs_source)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
           RETURNING id""",
        (dataset_id, result.method,
         weights["quantitative"], weights["binary"], weights["categorical"],
         result.best_k, result.silhouette, result.is_conclusive,
         list(result.plateau),
         significance.p_value if significance else None,
         significance.n_draws if significance else None,
         ari_truth, ari_source),
    )
    return cursor.fetchone()[0]


def load_dataset(connection, spec, n_permutations):
    """Полный прогон одного набора с записью всех результатов."""
    df = load_strains(ROOT / spec["csv"])
    truth = load_truth(ROOT / spec["truth_csv"]) if spec["truth_csv"] else None

    blocks = build_blocks(df)
    parts = mixed_distance(blocks)
    result = cluster(parts.mixed)
    significance = assess(df, result.silhouette, n_draws=n_permutations)

    from sklearn.metrics import adjusted_rand_score
    ari_truth = (round(adjusted_rand_score(truth["ecotype"], result.labels), 3)
                 if truth is not None else None)
    ari_source = round(adjusted_rand_score(df["source"], result.labels), 3)

    codes = list(df.strain_id)

    with connection.cursor() as cursor:
        dataset_id = insert_dataset(cursor, spec, truth is not None)
        keys = insert_strains(cursor, dataset_id, df)
        if truth is not None:
            insert_truth(cursor, keys, truth)
        n_pairs = insert_distances(cursor, dataset_id, keys, codes, parts)

        run_id = insert_run(cursor, dataset_id, result, DEFAULT_WEIGHTS,
                            significance, ari_truth, ari_source)

        cursor.executemany(
            "INSERT INTO silhouette_score (run_id, k, score) VALUES (%s, %s, %s)",
            [(run_id, k, score) for k, score in result.scores.items()],
        )
        cursor.executemany(
            "INSERT INTO cluster_assignment (run_id, strain_id, cluster) VALUES (%s, %s, %s)",
            [(run_id, keys[code], int(label)) for code, label in zip(codes, result.labels)],
        )

        summary = cluster_summary(df, result.labels)
        cursor.executemany(
            """INSERT INTO cluster_profile (run_id, cluster, n_strains, growth_rate,
                   product_yield, ph_tolerance, gc_content, genome_size_mbp,
                   plasmid_share, source_mode, source_mode_share, source_mix)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [(run_id, int(row.cluster), int(row.n_strains), row.growth_rate,
              row.product_yield, row.pH_tolerance, row.GC_content,
              row.genome_size_mbp, row.plasmid_share, row.source_mode,
              row.source_mode_share, row.source_mix)
             for row in summary.itertuples(index=False)],
        )

        coordinates, _ = coordinates_table(blocks, parts.mixed)
        cursor.executemany(
            "INSERT INTO projection (run_id, strain_id, pc1, pc2, mds1, mds2) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            [(run_id, keys[row.strain_id], row.pc1, row.pc2, row.mds1, row.mds2)
             for row in coordinates.itertuples(index=False)],
        )

        table = sweep(blocks, df, truth)
        cursor.executemany(
            """INSERT INTO weight_sensitivity (dataset_id, weight_categorical,
                   weight_quantitative, best_k, silhouette, ari_vs_truth,
                   ari_vs_source, is_assignment)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [(dataset_id, row.categorical, row.quantitative, int(row.best_k),
              row.silhouette,
              getattr(row, "ari_vs_ecotype", None) if truth is not None else None,
              row.ari_vs_source, bool(row.is_assignment))
             for row in table.itertuples(index=False)],
        )

    connection.commit()
    return {
        "dataset": spec["name"], "strains": len(df), "pairs": n_pairs,
        "best_k": result.best_k, "silhouette": round(result.silhouette, 3),
        "p_value": round(significance.p_value, 4), "ari_vs_truth": ari_truth,
    }


def main():
    parser = argparse.ArgumentParser(description="Загрузка проекта в PostgreSQL.")
    parser.add_argument("--dsn", default=os.environ.get("DATABASE_URL", DEFAULT_DSN))
    parser.add_argument("--create-schema", action="store_true",
                        help="пересоздать схему перед загрузкой (удаляет данные)")
    parser.add_argument("--n-permutations", type=int, default=200)
    args = parser.parse_args()

    print(f"Подключение: {args.dsn}")
    with connect(args.dsn) as connection:
        if args.create_schema:
            create_schema(connection)
            print("Схема создана")

        for spec in DATASETS:
            report = load_dataset(connection, spec, args.n_permutations)
            print(f"  {report['dataset']:12s} штаммов {report['strains']:2d}, "
                  f"пар {report['pairs']:3d}, k={report['best_k']}, "
                  f"силуэт {report['silhouette']}, p={report['p_value']}"
                  + (f", ARI к истине {report['ari_vs_truth']}"
                     if report["ari_vs_truth"] is not None else ""))

        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT table_name, (xpath('/row/c/text()',
                       query_to_xml(format('SELECT count(*) AS c FROM %I', table_name),
                                    false, true, '')))[1]::text::int AS rows
                FROM information_schema.tables
                WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
                ORDER BY table_name""")
            print("\nСтрок в таблицах:")
            for name, count in cursor.fetchall():
                print(f"  {name:22s} {count}")


if __name__ == "__main__":
    main()
