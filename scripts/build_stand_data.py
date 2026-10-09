"""Предрасчёт данных для интерактивного стенда.

Стенд - статическая веб-страница без сервера, поэтому всё, что на ней можно
покрутить, считается заранее и складывается в один JSON.

Главный элемент стенда - ползунок веса категориальных признаков. Для каждого
его положения заранее считается полный результат: число кластеров, профиль
силуэта, разбиение, совпадение с настоящими группами и координаты точек.

Координаты многомерного шкалирования выравниваются ортогональным
преобразованием к опорному положению ползунка. Без этого проекция при каждом
шаге поворачивалась и отражалась бы случайным образом, и точки прыгали бы по
экрану, хотя разбиение почти не менялось.
"""

import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate_uniform  # noqa: E402

from src.clustering import cluster  # noqa: E402
from src.distances import mixed_distance  # noqa: E402
from src.loader import QUANT_FEATURES, load_strains, load_truth, validate_strains  # noqa: E402
from src.preprocess import build_blocks  # noqa: E402
from src.profiling import cluster_summary  # noqa: E402
from src.sensitivity import ASSIGNMENT_SHARE, separating_gap, weights_for  # noqa: E402
from src.significance import assess  # noqa: E402
from src.viz import mds_coordinates, pca_coordinates  # noqa: E402

SHARES = [round(x / 100, 2) for x in range(0, 31)]
REFERENCE_SHARE = ASSIGNMENT_SHARE
N_PERMUTATIONS = 200
N_NOISE_DATASETS = 300


def align_to(coordinates, reference):
    """Поворот и отражение координат к опорным без изменения расстояний."""
    centered = coordinates - coordinates.mean(axis=0)
    target = reference - reference.mean(axis=0)
    rotation, _ = orthogonal_procrustes(centered, target)
    return centered @ rotation


def build():
    df = load_strains(ROOT / "data" / "strains_features.csv")
    truth = load_truth(ROOT / "data" / "strains_truth.csv")
    blocks = build_blocks(df)
    parts = mixed_distance(blocks)

    ecotypes = truth["ecotype"].to_numpy()
    sources = df["source"].to_numpy()

    # Опорная проекция - при весах из условия.
    reference_mds, _ = mds_coordinates(
        mixed_distance(blocks, weights_for(REFERENCE_SHARE)).mixed
    )

    print(f"Считаю {len(SHARES)} положений ползунка...")
    steps = []
    for share in SHARES:
        weights = weights_for(share)
        distance = mixed_distance(blocks, weights).mixed
        result = cluster(distance)
        coordinates, stress = mds_coordinates(distance)
        aligned = align_to(coordinates, reference_mds)
        summary = cluster_summary(df, result.labels)

        steps.append({
            "share": share,
            "weights": {k: round(v, 2) for k, v in weights.items()},
            "best_k": int(result.best_k),
            "silhouette": round(result.silhouette, 4),
            "plateau": [int(k) for k in result.plateau],
            "is_conclusive": bool(result.is_conclusive),
            "scores": {str(k): round(v, 4) for k, v in result.scores.items()},
            "labels": [int(x) for x in result.labels],
            "ari_ecotype": round(adjusted_rand_score(ecotypes, result.labels), 3),
            "ari_source": round(adjusted_rand_score(sources, result.labels), 3),
            "mds": [[round(float(x), 4), round(float(y), 4)] for x, y in aligned],
            "stress": round(stress, 4),
            "profiles": [
                {
                    "cluster": int(row.cluster),
                    "n_strains": int(row.n_strains),
                    "growth_rate": float(row.growth_rate),
                    "product_yield": float(row.product_yield),
                    "pH_tolerance": float(row.pH_tolerance),
                    "GC_content": float(row.GC_content),
                    "genome_size_mbp": float(row.genome_size_mbp),
                    "plasmid_share": float(row.plasmid_share),
                    "source_mode": str(row.source_mode),
                    "source_mode_share": float(row.source_mode_share),
                    "source_mix": str(row.source_mix),
                }
                for row in summary.itertuples(index=False)
            ],
        })

    print("Считаю проверку значимости...")
    reference_result = cluster(mixed_distance(blocks, weights_for(REFERENCE_SHARE)).mixed)
    significance = assess(df, reference_result.silhouette, n_draws=N_PERMUTATIONS)

    print(f"Считаю эталонный уровень шума ({N_NOISE_DATASETS} наборов)...")
    noise = [
        float(cluster(mixed_distance(build_blocks(
            validate_strains(generate_uniform(seed=5000 + i))
        )).mixed).silhouette)
        for i in range(N_NOISE_DATASETS)
    ]

    pca, variance = pca_coordinates(blocks)
    gap = separating_gap(blocks, ecotypes)
    upper = np.triu_indices(len(df), 1)

    payload = {
        "strains": [
            {
                "strain_id": row.strain_id,
                "growth_rate": float(row.growth_rate),
                "product_yield": float(row.product_yield),
                "pH_tolerance": float(row.pH_tolerance),
                "GC_content": float(row.GC_content),
                "genome_size_mbp": float(row.genome_size_mbp),
                "has_plasmid": int(row.has_plasmid),
                "source": str(row.source),
                "ecotype": str(eco),
                "source_mismatch": bool(mismatch),
                "pc": [round(float(x), 4), round(float(y), 4)],
            }
            for row, eco, mismatch, (x, y) in zip(
                df.itertuples(index=False), ecotypes,
                truth["source_mismatch"].to_numpy(), pca
            )
        ],
        "features": QUANT_FEATURES,
        "steps": steps,
        "reference_share": REFERENCE_SHARE,
        "pca_variance": [round(float(v), 4) for v in variance],
        "components": {
            "quantitative": [round(float(v), 4) for v in parts.quantitative[upper]],
            "binary": [round(float(v), 4) for v in parts.binary[upper]],
            "categorical": [round(float(v), 4) for v in parts.categorical[upper]],
        },
        "distance_matrix": [[round(float(v), 3) for v in row] for row in parts.mixed],
        "significance": {
            "observed": round(significance.observed, 4),
            "p_value": round(significance.p_value, 4),
            "null_scores": [round(float(v), 4) for v in significance.null_scores],
            "n_draws": significance.n_draws,
        },
        "noise_baseline": {
            "scores": [round(v, 4) for v in noise],
            "median": round(float(np.median(noise)), 4),
            "p95": round(float(np.percentile(noise, 95)), 4),
            "n": len(noise),
        },
        "separating_gap": gap,
    }

    destination = ROOT / "stand" / "data.json"
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    size = destination.stat().st_size / 1024
    print(f"\n{destination.relative_to(ROOT)}: {size:.0f} КБ")
    print(f"  положений ползунка: {len(steps)}")
    print(f"  штаммов: {len(payload['strains'])}")
    print(f"  при весах из условия: k={steps[SHARES.index(REFERENCE_SHARE)]['best_k']}, "
          f"силуэт {steps[SHARES.index(REFERENCE_SHARE)]['silhouette']}")


if __name__ == "__main__":
    build()
