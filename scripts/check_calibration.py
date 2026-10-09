"""Проверка теста значимости: калибровка и эталонный уровень шума.

Две вещи, которые нельзя заявлять без замера.

1. Калибровка. Тест прогоняется на наборах, где нулевая гипотеза верна по
   построению (все признаки независимо равномерные). Доля p < 0.05 должна
   быть около номинальных 5%, иначе тест выдумывает структуру.

2. Эталонный уровень шума. Максимум силуэта по k на независимых равномерных
   наборах того же размера. Нужен, чтобы понимать, какой силуэт на 20
   объектах вообще считается высоким.

Второй замер отдельно показывает, насколько типичен контрольный набор из
задания (seed=123): если он окажется выбросом, опираться на него как на
единственный эталон шума нельзя.

Запуск (около двух минут на значениях по умолчанию):
    python scripts/check_calibration.py
    python scripts/check_calibration.py --n-datasets 20 --n-draws 50
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

from generate_data import generate_uniform  # noqa: E402

from src.clustering import cluster  # noqa: E402
from src.distances import mixed_distance  # noqa: E402
from src.loader import load_strains, validate_strains  # noqa: E402
from src.preprocess import build_blocks  # noqa: E402
from src.significance import assess  # noqa: E402


def max_silhouette(df) -> float:
    """Максимум силуэта по k на смешанной матрице расстояний."""
    return cluster(mixed_distance(build_blocks(df)).mixed).silhouette


def check_calibration(n_datasets: int, n_draws: int) -> np.ndarray:
    """p-значения на наборах, где структуры нет по построению."""
    p_values = []
    for index in range(n_datasets):
        df = validate_strains(generate_uniform(seed=1000 + index))
        p_values.append(assess(df, max_silhouette(df), n_draws=n_draws, seed=index).p_value)
    return np.array(p_values)


def noise_baseline(n_datasets: int) -> np.ndarray:
    """Максимум силуэта на независимых равномерных наборах."""
    return np.array(
        [max_silhouette(validate_strains(generate_uniform(seed=5000 + i)))
         for i in range(n_datasets)]
    )


def main():
    parser = argparse.ArgumentParser(description="Проверка теста значимости.")
    parser.add_argument("--n-datasets", type=int, default=40,
                        help="сколько шумовых наборов для калибровки")
    parser.add_argument("--n-draws", type=int, default=100,
                        help="сколько перестановок на каждый набор")
    parser.add_argument("--n-baseline", type=int, default=300,
                        help="сколько наборов для эталонного уровня шума")
    args = parser.parse_args()

    print(f"1. Калибровка: {args.n_datasets} шумовых наборов "
          f"по {args.n_draws} перестановок каждый")
    p_values = check_calibration(args.n_datasets, args.n_draws)
    false_positives = int((p_values < 0.05).sum())
    print(f"   p < 0.05 у {false_positives} из {args.n_datasets} "
          f"({false_positives / args.n_datasets * 100:.1f}%), номинально ожидается 5%")
    print(f"   медиана p = {np.median(p_values):.3f}, ожидается около 0.5")
    verdict = ("тест консервативен: скорее пропустит структуру, чем выдумает"
               if np.median(p_values) > 0.5
               else "тест не консервативен, нужна осторожность в выводах")
    print(f"   {verdict}")

    print(f"\n2. Эталонный уровень шума: {args.n_baseline} равномерных наборов")
    baseline = noise_baseline(args.n_baseline)
    print(f"   максимум силуэта по k: медиана {np.median(baseline):.3f}, "
          f"разброс {baseline.min():.3f}-{baseline.max():.3f}")
    print(f"   95-й процентиль {np.percentile(baseline, 95):.3f}, "
          f"99-й {np.percentile(baseline, 99):.3f}")

    print("\n3. Где на этом фоне наши наборы")
    for title, path in (
        ("контрольный из задания (seed=123)", ROOT / "data" / "strains_uniform.csv"),
        ("основной со структурой", ROOT / "data" / "strains_features.csv"),
    ):
        observed = max_silhouette(load_strains(path))
        percentile = (baseline < observed).mean() * 100
        print(f"   {title:36s} силуэт {observed:.3f} -> процентиль {percentile:.1f}")


if __name__ == "__main__":
    main()
