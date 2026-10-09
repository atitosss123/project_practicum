"""Генераторы входных данных для кластеризации штаммов.

Два набора:

structured — основной. Три экотипа, привязанных к источнику выделения, с
    перекрытием классов и шумом. На нём считаем и интерпретируем результат.
uniform    — контрольный. Код из задания (seed=123): все признаки независимо
    равномерные, кластерной структуры нет.

Как подобраны параметры основного набора
----------------------------------------
Центроиды экотипов разведены по РАЗНЫМ направлениям признакового
пространства. Первая версия задавала одинаковый порядок по всем пяти
признакам (кишечник < вода < почва) - данные выродились в одномерный
градиент, где "вода" оказалась буквально серединой, и силуэт справедливо
находил два кластера вместо трёх.

Расстояние между центроидами подобрано так, чтобы задача не решалась
тривиально: при разведении в 1.5 раза сильнее восстановление экотипов
становится идеальным на любом seed, и демонстрировать нечего. На текущих
параметрах восстановление по смешанному расстоянию в среднем ARI ~ 0.75 и
идеально лишь в 5 прогонах из 30.

Seed основного набора выбран перебором (см. seed_sweep.py) среди тех, где
силуэт определяет k=3 и ARI попадает в 0.65-0.93, то есть один-два штамма
относятся не к своему экотипу - это и есть предмет интерпретации. Условию
отвечают 58 из первых 200 seed'ов, то есть выбор не уникален и ни на что
исключительное не опирается.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent

STRUCTURED_SEED = 11
UNIFORM_SEED = 123

COLUMNS = [
    "strain_id",
    "growth_rate",
    "product_yield",
    "pH_tolerance",
    "GC_content",
    "genome_size_mbp",
    "has_plasmid",
    "source",
]

# Физически осмысленные границы признаков (взяты из задания).
LIMITS = {
    "growth_rate": (0.5, 1.5),
    "product_yield": (1.5, 4.0),
    "pH_tolerance": (5.0, 9.0),
    "GC_content": (45.0, 65.0),
    "genome_size_mbp": (3.0, 6.0),
}
QUANT_FEATURES = list(LIMITS)

# Разброс внутри экотипа, общий для всех трёх.
SPREAD = {
    "growth_rate": 0.15,
    "product_yield": 0.33,
    "pH_tolerance": 0.45,
    "GC_content": 2.3,
    "genome_size_mbp": 0.30,
}

# Центры экотипов. Каждый отличается от остальных по своему набору признаков:
#   кишечник - быстрый рост, низкий выход продукта, низкое GC, компактный геном;
#   почва    - медленный рост, высокий выход, щелочная толерантность, крупный геном;
#   вода     - быстрый рост и высокий выход при мелком геноме и высоком GC.
# Коллинеарности нет: ни по одному признаку порядок троек не повторяет другие.
#
# plasmid_share задаёт долю носителей плазмиды как свойство экотипа, а не как
# вероятность независимых бросков: на 6-7 штаммах биномиальный разброс съедает
# задуманный контраст (в первой версии 0.8/0.2/0.5 выродились в 0.43/0.29/0.17,
# и признак стал почти константой). Случайным остаётся только то, какие именно
# штаммы экотипа получают плазмиду.
ECOTYPES = {
    "gut": {
        "n": 7,
        "growth_rate": 1.195,
        "product_yield": 2.188,
        "pH_tolerance": 6.293,
        "GC_content": 50.55,
        "genome_size_mbp": 3.768,
        "plasmid_share": 0.80,
    },
    "soil": {
        "n": 7,
        "growth_rate": 0.783,
        "product_yield": 3.313,
        "pH_tolerance": 7.868,
        "GC_content": 60.30,
        "genome_size_mbp": 5.193,
        "plasmid_share": 0.20,
    },
    "water": {
        "n": 6,
        "growth_rate": 1.120,
        "product_yield": 3.200,
        "pH_tolerance": 6.443,
        "GC_content": 59.18,
        "genome_size_mbp": 3.843,
        "plasmid_share": 0.50,
    },
}

# Штаммы, у которых источник выделения не совпадает с экотипом: организм
# кишечного типа, выделенный из воды (фекальное загрязнение), и наоборот.
# Проверяют, могут ли количественные признаки перебить ложную категорию.
MISMATCH_COUNT = 3

ROUNDING = {
    "growth_rate": 3,
    "product_yield": 2,
    "pH_tolerance": 2,
    "GC_content": 2,
    "genome_size_mbp": 3,
}


def generate_structured(seed: int = STRUCTURED_SEED, mismatch_count: int = MISMATCH_COUNT):
    """Набор с заложенной структурой.

    Возвращает (features, truth): таблицу по схеме задания и таблицу скрытой
    истины с настоящим экотипом каждого штамма. Экотип в кластеризацию не
    подаётся - только для проверки, насколько разбиение его воспроизводит.
    """
    rng = np.random.default_rng(seed)

    rows, ecotypes = [], []
    for name, params in ECOTYPES.items():
        n = params["n"]
        for _ in range(n):
            row = {}
            for feature in QUANT_FEATURES:
                low, high = LIMITS[feature]
                value = rng.normal(params[feature], SPREAD[feature])
                row[feature] = float(np.clip(value, low, high))
            rows.append(row)
            ecotypes.append(name)

        # Квота носителей плазмиды внутри экотипа: сколько - задано, кто - случайно.
        carriers = int(round(params["plasmid_share"] * n))
        flags = np.array([1] * carriers + [0] * (n - carriers))
        rng.shuffle(flags)
        for row, flag in zip(rows[-n:], flags):
            row["has_plasmid"] = int(flag)

    df = pd.DataFrame(rows)
    df["ecotype"] = ecotypes
    df["source"] = df["ecotype"]

    # Перемешиваем, чтобы номер штамма не выдавал группу.
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df["strain_id"] = [f"S{i:02d}" for i in range(1, len(df) + 1)]

    # Подменяем источник у части штаммов на любой другой.
    all_sources = sorted(ECOTYPES)
    for i in rng.choice(len(df), size=mismatch_count, replace=False):
        others = [s for s in all_sources if s != df.at[i, "ecotype"]]
        df.at[i, "source"] = str(rng.choice(others))

    df = df.round(ROUNDING)
    truth = df[["strain_id", "ecotype"]].copy()
    truth["source_mismatch"] = df["source"] != df["ecotype"]
    return df[COLUMNS], truth


def generate_uniform(seed: int = UNIFORM_SEED, n_strains: int = 20):
    """Контрольный набор: код генерации из задания, без изменений."""
    np.random.seed(seed)
    strain_ids = [f"S{i:02d}" for i in range(1, n_strains + 1)]
    sources = ["soil", "water", "gut"]
    df = pd.DataFrame(
        {
            "strain_id": strain_ids,
            "growth_rate": np.random.uniform(0.5, 1.5, n_strains),
            "product_yield": np.random.uniform(1.5, 4.0, n_strains),
            "pH_tolerance": np.random.uniform(5, 9, n_strains),
            "GC_content": np.random.uniform(45, 65, n_strains),
            "genome_size_mbp": np.random.uniform(3, 6, n_strains),
            "has_plasmid": np.random.randint(0, 2, n_strains),
            "source": np.random.choice(sources, n_strains),
        }
    )
    return df[COLUMNS].round(ROUNDING)


def main():
    parser = argparse.ArgumentParser(description="Генерация входных данных.")
    parser.add_argument(
        "--preset",
        choices=["structured", "uniform", "both"],
        default="both",
        help="какой набор сгенерировать (по умолчанию оба)",
    )
    parser.add_argument("--seed", type=int, default=None, help="переопределить seed")
    args = parser.parse_args()

    if args.preset in ("structured", "both"):
        seed = args.seed if args.seed is not None else STRUCTURED_SEED
        features, truth = generate_structured(seed=seed)
        features.to_csv(DATA_DIR / "strains_features.csv", index=False)
        truth.to_csv(DATA_DIR / "strains_truth.csv", index=False)
        print(f"strains_features.csv: {len(features)} штаммов, seed={seed}, "
              f"источники {features['source'].value_counts().to_dict()}")
        print(f"strains_truth.csv:    скрытая истина, "
              f"{int(truth.source_mismatch.sum())} штаммов с несовпадающим источником")

    if args.preset in ("uniform", "both"):
        seed = args.seed if args.seed is not None else UNIFORM_SEED
        generate_uniform(seed=seed).to_csv(DATA_DIR / "strains_uniform.csv", index=False)
        print(f"strains_uniform.csv:  20 штаммов, seed={seed}, контрольный набор")


if __name__ == "__main__":
    main()
