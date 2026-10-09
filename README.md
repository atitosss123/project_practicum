# Кластеризация штаммов бактерий по множеству признаков

Учебный проект по дисциплине «Проектный практикум». Задача 5 курса «Технологии
искусственного интеллекта в биоэкономике»: 20 штаммов, семь признаков трёх разных типов,
автоматическое определение числа кластеров, визуализация.

## Результат

Оптимальное число кластеров 3, силуэт 0.494, размеры 6 / 7 / 7. Структура значимо
отличается от случайной (p = 0.005 по перестановочному тесту).

Три оговорки, без которых результат легко перетолковать:

| Что обнаружено | Чем подтверждено |
|---|---|
| Число кластеров определено неустойчиво: силуэт не различает k = 2, 3 и 5 | Разрыв между первым и вторым местом 0.004 при 20 объектах |
| На 20 объектах высокий силуэт сам по себе ничего не значит | На 300 равномерных наборах без структуры медиана силуэта 0.307, отдельные выборки доходят до 0.420 |
| Веса из условия отдают решающую роль категориальному признаку | Плоский штраф 0.2 за несовпадение источника = 75% разделяющей способности пяти количественных признаков; при весе 0.15 экотипы восстанавливаются полностью |

## Что где смотреть

| Что нужно | Куда идти |
|---|---|
| Отчёт для сдачи | [docs/REPORT.md](docs/REPORT.md) |
| Интерактивный стенд: покрутить веса и увидеть, как ломается разбиение | [stand/](stand/) |
| Пошаговый разбор с графиками | [notebooks/analysis.ipynb](notebooks/analysis.ipynb) |
| Принятые решения и отвергнутые варианты | [docs/PIPELINE.md](docs/PIPELINE.md) |
| База данных с результатами | [db/README.md](db/README.md) |
| Условие задачи | [docs/task-source.docx](docs/task-source.docx) |

## Запуск

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Сквозной прогон — таблицы и графики появятся в `out/`:

```bash
.venv/bin/python main.py
```

Контрольный набор без заложенной структуры:

```bash
.venv/bin/python main.py --data data/strains_uniform.csv --no-truth
```

Тесты:

```bash
.venv/bin/python -m pytest tests/ -q
```

База данных — хранит исходные данные и все посчитанные результаты:

```bash
brew services start postgresql@16 && createdb strains && .venv/bin/python db/load.py --create-schema
```

Стенд — статическая страница, пересобирается после изменения расчёта:

```bash
.venv/bin/python scripts/build_stand_data.py && .venv/bin/python stand/build.py
```

## Что где лежит

| Путь | Назначение |
|---|---|
| [data/generate_data.py](data/generate_data.py) | Генераторы обоих наборов данных |
| [data/seed_sweep.py](data/seed_sweep.py) | Перебор seed'ов: чем обоснован выбор основного набора |
| [src/loader.py](src/loader.py) | Чтение и валидация схемы |
| [src/preprocess.py](src/preprocess.py) | Стандартизация и one-hot кодировка |
| [src/distances.py](src/distances.py) | Смешанная матрица расстояний |
| [src/clustering.py](src/clustering.py) | Иерархическая кластеризация, подбор k по силуэту |
| [src/significance.py](src/significance.py) | Перестановочный тест значимости |
| [src/profiling.py](src/profiling.py) | Профили кластеров и проверки интерпретации |
| [src/sensitivity.py](src/sensitivity.py) | Устойчивость разбиения к выбору весов |
| [src/viz.py](src/viz.py) | Дендрограмма, PCA, MDS |
| [scripts/check_calibration.py](scripts/check_calibration.py) | Проверка калибровки теста значимости |
| [scripts/build_stand_data.py](scripts/build_stand_data.py) | Предрасчёт всех положений ползунка для стенда |
| [db/schema.sql](db/schema.sql) | Схема базы данных |
| [db/load.py](db/load.py) | Загрузка данных и результатов в базу |
| [db/queries.sql](db/queries.sql) | Восемь запросов, по одному на каждый вывод отчёта |
| [stand/index.html](stand/index.html) | Исходник интерактивного стенда |

Данные: [strains_features.csv](data/strains_features.csv) — основной набор с заложенными
экотипами, [strains_uniform.csv](data/strains_uniform.csv) — контрольный по коду из условия,
[strains_truth.csv](data/strains_truth.csv) — скрытая истина, в кластеризацию не подаётся.

Окружение: Python 3.9.6, numpy 2.0.2, pandas 2.3.3, scipy 1.13.1, scikit-learn 1.6.1,
matplotlib 3.9.4, PostgreSQL 16.15. Тестов 103.
