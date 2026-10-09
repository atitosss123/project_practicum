-- Схема базы данных проекта «Кластеризация штаммов бактерий».
--
-- Хранит и исходные данные, и результаты расчёта, чтобы любой показатель
-- в отчёте можно было поднять запросом, а не пересчитывать заново.
--
-- Структура повторяет ход пайплайна:
--   dataset            какие наборы данных вообще есть
--   strain             сами штаммы и их признаки
--   strain_truth       скрытая истина, в расчёт не подаётся
--   pair_distance      матрица непохожести, три компоненты и итог
--   clustering_run     один прогон кластеризации с конкретными весами
--   silhouette_score   профиль силуэта по числу кластеров
--   cluster_assignment кто в каком кластере
--   cluster_profile    средние признаки по кластерам
--   projection         координаты точек для графиков
--   weight_sensitivity как меняется результат при других весах

BEGIN;

DROP TABLE IF EXISTS weight_sensitivity, projection, cluster_profile,
    cluster_assignment, silhouette_score, clustering_run, pair_distance,
    strain_truth, strain, dataset CASCADE;
DROP TYPE IF EXISTS isolation_source CASCADE;
DROP VIEW IF EXISTS distance_matrix_full CASCADE;

-- Источник выделения - закрытый список, а не свободный текст: опечатка
-- «Soil» вместо «soil» молча создала бы четвёртую категорию и изменила
-- размерность кодировки.
CREATE TYPE isolation_source AS ENUM ('gut', 'soil', 'water');


CREATE TABLE dataset (
    id           smallint PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    name         text        NOT NULL UNIQUE,
    description  text        NOT NULL,
    generator    text        NOT NULL,              -- чем сгенерирован
    seed         integer     NOT NULL,
    has_truth    boolean     NOT NULL DEFAULT false,
    created_at   timestamptz NOT NULL DEFAULT now()
);

COMMENT ON TABLE dataset IS 'Наборы данных: основной со структурой и контрольный из условия';


CREATE TABLE strain (
    id               integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    dataset_id       smallint NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    strain_id        text     NOT NULL,
    growth_rate      numeric(6, 3) NOT NULL CHECK (growth_rate     BETWEEN 0    AND 5),
    product_yield    numeric(6, 3) NOT NULL CHECK (product_yield   BETWEEN 0    AND 20),
    ph_tolerance     numeric(6, 3) NOT NULL CHECK (ph_tolerance    BETWEEN 0    AND 14),
    gc_content       numeric(6, 3) NOT NULL CHECK (gc_content      BETWEEN 20   AND 80),
    genome_size_mbp  numeric(6, 3) NOT NULL CHECK (genome_size_mbp BETWEEN 0.1  AND 15),
    has_plasmid      boolean  NOT NULL,
    source           isolation_source NOT NULL,
    UNIQUE (dataset_id, strain_id)
);

COMMENT ON TABLE strain IS 'Штаммы и их признаки; границы повторяют валидацию в src/loader.py';
COMMENT ON COLUMN strain.has_plasmid IS 'Бинарный признак; в расчёте даёт расстояние Хэмминга 0 или 1';


CREATE TABLE strain_truth (
    strain_id        integer PRIMARY KEY REFERENCES strain(id) ON DELETE CASCADE,
    ecotype          isolation_source NOT NULL,
    source_mismatch  boolean NOT NULL
);

COMMENT ON TABLE strain_truth IS
    'Настоящая группа штамма. В кластеризацию НЕ подаётся, только для оценки качества';
COMMENT ON COLUMN strain_truth.source_mismatch IS
    'Источник выделения намеренно не совпадает с группой: проверка, перевесят ли измерения метку';


CREATE TABLE pair_distance (
    dataset_id    smallint NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    strain_a      integer  NOT NULL REFERENCES strain(id) ON DELETE CASCADE,
    strain_b      integer  NOT NULL REFERENCES strain(id) ON DELETE CASCADE,
    d_quantitative numeric(8, 6) NOT NULL CHECK (d_quantitative BETWEEN 0 AND 1),
    d_binary       numeric(8, 6) NOT NULL CHECK (d_binary       BETWEEN 0 AND 1),
    d_categorical  numeric(8, 6) NOT NULL CHECK (d_categorical  BETWEEN 0 AND 1),
    d_mixed        numeric(8, 6) NOT NULL CHECK (d_mixed        BETWEEN 0 AND 1),
    PRIMARY KEY (strain_a, strain_b),
    -- Пара хранится один раз: матрица симметрична, диагональ нулевая.
    CHECK (strain_a < strain_b)
);

CREATE INDEX ON pair_distance (dataset_id, d_mixed);

COMMENT ON TABLE pair_distance IS
    'Непохожесть пар штаммов: три компоненты после нормировки и взвешенный итог';


CREATE TABLE clustering_run (
    id              integer PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    dataset_id      smallint NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    linkage_method  text     NOT NULL,
    weight_quantitative numeric(4, 3) NOT NULL,
    weight_binary       numeric(4, 3) NOT NULL,
    weight_categorical  numeric(4, 3) NOT NULL,
    best_k          smallint NOT NULL CHECK (best_k >= 2),
    silhouette      numeric(6, 4) NOT NULL CHECK (silhouette BETWEEN -1 AND 1),
    is_conclusive   boolean  NOT NULL,
    plateau         smallint[] NOT NULL,
    p_value         numeric(6, 4),
    n_permutations  integer,
    ari_vs_truth    numeric(5, 3),
    ari_vs_source   numeric(5, 3),
    created_at      timestamptz NOT NULL DEFAULT now(),
    CHECK (weight_quantitative + weight_binary + weight_categorical = 1.0)
);

COMMENT ON COLUMN clustering_run.is_conclusive IS
    'false, если силуэт не различает несколько k: argmax выбран шумом';
COMMENT ON COLUMN clustering_run.plateau IS
    'Все k, чей силуэт отличается от лучшего не более чем на 0.01';
COMMENT ON COLUMN clustering_run.p_value IS
    'Перестановочный тест: доля случайных перестановок с силуэтом не хуже наблюдённого';


CREATE TABLE silhouette_score (
    run_id   integer  NOT NULL REFERENCES clustering_run(id) ON DELETE CASCADE,
    k        smallint NOT NULL CHECK (k >= 2),
    score    numeric(6, 4) NOT NULL,
    PRIMARY KEY (run_id, k)
);


CREATE TABLE cluster_assignment (
    run_id     integer  NOT NULL REFERENCES clustering_run(id) ON DELETE CASCADE,
    strain_id  integer  NOT NULL REFERENCES strain(id) ON DELETE CASCADE,
    cluster    smallint NOT NULL CHECK (cluster >= 1),
    PRIMARY KEY (run_id, strain_id)
);

CREATE INDEX ON cluster_assignment (run_id, cluster);


CREATE TABLE cluster_profile (
    run_id            integer  NOT NULL REFERENCES clustering_run(id) ON DELETE CASCADE,
    cluster           smallint NOT NULL CHECK (cluster >= 1),
    n_strains         smallint NOT NULL CHECK (n_strains > 0),
    growth_rate       numeric(6, 3) NOT NULL,
    product_yield     numeric(6, 3) NOT NULL,
    ph_tolerance      numeric(6, 3) NOT NULL,
    gc_content        numeric(6, 3) NOT NULL,
    genome_size_mbp   numeric(6, 3) NOT NULL,
    plasmid_share     numeric(4, 2) NOT NULL CHECK (plasmid_share BETWEEN 0 AND 1),
    source_mode       isolation_source NOT NULL,
    source_mode_share numeric(4, 2) NOT NULL CHECK (source_mode_share BETWEEN 0 AND 1),
    source_mix        text NOT NULL,
    PRIMARY KEY (run_id, cluster)
);

COMMENT ON TABLE cluster_profile IS
    'Средние в ИСХОДНЫХ единицах: проценты GC и мегабазы, а не стандартизованные значения';


CREATE TABLE projection (
    run_id     integer NOT NULL REFERENCES clustering_run(id) ON DELETE CASCADE,
    strain_id  integer NOT NULL REFERENCES strain(id) ON DELETE CASCADE,
    pc1        numeric(8, 4) NOT NULL,
    pc2        numeric(8, 4) NOT NULL,
    mds1       numeric(8, 4) NOT NULL,
    mds2       numeric(8, 4) NOT NULL,
    PRIMARY KEY (run_id, strain_id)
);

COMMENT ON TABLE projection IS
    'pc1/pc2 - главные компоненты признаков (требование условия); '
    'mds1/mds2 - проекция той матрицы, по которой кластеризовали';


CREATE TABLE weight_sensitivity (
    dataset_id          smallint NOT NULL REFERENCES dataset(id) ON DELETE CASCADE,
    weight_categorical  numeric(4, 3) NOT NULL,
    weight_quantitative numeric(4, 3) NOT NULL,
    best_k              smallint NOT NULL,
    silhouette          numeric(6, 4) NOT NULL,
    ari_vs_truth        numeric(5, 3),
    ari_vs_source       numeric(5, 3) NOT NULL,
    is_assignment       boolean NOT NULL,
    PRIMARY KEY (dataset_id, weight_categorical)
);

COMMENT ON TABLE weight_sensitivity IS
    'Как меняется разбиение при других весах. Главный результат проекта: '
    'веса из условия стоят на точке перелома';


-- Полная симметричная матрица: в таблице пара хранится один раз.
CREATE VIEW distance_matrix_full AS
SELECT dataset_id, strain_a AS strain_id, strain_b AS other_strain_id,
       d_quantitative, d_binary, d_categorical, d_mixed
FROM pair_distance
UNION ALL
SELECT dataset_id, strain_b, strain_a,
       d_quantitative, d_binary, d_categorical, d_mixed
FROM pair_distance
UNION ALL
SELECT s.dataset_id, s.id, s.id, 0, 0, 0, 0
FROM strain s;

COMMENT ON VIEW distance_matrix_full IS
    'Матрица непохожести целиком, включая нулевую диагональ и обе половины';

COMMIT;
