-- Запросы к базе проекта. Каждый поднимает число, которое есть в отчёте,
-- то есть показывает, что база хранит результат целиком, а не выборочно.
-- Запуск по одному:  psql strains -f db/queries.sql

\echo '=== 1. Какие наборы данных загружены ==='
SELECT name, description, seed, has_truth,
       (SELECT count(*) FROM strain WHERE dataset_id = d.id) AS strains
FROM dataset d ORDER BY id;

\echo ''
\echo '=== 2. Профили кластеров основного набора ==='
SELECT p.cluster, p.n_strains,
       p.growth_rate AS "рост", p.product_yield AS "продукт",
       p.gc_content AS "GC", p.genome_size_mbp AS "геном",
       round(p.plasmid_share * 100) || '%' AS "с плазмидой",
       p.source_mix AS "источники"
FROM cluster_profile p
JOIN clustering_run r ON r.id = p.run_id
JOIN dataset d ON d.id = r.dataset_id
WHERE d.name = 'structured'
ORDER BY p.cluster;

\echo ''
\echo '=== 3. Профиль силуэта: насколько выражен максимум ==='
-- Показывает плато: разница между лучшим и вторым местом меньше допуска.
WITH run AS (
    SELECT r.id FROM clustering_run r JOIN dataset d ON d.id = r.dataset_id
    WHERE d.name = 'structured' LIMIT 1
)
SELECT k, score,
       round(max(score) OVER () - score, 4) AS "отставание от лучшего",
       CASE WHEN max(score) OVER () - score <= 0.01 THEN 'неотличимо' ELSE '' END AS "пометка"
FROM silhouette_score WHERE run_id = (SELECT id FROM run)
ORDER BY k;

\echo ''
\echo '=== 4. Самые похожие и самые непохожие пары штаммов ==='
(SELECT 'самые похожие' AS "тип", a.strain_id AS "штамм A", b.strain_id AS "штамм B",
        pd.d_mixed AS "непохожесть"
 FROM pair_distance pd
 JOIN strain a ON a.id = pd.strain_a JOIN strain b ON b.id = pd.strain_b
 JOIN dataset d ON d.id = pd.dataset_id WHERE d.name = 'structured'
 ORDER BY pd.d_mixed LIMIT 3)
UNION ALL
(SELECT 'самые непохожие', a.strain_id, b.strain_id, pd.d_mixed
 FROM pair_distance pd
 JOIN strain a ON a.id = pd.strain_a JOIN strain b ON b.id = pd.strain_b
 JOIN dataset d ON d.id = pd.dataset_id WHERE d.name = 'structured'
 ORDER BY pd.d_mixed DESC LIMIT 3);

\echo ''
\echo '=== 5. Сколько различных значений принимает каждая компонента ==='
-- Ключевой замер проекта: две компоненты из трёх ступенчатые.
SELECT count(DISTINCT d_quantitative) AS "измеримые величины",
       count(DISTINCT d_binary)       AS "плазмида",
       count(DISTINCT d_categorical)  AS "источник",
       count(*)                       AS "всего пар"
FROM pair_distance pd JOIN dataset d ON d.id = pd.dataset_id
WHERE d.name = 'structured';

\echo ''
\echo '=== 6. Куда попали штаммы с неверной меткой источника ==='
-- mode() - сортирующий агрегат, оконной функцией он быть не может,
-- поэтому преобладающая группа кластера считается отдельным подзапросом.
WITH run AS (
    SELECT r.id FROM clustering_run r JOIN dataset d ON d.id = r.dataset_id
    WHERE d.name = 'structured' LIMIT 1
), dominant AS (
    SELECT ca.cluster, mode() WITHIN GROUP (ORDER BY t.ecotype) AS cluster_ecotype
    FROM cluster_assignment ca
    JOIN strain_truth t ON t.strain_id = ca.strain_id
    WHERE ca.run_id = (SELECT id FROM run)
    GROUP BY ca.cluster
)
SELECT s.strain_id AS "штамм", t.ecotype AS "настоящая группа",
       s.source AS "метка в таблице", ca.cluster AS "кластер",
       dom.cluster_ecotype AS "преобладает в кластере",
       (t.ecotype = dom.cluster_ecotype) AS "пошёл за своими"
FROM strain s
JOIN strain_truth t ON t.strain_id = s.id
JOIN cluster_assignment ca ON ca.strain_id = s.id AND ca.run_id = (SELECT id FROM run)
JOIN dominant dom ON dom.cluster = ca.cluster
WHERE t.source_mismatch
ORDER BY s.strain_id;

\echo ''
\echo '=== 7. Главный результат: устойчивость к весам ==='
SELECT weight_quantitative AS "измеримые", weight_categorical AS "каждая пометка",
       best_k AS "групп", silhouette AS "силуэт",
       ari_vs_truth AS "совпало с группой", ari_vs_source AS "совпало с меткой",
       CASE WHEN is_assignment THEN '<- веса из условия' ELSE '' END AS "пометка"
FROM weight_sensitivity ws JOIN dataset d ON d.id = ws.dataset_id
WHERE d.name = 'structured'
ORDER BY weight_categorical;

\echo ''
\echo '=== 8. Сводка по обоим наборам ==='
SELECT d.name AS "набор", r.best_k AS "групп", r.silhouette AS "силуэт",
       r.is_conclusive AS "определено однозначно", r.plateau AS "неотличимые k",
       r.p_value AS "p-значение", r.ari_vs_truth AS "совпало с группой",
       r.ari_vs_source AS "совпало с меткой"
FROM clustering_run r JOIN dataset d ON d.id = r.dataset_id
ORDER BY d.name;
