# News Dedup — семантическая дедупликация русских новостей

Группирует публикации об одном событии из разных источников, выбирает каноническую и возвращает кластеры `group_id → [news_id]`. Масштабируется на 100k+.

## Результаты

### 100k постов (27 Telegram-каналов)

| Метрика | Значение |
|---|---|
| Постов на входе | 100 000 |
| После фильтров (дайджесты + URL-only) | 77 482 |
| Кластеров (событий) | 56 383 |
| Дублей убрано | 21 099 (27%) |
| Макс. кластер | 423 |
| **Pairwise F1 (экспертная разметка, 211 пар)** | **0.933** |
| Pairwise Precision | 0.875 |
| Pairwise Recall | 1.000 |

Оценка по `scripts/evaluate_quality.py` на экспертно-размеченных парах `labeled_pairs.json`.
3 остаточных FP — грани одного инфоповода (массированная атака БПЛА на Москву: число
сбитых / закрытие аэропортов / удар по НПЗ), граничный случай разметки.

Воспроизведение: `python -m dedup.pipeline --input tests/fixtures/news_100k.json.gz` →
`python scripts/evaluate_quality.py`.

### Малая выборка (1500 постов, отладка)

| Метрика | Значение |
|---|---|
| Постов после фильтра | 1168 |
| Pairwise F1 — held-out (65 пар) | 1.000 |

## Подход

### Пайплайн

```
Telegram-каналы → Ingest (Telethon → SQLite)
                        │
                  Фильтр дайджестов   ← regex + bullet-эвристика
                        │
              MinHash/LSH пре-фильтр  ← Jaccard ≥ 0.7, 5-gram, 256 perm
                        │
             Эмбеддинги (USER-bge-m3) ← 1024-dim, L2-norm
                        │
                  FAISS ANN            ← top-20 соседей по косинусу
                        │
             Граф рёбер: cosine ≥ 0.80 AND Δt ≤ 72h
                        │
            Louvain communities        ← режет транзитивные мостики
                        │
          Canonical: earliest pub_date
                        │
                  clusters.json
```

### Ключевые решения

**Модель: `deepvk/USER-bge-m3`** — лучший результат на RusBEIR (NDCG@10 61.13) и кластеризации новостей (AMI 0.84) среди открытых русскоязычных моделей.

**Кластеризация: Louvain communities** — вместо connected components (union-find). На 100k connected-components страдает транзитивным слипанием: горячий инфоповод (атака БПЛА, война) через цепочку похожих постов сливает соседние события в мегакластер (max-кластер был 1720). Louvain оптимизирует модулярность взвешенного графа (вес = косинус) и режет слабые межсобытийные мостики. Результат: F1 0.857 → **0.933**, max-кластер 1720 → 423, recall=1.0 сохраняется. Connected components доступен через `clustering: connected_components`.

**Порог tau=0.75** откалиброван на train-части (149 пар) по pairwise F1:

| tau | F1 (train) | P | R |
|---|---|---|---|
| 0.70 | 0.875 | 0.778 | 1.000 |
| **0.75** | **0.966** | **0.933** | **1.000** |
| 0.80 | 0.963 | 1.000 | 0.929 |
| 0.83 | 0.667 | 1.000 | 0.500 |

**Фильтр дайджестов** — удаляет сводки и радио-анонсы (regex + bullet-count), которые не являются новостями и создают ложные кластеры.

### Ограничения

- `USER-bge-m3` обрезает вход на ~512 токенах; для длинных статей — `deepvk/USER2-base` (8192 токенов).
- Транзитивность компонент может склеить цепочку похожих (но не одинаковых) новостей — контролируется порогом и временным окном.
- ANN (HNSW) приближённый: дубли вне top-k не найдутся (повысить `ann.top_k` / `ef_search`).

## Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # TG_API_ID, TG_API_HASH (my.telegram.org) — только для Telegram ingest
```

## Запуск

### На fixture-данных (без Telegram API)

```bash
# macOS: обход конфликта OpenMP между torch и faiss
OMP_NUM_THREADS=1 python -m dedup.pipeline \
    --config config.yaml \
    --input tests/fixtures/news_sample.json
```

### Полный цикл через Telegram API

```bash
python -m ingest.telegram --config config.yaml   # сбор → news.db
OMP_NUM_THREADS=1 python -m dedup.pipeline --config config.yaml
```

Выход `clusters.json`:
```json
{
  "clusters": {
    "0": ["@tass_agency:12345", "@rbc_news:67890"],
    "1": ["@kommersant:11111"]
  },
  "per_news": [
    {"id": "@tass_agency:12345", "cluster_id": 0, "is_canonical": true}
  ]
}
```

### Сбор fixture + разметка + калибровка tau

```bash
python scripts/scrape_fixture.py --limit 300        # без TG-кредов
python scripts/label_pairs.py                        # интерактивная разметка
OMP_NUM_THREADS=1 python scripts/calibrate_tau.py   # → лучший tau
```

### Тесты

```bash
pytest tests/
```

## Конфигурация

```yaml
dedup:
  model: "deepvk/USER-bge-m3"
  cosine_threshold: 0.75   # откалибровано по labeled_pairs (F1=0.978)
  time_window_hours: 72
  canonical: "earliest"    # earliest | longest
  prefilter:
    enabled: true
    jaccard_threshold: 0.7
    num_perm: 256
    shingle_size: 5
  ann:
    top_k: 20
    hnsw_m: 32
    ef_search: 64
```

## Масштабируемость

| n постов | Шаг | Сложность |
|---|---|---|
| 100k | Эмбеддинги (batch=64, CPU) | O(n), ~15 мин |
| 100k | FAISS HNSW top-20 | O(n log n), ~10 сек |
| 100k | MinHash LSH | O(n), ~30 сек |
| 100k | Connected components | O(n·k), мгновенно |

При n ≤ 50k используется `IndexFlatIP` (точный, стабильный); при n > 50k — `IndexHNSWFlat`.

## Датасет

Fixture: `tests/fixtures/news_sample.json`  
SHA-256: `af1a470fd14fe85e0c62169ef556b0001549ba7532791f3f45eb7868c9770a60`  
Скрейп: 2026-06-18, по 300 постов на канал

| Канал | Постов |
|---|---|
| @interfaxonline | 300 |
| @kommersant | 300 |
| @rbc_news | 300 |
| @tass_agency | 300 |
| @vedomosti | 300 |
| **Итого** | **1500** |

## Структура

```
ingest/
  store.py          SQLite: схема, upsert, checkpoint
  telegram.py       Telethon MTProto → SQLite (инкрементально)
dedup/
  text.py           clean(), word_tokens(), is_digest()
  prefilter.py      MinHash/LSH лексический пре-фильтр
  embed.py          sentence-transformers → L2-norm
  cluster.py        FAISS ANN + build_edges + connected_clusters + canonical
  eval.py           pairwise P/R/F1, ARI, cluster_size_stats
  pipeline.py       оркестратор + вывод
scripts/
  scrape_fixture.py       t.me/s/<channel> без авторизации
  label_pairs.py          интерактивная разметка пар
  calibrate_tau.py        grid search косинусного порога (с held-out split)
  audit_digest_filter.py  интерактивный аудит FP-rate digest-фильтра
tests/
  test_smoke.py           offline unit-тесты (F1=1.0, time window, lexical boilerplate)
  test_integration.py     integration-тесты с реальной моделью (pytest -m integration)
  fixtures/
    news_sample.json      1500 постов, 5 каналов
    labeled_pairs.json    214 размеченных пар
config.yaml         пороги, каналы, параметры модели
```
