# News Dedup — семантическая дедупликация русских новостей

Группирует публикации об одном событии из разных источников, выбирает каноническую и возвращает кластеры `group_id → [news_id]`. Масштабируется на 100k+.

## Результаты

На выборке из 7 Telegram-каналов (1500 постов):

| Метрика | Значение |
|---|---|
| Постов после фильтра дайджестов | 1168 |
| Кластеров (событий) | 669 |
| Дублей убрано | 499 (43%) |
| Pairwise F1 (214 размеченных пар) | **0.978** |
| Pairwise Precision | 0.957 |
| Pairwise Recall | 1.000 |

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
             Граф рёбер: cosine ≥ 0.75 AND Δt ≤ 72h
                        │
            Connected components       ← union-find (scipy)
                        │
          Canonical: earliest pub_date
                        │
                  clusters.json
```

### Ключевые решения

**Модель: `deepvk/USER-bge-m3`** — лучший результат на RusBEIR (NDCG@10 61.13) и кластеризации новостей (AMI 0.84) среди открытых русскоязычных моделей.

**Кластеризация: connected components** — вместо K-means (не знаем число кластеров) и DBSCAN (O(n²) на 100k). FAISS ANN граф + union-find: O(n·k) память, инкрементальное пополнение.

**Порог tau=0.75** откалиброван на размеченных парах по pairwise F1:

| tau | F1 | P | R |
|---|---|---|---|
| 0.70 | 0.917 | 0.846 | 1.000 |
| **0.75** | **0.978** | **0.957** | **1.000** |
| 0.80 | 0.977 | 1.000 | 0.955 |
| 0.85 | 0.667 | 1.000 | 0.500 |

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

## Структура

```
ingest/
  store.py          SQLite: схема, upsert, checkpoint
  telegram.py       Telethon MTProto → SQLite (инкрементально)
dedup/
  text.py           clean(), word_tokens()
  prefilter.py      MinHash/LSH лексический пре-фильтр
  embed.py          sentence-transformers → L2-norm
  cluster.py        FAISS ANN + build_edges + connected_clusters + canonical
  eval.py           pairwise P/R/F1, ARI, cluster_size_stats
  pipeline.py       оркестратор + вывод
scripts/
  scrape_fixture.py t.me/s/<channel> без авторизации
  label_pairs.py    интерактивная разметка пар
  calibrate_tau.py  grid search косинусного порога
tests/
  test_smoke.py     offline unit-тесты (F1=1.0, time window)
  fixtures/
    news_sample.json      1500 постов, 7 каналов
    labeled_pairs.json    214 размеченных пар
config.yaml         пороги, каналы, параметры модели
```
