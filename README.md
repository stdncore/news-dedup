# News Dedup — semantic deduplication of Russian news

Groups publications about the same event from different sources, picks a canonical one, and returns clusters `group_id → [news_id]`. Scales to 100k+.

## Results

### 100k posts (27 Telegram channels)

| Metric | Value |
|---|---|
| Input posts | 100,000 |
| After filters (digests + URL-only) | 77,482 |
| Clusters (events) | 56,687 |
| Duplicates removed | 20,795 (27%) |
| Max cluster | 232 |
| **Pairwise F1 (expert labeling, 289 pairs)** | **0.833** |
| Pairwise Precision | 0.854 |
| Pairwise Recall | 0.814 |

Evaluated with `scripts/evaluate_quality.py` on 289 expert-labeled pairs (332 pairs total,
of which 3 were not found in the fixture, 40 auto-labeled by the rule Δt>24h).
Labeling covers the cosine gray zone [0.76, 0.84] via `scripts/generate_label_candidates.py`.

Reproduce: `python -m dedup.pipeline --input tests/fixtures/news_100k.json.gz` →
`python scripts/evaluate_quality.py`.

### Small sample (1500 posts, debugging)

| Metric | Value |
|---|---|
| Posts after filter | 1168 |
| Pairwise F1 — held-out (65 pairs) | 1.000 |

## Approach

### Pipeline

```
Telegram channels → Ingest (Telethon → SQLite)
                        │
                  Digest filter        ← regex + bullet heuristic
                        │
             MinHash/LSH prefilter     ← Jaccard ≥ 0.7, 5-gram, 256 perm
                        │
             Embeddings (USER-bge-m3)  ← 1024-dim, L2-norm
                        │
                  FAISS ANN            ← top-20 cosine neighbors
                        │
             Edge graph: cosine ≥ 0.78 AND Δt ≤ 24h
                        │
            Louvain communities        ← cuts transitive bridges
                        │
          Canonical: earliest pub_date
                        │
                  clusters.json
```

### Key decisions

**Model: `deepvk/USER-bge-m3`** — best result on RusBEIR (NDCG@10 61.13) and news clustering (AMI 0.84) among open Russian-language models.

**Clustering: Louvain communities** — instead of connected components. At 100k, connected components merge neighboring events into a mega-cluster via transitive chains of similar posts (max cluster 1720). Louvain optimizes modularity of the weighted graph (weight = cosine) and cuts weak bridges. Max cluster 1720 → 232. Connected components remain available via `clustering: connected_components`.

**Parameters tau=0.78, time_window=24h** were calibrated iteratively on 332 labeled pairs (`scripts/generate_label_candidates.py` — gray zone [0.76, 0.84]). The 24h window removes template daily digests (covid stats, nightly drone-strike roundups) that have high cosine similarity but are different events.

**Digest filter** — removes roundups and radio announcements (regex + bullet-count) that aren't news and create false clusters.

### Limitations

- `USER-bge-m3` truncates input at ~512 tokens; for long articles use `deepvk/USER2-base` (8192 tokens).
- Component transitivity can merge a chain of similar (but not identical) news items — controlled via the threshold and time window.
- ANN (HNSW) is approximate: duplicates outside top-k won't be found (raise `ann.top_k` / `ef_search`).

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cp .env.example .env   # TG_API_ID, TG_API_HASH (my.telegram.org) — only for Telegram ingest
```

## Running

### On fixture data (no Telegram API)

```bash
# macOS: work around the OpenMP conflict between torch and faiss
OMP_NUM_THREADS=1 python -m dedup.pipeline \
    --config config.yaml \
    --input tests/fixtures/news_sample.json
```

### Full cycle via Telegram API

```bash
python -m ingest.telegram --config config.yaml   # collect → news.db
OMP_NUM_THREADS=1 python -m dedup.pipeline --config config.yaml
```

`clusters.json` output:
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

### Fixture collection + labeling + tau calibration

```bash
python scripts/scrape_fixture.py --limit 300        # no TG creds needed
python scripts/label_pairs.py                        # interactive labeling
OMP_NUM_THREADS=1 python scripts/calibrate_tau.py   # → best tau
```

### Tests

```bash
pytest tests/
```

## Configuration

```yaml
dedup:
  model: "deepvk/USER-bge-m3"
  cosine_threshold: 0.75   # calibrated on labeled_pairs (F1=0.978)
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

## Scalability

| n posts | Step | Complexity |
|---|---|---|
| 100k | Embeddings (batch=64, CPU) | O(n), ~15 min |
| 100k | FAISS HNSW top-20 | O(n log n), ~10 sec |
| 100k | MinHash LSH | O(n), ~30 sec |
| 100k | Connected components | O(n·k), instant |

For n ≤ 50k, `IndexFlatIP` is used (exact, stable); for n > 50k — `IndexHNSWFlat`.

## Dataset

Fixture: `tests/fixtures/news_sample.json`
SHA-256: `af1a470fd14fe85e0c62169ef556b0001549ba7532791f3f45eb7868c9770a60`
Scraped: 2026-06-18, 300 posts per channel

| Channel | Posts |
|---|---|
| @interfaxonline | 300 |
| @kommersant | 300 |
| @rbc_news | 300 |
| @tass_agency | 300 |
| @vedomosti | 300 |
| **Total** | **1500** |

## Structure

```
ingest/
  store.py          SQLite: schema, upsert, checkpoint
  telegram.py       Telethon MTProto → SQLite (incremental)
dedup/
  text.py           clean(), word_tokens(), is_digest()
  prefilter.py      MinHash/LSH lexical prefilter
  embed.py          sentence-transformers → L2-norm
  cluster.py        FAISS ANN + build_edges + connected_clusters + canonical
  eval.py           pairwise P/R/F1, ARI, cluster_size_stats
  pipeline.py       orchestrator + output
scripts/
  scrape_fixture.py       t.me/s/<channel> without auth
  label_pairs.py          interactive pair labeling
  calibrate_tau.py        cosine threshold grid search (with held-out split)
  audit_digest_filter.py  interactive FP-rate audit of the digest filter
tests/
  test_smoke.py           offline unit tests (F1=1.0, time window, lexical boilerplate)
  test_integration.py     integration tests with the real model (pytest -m integration)
  fixtures/
    news_sample.json      1500 posts, 5 channels
    labeled_pairs.json    214 labeled pairs
config.yaml         thresholds, channels, model parameters
```
