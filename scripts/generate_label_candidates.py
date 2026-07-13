"""Generator of candidate pairs for labeling.

Strategy: pairs from the cosine "gray zone" [lo, hi] give the largest gain
from labeling — that's where the tau boundary lies. We additionally take:
- pairs IN the same cluster with cosine below the median (potential FPs via transitivity)
- pairs NOT in the same cluster but with high cosine (potential FNs)

Output format: JSON compatible with labeled_pairs.json (label=null until labeled).
The text of both posts is printed for manual labeling in the terminal.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import yaml

FIXTURE = Path("tests/fixtures/news_100k.json.gz")
LABELS = Path("tests/fixtures/labeled_pairs.json")
CLUSTERS = Path("clusters.json")
CACHE = "embeddings_cache.npz"
CONFIG = Path("config.yaml")


def load():
    import gzip
    cfg = yaml.safe_load(CONFIG.read_text())["dedup"]
    result = json.loads(CLUSTERS.read_text())
    with gzip.open(FIXTURE, "rt", encoding="utf-8") as f:
        news = {p["id"]: p for p in json.load(f)}
    id_to_cluster = {r["id"]: int(r["cluster_id"]) for r in result["per_news"]}
    existing = {
        (p["id1"], p["id2"]) for p in json.loads(LABELS.read_text())
    } | {
        (p["id2"], p["id1"]) for p in json.loads(LABELS.read_text())
    }
    cache = np.load(cfg["embeddings_cache"], allow_pickle=True)
    vec = {str(k): v for k, v in zip(cache["ids"], cache["vectors"])}
    return news, id_to_cluster, existing, vec


def cos(vec, a, b):
    return float(np.dot(vec[a], vec[b]))


def sample_grey_zone(vec, id_to_cluster, existing, news, lo=0.76, hi=0.84, n=50, seed=42):
    """Random pairs from the cache with cosine in [lo, hi], not already in existing."""
    rng = random.Random(seed)
    # Only posts with enough text (min 40 chars after strip)
    ids = [k for k in vec if k in id_to_cluster and k in news
           and len(((news[k].get("title") or "") + (news[k].get("text") or "")).strip()) >= 40]
    rng.shuffle(ids)
    pairs = []
    checked = 0
    for i, a in enumerate(ids):
        if len(pairs) >= n:
            break
        # Compare against a small window of neighbors in the shuffled list
        for b in ids[i+1:i+200]:
            if len(pairs) >= n:
                break
            key = (min(a, b), max(a, b))
            if key in existing:
                continue
            c = cos(vec, a, b)
            if lo <= c <= hi:
                pairs.append((a, b, c))
                existing.add(key)
            checked += 1
    print(f"Проверено ~{checked} пар, найдено {len(pairs)} в зоне [{lo}, {hi}]")
    return pairs


def parse_dt(n):
    from datetime import datetime, timezone
    d = (n.get("published_at") or "")[:19]
    try:
        return datetime.fromisoformat(d).replace(tzinfo=timezone.utc)
    except Exception:
        return None


def auto_label(pairs, news, auto_threshold_hours=24):
    """Pairs with |Δt| > threshold -> auto-label 0 (different events), don't show them."""
    auto, manual = [], []
    for a, b, c in pairs:
        ta = parse_dt(news.get(a, {}))
        tb = parse_dt(news.get(b, {}))
        if ta and tb and abs((ta - tb).total_seconds()) > auto_threshold_hours * 3600:
            auto.append((a, b, c, 0))
        else:
            manual.append((a, b, c))
    if auto:
        print(f"Авто-разметка (|Δt| > {auto_threshold_hours}ч): {len(auto)} пар → label=0")
    return auto, manual


def interactive_label(pairs, news, id_to_cluster):
    """Interactive labeling: shows a pair, prompts for 0/1/s(skip)/q(quit)."""
    results = []
    print("\n" + "="*60)
    print("РАЗМЕТКА ПАР")
    print("1 = дубли (одно событие)   0 = разные события")
    print("s = пропустить   q = завершить")
    print("="*60)
    def post_text(n):
        title = (n.get("title") or "").strip()
        text = (n.get("text") or "").strip()
        if title and text.startswith(title):
            return text
        return (title + " " + text).strip() if title else text

    def wrap(text, width=72, indent="    "):
        import textwrap
        return "\n".join(indent + line for line in textwrap.wrap(text, width))

    def fmt_date(n):
        d = n.get("published_at") or ""
        return d[:16].replace("T", " ") if d else "?"

    for idx, (a, b, c) in enumerate(pairs, 1):
        na = news.get(a, {}); nb = news.get(b, {})
        same_cluster = id_to_cluster.get(a) == id_to_cluster.get(b)
        cluster_tag = "⚠ ОДИН КЛАСТЕР" if same_cluster else "○ разные кластеры"
        print("\n" + "─" * 72)
        print(f"  Пара {idx}/{len(pairs)}   cosine={c:.3f}   {cluster_tag}")
        print("─" * 72)
        print(f"  [A] {a}  ({fmt_date(na)})")
        print(wrap(post_text(na)[:300]))
        print()
        print(f"  [B] {b}  ({fmt_date(nb)})")
        print(wrap(post_text(nb)[:300]))
        print()
        print("  1 = дубли  │  0 = разные события  │  s = пропустить  │  q = выйти")
        while True:
            ch = input("  > ").strip().lower()
            if ch in ("0", "1"):
                results.append({
                    "id1": a, "id2": b,
                    "label": int(ch),
                    "source1": a.split(":")[0],
                    "source2": b.split(":")[0],
                    "cosine": round(c, 4),
                })
                break
            elif ch == "s":
                break
            elif ch == "q":
                return results
            else:
                print("  Введи 0, 1, s или q")
    return results


def merge_and_save(new_pairs, labels_path):
    existing = json.loads(Path(labels_path).read_text())
    seen = {(p["id1"], p["id2"]) for p in existing}
    added = 0
    for p in new_pairs:
        if (p["id1"], p["id2"]) not in seen:
            # remove the internal cosine field from the final file
            entry = {k: v for k, v in p.items() if k != "cosine"}
            existing.append(entry)
            seen.add((p["id1"], p["id2"]))
            added += 1
    Path(labels_path).write_text(json.dumps(existing, ensure_ascii=False, indent=2))
    print(f"\nДобавлено {added} пар → {labels_path} (всего {len(existing)})")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="Кандидатов для разметки")
    ap.add_argument("--lo", type=float, default=0.76)
    ap.add_argument("--hi", type=float, default=0.84)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--auto-hours", type=float, default=24,
                    help="Пары с |Δt| > N часов авто-метка 0 (default: 24)")
    ap.add_argument("--dry-run", action="store_true", help="Показать кандидатов без разметки")
    args = ap.parse_args()

    print("Загружаю данные...")
    news, id_to_cluster, existing, vec = load()
    print(f"Постов: {len(vec)}, существующих пар: {len(existing)//2}")

    pairs = sample_grey_zone(vec, id_to_cluster, existing, news, lo=args.lo, hi=args.hi, n=args.n, seed=args.seed)

    def post_text(n):
        title = (n.get("title") or "").strip()
        text = (n.get("text") or "").strip()
        if title and text.startswith(title):
            return text
        return (title + " " + text).strip() if title else text

    if args.dry_run:
        for a, b, c in pairs[:10]:
            na = news.get(a, {}); nb = news.get(b, {})
            print(f"\ncos={c:.3f}  {a} <-> {b}")
            print(f"  A: {post_text(na)[:100]}")
            print(f"  B: {post_text(nb)[:100]}")
        print(f"\n... итого {len(pairs)} кандидатов")
        return

    auto, manual = auto_label(pairs, news, auto_threshold_hours=args.auto_hours)
    print(f"К разметке вручную: {len(manual)} пар\n")

    auto_results = [
        {"id1": a, "id2": b, "label": lbl,
         "source1": a.split(":")[0], "source2": b.split(":")[0], "cosine": round(c, 4)}
        for a, b, c, lbl in auto
    ]
    manual_results = interactive_label(manual, news, id_to_cluster)
    labeled = auto_results + manual_results

    if labeled:
        merge_and_save(labeled, LABELS)
        # Quick evaluation after labeling
        print("\nПересчёт F1 на обновлённой разметке...")
        import subprocess, sys
        subprocess.run([sys.executable, "scripts/evaluate_quality.py"])
    else:
        print("Ничего не размечено.")


if __name__ == "__main__":
    main()
