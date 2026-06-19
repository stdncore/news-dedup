"""Аудит FP-rate digest-фильтра.

Загружает fixture, прогоняет is_digest(), берёт случайную выборку отброшенных
постов и предлагает разметить их вручную (y = правда дайджест, n = ошибочно отброшен).
Сохраняет результат в tests/fixtures/digest_audit.json.

Запуск:
    python scripts/audit_digest_filter.py --sample 50
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

FIXTURE = Path("tests/fixtures/news_sample.json")
AUDIT_OUT = Path("tests/fixtures/digest_audit.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture", default=str(FIXTURE))
    ap.add_argument("--out", default=str(AUDIT_OUT))
    ap.add_argument("--sample", type=int, default=50, help="Число постов для разметки")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from dedup.text import is_digest

    posts = json.loads(Path(args.fixture).read_text(encoding="utf-8"))
    total = len(posts)
    filtered = [p for p in posts if is_digest(p.get("text", ""))]
    kept = total - len(filtered)

    print(f"Всего постов: {total}")
    print(f"Отброшено фильтром: {len(filtered)} ({100*len(filtered)/total:.1f}%)")
    print(f"Оставлено: {kept}\n")

    if not filtered:
        print("Ничего не отброшено — аудит не нужен.")
        return

    rng = random.Random(args.seed)
    sample = rng.sample(filtered, min(args.sample, len(filtered)))
    print(f"Размечаем {len(sample)} случайных отброшенных постов.")
    print("y = правда дайджест (фильтр прав), n = ложное срабатывание (FP)\n")
    print("-" * 60)

    results = []
    fp_count = 0
    for i, post in enumerate(sample, 1):
        text = (post.get("text") or "").strip()
        preview = text[:300].replace("\n", " ")
        print(f"\n[{i}/{len(sample)}] id={post.get('id')}  source={post.get('source')}")
        print(f"  {preview}{'...' if len(text) > 300 else ''}")
        while True:
            ans = input("  Дайджест? [y/n/q]: ").strip().lower()
            if ans in ("y", "n", "q"):
                break
            print("  Введи y, n или q (выход)")
        if ans == "q":
            print("Прервано.")
            break
        is_true_digest = ans == "y"
        if not is_true_digest:
            fp_count += 1
        results.append({
            "id": post.get("id"),
            "source": post.get("source"),
            "is_true_digest": is_true_digest,
            "text_preview": text[:200],
        })

    if results:
        fp_rate = fp_count / len(results)
        print(f"\n--- Результат ---")
        print(f"Размечено: {len(results)}  FP: {fp_count}  FP-rate: {fp_rate:.1%}")
        Path(args.out).write_text(
            json.dumps(
                {"fp_count": fp_count, "labeled": len(results), "fp_rate": fp_rate, "items": results},
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Сохранено в {args.out}")
        print(f"\nДобавь в README: FP digest-фильтра: {fp_count} из {len(results)} ({fp_rate:.1%})")


if __name__ == "__main__":
    main()
