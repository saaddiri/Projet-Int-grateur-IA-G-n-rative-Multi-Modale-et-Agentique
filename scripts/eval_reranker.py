"""Compare les configurations de reranking (livrable L2.5).

Pour chaque question de test, la liste de candidats est celle de la recherche
vectorielle (top 20, enregistrée lors de la construction du jeu de données).
On mesure la position du passage positif selon trois configurations :
- sans reranker (ordre de la recherche vectorielle) ;
- reranker initial (cross-encoder multilingue pré-entraîné) ;
- reranker adapté (fine-tuné sur le corpus).

Métriques : nDCG@5 et MRR@10 (un seul passage pertinent par question),
plus Recall@5 et le temps moyen de reranking.

Usage :  python -m scripts.eval_reranker [--split test]
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from maintidoc.config import ROOT_DIR
from maintidoc.reranker import ADAPTED_RERANKER, BASE_RERANKER, rerank

DATA_DIR = ROOT_DIR / "data" / "reranker"


def ndcg_at_k(rank: int | None, k: int = 5) -> float:
    return 1.0 / math.log2(rank + 1) if rank is not None and rank <= k else 0.0


def mrr_at_k(rank: int | None, k: int = 10) -> float:
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def evaluate(questions: list[dict], passages: dict[str, str], model: str | None) -> dict:
    scores = {"ndcg@5": 0.0, "mrr@10": 0.0, "recall@5": 0.0}
    by_lang: dict[str, list[float]] = {}
    elapsed = 0.0
    for q in questions:
        candidates = [c for c in q["candidates"] if c in passages]
        start = time.perf_counter()
        ranked = rerank(q["question"], candidates, text_of=lambda c: passages[c], model=model)
        elapsed += time.perf_counter() - start
        order = [c for c, _ in ranked]
        rank = order.index(q["node_id"]) + 1 if q["node_id"] in order else None
        scores["ndcg@5"] += ndcg_at_k(rank)
        scores["mrr@10"] += mrr_at_k(rank)
        scores["recall@5"] += 1.0 if rank is not None and rank <= 5 else 0.0
        by_lang.setdefault(q["lang"], []).append(ndcg_at_k(rank))
    n = max(1, len(questions))
    result = {name: round(value / n, 4) for name, value in scores.items()}
    result["ndcg@5_par_langue"] = {lang: round(sum(v) / len(v), 4) for lang, v in by_lang.items()}
    result["ms_par_question"] = round(1000 * elapsed / n, 1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["train", "test"])
    args = parser.parse_args()

    questions = [json.loads(l) for l in (DATA_DIR / "questions.jsonl").read_text(encoding="utf-8").splitlines()]
    questions = [q for q in questions if q["split"] == args.split]
    passages = {p["node_id"]: p["text"] for p in
                (json.loads(l) for l in (DATA_DIR / "passages.jsonl").read_text(encoding="utf-8").splitlines())}
    in_pool = sum(q["node_id"] in q["candidates"] for q in questions)
    print(f"{len(questions)} questions ({args.split}) ; positif présent dans le top 20 : {in_pool}/{len(questions)}")

    configs = {"sans reranker": None, "reranker initial": BASE_RERANKER}
    if Path(ADAPTED_RERANKER).exists():
        configs["reranker adapté"] = str(ADAPTED_RERANKER)

    results = {}
    for name, model in configs.items():
        results[name] = evaluate(questions, passages, model)
        print(f"{name:18s} {results[name]}", flush=True)

    report = {"split": args.split, "questions": len(questions), "positive_in_top20": in_pool, "results": results}
    out = DATA_DIR / f"eval_{args.split}.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Résultats écrits dans {out.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()
