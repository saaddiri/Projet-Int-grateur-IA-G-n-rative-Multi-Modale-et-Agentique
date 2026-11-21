"""Construit le jeu de données d'adaptation du reranker (livrable L2.3).

Étapes :
1. Génération : pour chaque passage indexé (texte, tableau, description de figure),
   le LLM local génère des questions en français et en anglais auxquelles seul
   ce passage répond.
2. Vérification : un second appel au LLM (juge) confirme que le passage contient
   bien la réponse ; les questions rejetées et les doublons sont écartés.
3. Négatifs difficiles : pour chaque question, les passages les mieux classés par
   la recherche vectorielle, hors passage positif, servent d'exemples négatifs.
4. Découpage train / test par question (stratifié par passage, graine fixe).

La génération est reprise là où elle s'est arrêtée (cache questions_raw.jsonl).

Usage :  python -m scripts.build_dataset [--per-lang 6] [--negatives 3]
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import ollama
from sqlalchemy import text

from maintidoc.config import ROOT_DIR, settings
from maintidoc.store import _engine, get_text_embedder
from maintidoc.retrieval import search

OUT_DIR = ROOT_DIR / "data" / "reranker"
RAW_FILE = OUT_DIR / "questions_raw.jsonl"

GEN_PROMPT = """Voici un passage extrait d'un document technique ({ctype}, page {page}).

PASSAGE :
\"\"\"{passage}\"\"\"

Rédige {n} questions en français ET {n} questions en anglais auxquelles ce passage
répond précisément. Contraintes :
- chaque question doit être compréhensible seule (pas de « ce passage », « ci-dessus ») ;
- varie les formulations (définition, valeur chiffrée, comparaison, rôle d'un composant) ;
- n'invente aucune information absente du passage.
Réponds en JSON : {{"questions_fr": [...], "questions_en": [...]}}"""

JUDGE_PROMPT = """PASSAGE :
\"\"\"{passage}\"\"\"

QUESTIONS :
{questions}

Pour chaque question numérotée, le passage contient-il l'information nécessaire
pour y répondre ? Sois strict : réponds false si la réponse n'est pas dans le passage.
Réponds en JSON : {{"answerable": [true, false, ...]}} (une valeur par question, dans l'ordre)."""

GEN_SCHEMA = {
    "type": "object",
    "properties": {
        "questions_fr": {"type": "array", "items": {"type": "string"}},
        "questions_en": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["questions_fr", "questions_en"],
}
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {"answerable": {"type": "array", "items": {"type": "boolean"}}},
    "required": ["answerable"],
}


def _client() -> ollama.Client:
    return ollama.Client(host=settings.ollama_base_url)


def _chat_json(prompt: str, schema: dict, num_predict: int) -> dict:
    response = _client().chat(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        format=schema,
        options={"temperature": 0.4, "num_predict": num_predict, "num_ctx": 4096},
    )
    return json.loads(response["message"]["content"])


def load_passages() -> list[dict]:
    query = text(
        "SELECT node_id, text, metadata_ FROM data_" + settings.text_table + " ORDER BY id"
    )
    with _engine().connect() as conn:
        rows = conn.execute(query).all()
    return [
        {
            "node_id": node_id,
            "text": body,
            "page": meta.get("page"),
            "content_type": meta.get("content_type"),
            "source": meta.get("source"),
        }
        for node_id, body, meta in rows
    ]


def _normalize(question: str) -> str:
    return re.sub(r"\W+", " ", question.lower()).strip()


def generate(passages: list[dict], per_lang: int) -> list[dict]:
    """Génère et vérifie les questions (reprise possible)."""
    done: dict[str, list[dict]] = {}
    if RAW_FILE.exists():
        for line in RAW_FILE.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            done.setdefault(item["node_id"], []).append(item)

    with RAW_FILE.open("a", encoding="utf-8") as out:
        for i, passage in enumerate(passages, start=1):
            if passage["node_id"] in done:
                continue
            body = passage["text"][:2500]
            try:
                generated = _chat_json(
                    GEN_PROMPT.format(ctype=passage["content_type"], page=passage["page"],
                                      passage=body, n=per_lang),
                    GEN_SCHEMA, num_predict=900,
                )
            except Exception as exc:
                print(f"[{i}/{len(passages)}] génération impossible : {exc}")
                continue
            candidates = [(q, "fr") for q in generated.get("questions_fr", [])[:per_lang]]
            candidates += [(q, "en") for q in generated.get("questions_en", [])[:per_lang]]
            candidates = [(q.strip(), lang) for q, lang in candidates if len(q.strip()) >= 12]
            # Vérification groupée : un seul appel au juge pour toutes les questions du passage.
            try:
                numbered = "\n".join(f"{n}. {q}" for n, (q, _) in enumerate(candidates, start=1))
                verdicts = _chat_json(JUDGE_PROMPT.format(passage=body, questions=numbered),
                                      JUDGE_SCHEMA, num_predict=120).get("answerable", [])
            except Exception:
                verdicts = []
            items = []
            for n, (question, lang) in enumerate(candidates):
                verified = bool(verdicts[n]) if n < len(verdicts) else False
                items.append({"node_id": passage["node_id"], "question": question, "lang": lang,
                              "verified": verified, "page": passage["page"],
                              "content_type": passage["content_type"]})
            # Le marqueur garantit la reprise même si aucune question n'est retenue.
            items = items or [{"node_id": passage["node_id"], "question": "", "lang": "", "verified": False}]
            for item in items:
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            out.flush()
            done[passage["node_id"]] = items
            kept = sum(it["verified"] for it in items)
            print(f"[{i}/{len(passages)}] p.{passage['page']} {passage['content_type']} : "
                  f"{kept}/{len(candidates)} questions vérifiées", flush=True)

    return [it for items in done.values() for it in items]


def build(per_lang: int, negatives: int, test_ratio: float, seed: int) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    passages = load_passages()
    by_id = {p["node_id"]: p for p in passages}
    print(f"{len(passages)} passages indexés.")

    raw = generate(passages, per_lang)
    seen: set[str] = set()
    questions = []
    for item in raw:
        key = _normalize(item["question"])
        if item["verified"] and key and key not in seen and item["node_id"] in by_id:
            seen.add(key)
            questions.append(item)
    rejected = sum(1 for it in raw if it["question"] and not it["verified"])
    print(f"{len(questions)} questions retenues, {rejected} rejetées par le juge.")

    # Découpage train / test stratifié par passage.
    rng = random.Random(seed)
    groups: dict[str, list[dict]] = {}
    for q in questions:
        groups.setdefault(q["node_id"], []).append(q)
    for items in groups.values():
        rng.shuffle(items)
        n_test = max(1, round(len(items) * test_ratio)) if len(items) > 2 else 0
        for j, q in enumerate(items):
            q["split"] = "test" if j < n_test else "train"

    get_text_embedder()  # charge le modèle une fois
    pairs = {"train": [], "test": []}
    for n, q in enumerate(questions, start=1):
        hits = search(q["question"], ["text", "table", "image_description"], k=20)
        candidates = [h.node_id for h in hits]
        q["candidates"] = candidates  # liste de premier niveau, utilisée pour l'évaluation
        positive = by_id[q["node_id"]]["text"]
        pairs[q["split"]].append({"query": q["question"], "passage": positive, "label": 1.0})
        hard = [c for c in candidates if c != q["node_id"]][:negatives]
        for neg in hard:
            pairs[q["split"]].append({"query": q["question"], "passage": by_id[neg]["text"], "label": 0.0})
        if n % 50 == 0:
            print(f"Négatifs difficiles : {n}/{len(questions)}", flush=True)

    with (OUT_DIR / "questions.jsonl").open("w", encoding="utf-8") as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    for split, rows in pairs.items():
        with (OUT_DIR / f"pairs_{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (OUT_DIR / "passages.jsonl").open("w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = {
        "passages": len(passages),
        "questions_generated": sum(1 for it in raw if it["question"]),
        "questions_rejected_by_judge": rejected,
        "questions_kept": len(questions),
        "questions_train": sum(q["split"] == "train" for q in questions),
        "questions_test": sum(q["split"] == "test" for q in questions),
        "pairs_train": len(pairs["train"]),
        "pairs_test": len(pairs["test"]),
        "pairs_total": len(pairs["train"]) + len(pairs["test"]),
        "negatives_per_question": negatives,
        "generator": settings.llm_model,
        "seed": seed,
    }
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-lang", type=int, default=6, help="questions générées par langue et par passage")
    parser.add_argument("--negatives", type=int, default=3, help="négatifs difficiles par question")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.per_lang, args.negatives, args.test_ratio, args.seed)


if __name__ == "__main__":
    main()
