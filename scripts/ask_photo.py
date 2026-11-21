"""Pipeline V2 en ligne de commande : question + photo -> réponse structurée (JSON).

Usage :  python -m scripts.ask_photo "Que représente ce schéma ?" --photo data/photos_test/schema_photo.jpg
"""

import argparse
import json
import time

from PIL import Image

from maintidoc.multimodal import answer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", nargs="?", default="")
    parser.add_argument("--photo", help="Chemin d'une photo (optionnel)")
    parser.add_argument("--reranker", default="default", help="default | none | base | adapted")
    args = parser.parse_args()

    from maintidoc.reranker import resolve

    reranker = "default" if args.reranker == "default" else resolve(args.reranker)
    photo = Image.open(args.photo) if args.photo else None
    start = time.time()
    result, trace = answer(args.question, photo, reranker=reranker)
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))
    print(f"\nRequêtes : {trace.queries}")
    print(f"Listes fusionnées : {trace.lists} ; reranker : {trace.reranker}")
    for hit, score in trace.ranked:
        print(f"  {hit.reference} — score {score:.3f}")
    print(f"Durée : {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
