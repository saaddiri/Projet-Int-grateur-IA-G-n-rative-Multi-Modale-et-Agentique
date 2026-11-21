"""Pose une question en ligne de commande (agent ou RAG simple).

Usage :  python -m scripts.ask "Quelle est la valeur de d_model ?" [--rag]
"""

import argparse

from maintidoc.agent import MaintiDocAgent
from maintidoc.rag import answer_with_rag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", nargs="+")
    parser.add_argument("--rag", action="store_true", help="RAG simple au lieu de l'agent")
    args = parser.parse_args()

    agent = None if args.rag else MaintiDocAgent()
    for question in args.questions:
        answer = answer_with_rag(question) if args.rag else agent.ask(question)
        print(f"\n### Q : {question}")
        if answer.tool_calls:
            print("Outils :", "; ".join(answer.tool_calls))
        print(answer.text)
        print("Sources :")
        for i, hit in enumerate(answer.sources, start=1):
            print(f"  [{i}] {hit.reference} (score {hit.score:.3f})")


if __name__ == "__main__":
    main()
