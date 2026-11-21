"""Indexe un ou plusieurs PDF dans PGVector.

Usage :  python -m scripts.ingest data/attention_is_all_you_need.pdf
"""

import argparse
import logging

from maintidoc.indexing import index_pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", help="Chemins des fichiers PDF à indexer")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for pdf in args.pdfs:
        index_pdf(pdf, progress=print)


if __name__ == "__main__":
    main()
