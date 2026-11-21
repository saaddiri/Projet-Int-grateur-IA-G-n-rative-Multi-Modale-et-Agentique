"""Reranking des passages par un cross-encoder multilingue.

Trois configurations comparées (livrable L2.5) :
- ``none``    : ordre de la recherche vectorielle (pas de reranker) ;
- ``base``    : reranker initial ``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` ;
- ``adapted`` : le même modèle adapté au corpus (``models/reranker-maintidoc-v1``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from .config import ROOT_DIR, settings

BASE_RERANKER = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
ADAPTED_RERANKER = ROOT_DIR / "models" / "reranker-maintidoc-v1"


def resolve(name: str) -> str | None:
    """Traduit un nom de configuration en chemin / identifiant de modèle."""
    if name in ("", "none"):
        return None
    if name == "base":
        return BASE_RERANKER
    if name == "adapted":
        return str(ADAPTED_RERANKER)
    return name


@lru_cache(maxsize=4)
def load(model: str):
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model, max_length=384, device="cpu")


def default_reranker() -> str | None:
    """Reranker adapté s'il existe, sinon le reranker initial (surchargeable via RERANKER)."""
    if settings.reranker:
        return resolve(settings.reranker)
    return str(ADAPTED_RERANKER) if Path(ADAPTED_RERANKER).exists() else BASE_RERANKER


def rerank(query: str, items: list, text_of=lambda x: x.text, model: str | None = "default",
           top_k: int | None = None):
    """Réordonne ``items`` selon le score du cross-encoder ; renvoie (item, score).

    ``model="default"`` utilise :func:`default_reranker` ; ``model=None`` désactive
    le reranking (ordre d'entrée conservé).
    """
    model = default_reranker() if model == "default" else model
    if not items:
        return []
    if model is None:
        ranked = [(item, float(len(items) - i)) for i, item in enumerate(items)]
    else:
        scores = load(model).predict([(query, text_of(item)) for item in items], batch_size=16)
        ranked = sorted(zip(items, map(float, scores)), key=lambda pair: pair[1], reverse=True)
    return ranked[:top_k] if top_k else ranked
