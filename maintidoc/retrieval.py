"""Recherche vectorielle dans PGVector (texte, tableaux, images)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
    VectorStoreQuery,
)

from .images import get_clip
from .store import get_text_embedder, image_store, text_store, type_filter


@dataclass
class Hit:
    text: str
    score: float
    source: str
    page: int
    content_type: str
    caption: str = ""
    image_path: str = ""
    node_id: str = ""

    @property
    def reference(self) -> str:
        label = {"text": "texte", "table": "tableau", "image_description": "image", "image": "image"}
        return f"{self.source}, p. {self.page} ({label.get(self.content_type, self.content_type)})"


def _to_hits(result) -> list[Hit]:
    hits = []
    for node, score in zip(result.nodes or [], result.similarities or []):
        meta = node.metadata
        hits.append(
            Hit(
                text=node.get_content(),
                score=float(score),
                source=meta.get("source", "?"),
                page=int(meta.get("page", 0)),
                content_type=meta.get("content_type", "text"),
                caption=meta.get("caption", ""),
                image_path=meta.get("image_path", ""),
                node_id=node.node_id,
            )
        )
    return hits


# « figure 3 », « fig. 2 », « table 1 », « tableau 2 »...
REF_RE = re.compile(r"\b(fig(?:ure)?\.?|tab(?:le|leau)?)\s*(\d+)", re.IGNORECASE)


def _explicit_captions(query: str, content_types: list[str]) -> list[str]:
    """Légendes citées explicitement dans la question (ex. « Figure 1: »)."""
    captions = []
    for kind, number in REF_RE.findall(query):
        if kind.lower().startswith("fig") and "image_description" in content_types:
            captions.append(f"Figure {number}:")
        elif kind.lower().startswith("tab") and "table" in content_types:
            captions.append(f"Table {number}:")
    return captions


def _query(embedding: list[float], filters: MetadataFilters, k: int) -> list[Hit]:
    result = text_store().query(
        VectorStoreQuery(query_embedding=embedding, similarity_top_k=k, filters=filters)
    )
    return _to_hits(result)


def _dedupe(hits: list[Hit]) -> list[Hit]:
    seen, unique = set(), []
    for hit in hits:
        key = hit.node_id or hit.text
        if key not in seen:
            seen.add(key)
            unique.append(hit)
    return unique


def search(query: str, content_types: list[str], k: int = 4) -> list[Hit]:
    """Recherche sémantique (embeddings texte) restreinte à certains types de contenu.

    Si la question cite explicitement une figure ou un tableau par son numéro,
    l'élément correspondant (retrouvé par sa légende) est placé en tête : la
    similarité sémantique seule capture mal ce type de référence.
    """
    embedding = get_text_embedder().get_query_embedding(query)
    explicit: list[Hit] = []
    for caption in _explicit_captions(query, content_types):
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="content_type", value=content_types, operator=FilterOperator.IN),
                MetadataFilter(key="caption", value=caption, operator=FilterOperator.TEXT_MATCH_INSENSITIVE),
            ]
        )
        explicit += _query(embedding, filters, 2)
    hits = _dedupe(explicit + _query(embedding, type_filter(content_types), k))
    return hits[:max(k, len(explicit))]


def search_text(query: str, k: int = 4) -> list[Hit]:
    return search(query, ["text"], k)


def search_tables(query: str, k: int = 3) -> list[Hit]:
    return search(query, ["table"], k)


def search_images(query: str, k: int = 2) -> list[Hit]:
    """Recherche d'images : fusion CLIP (texte -> image) et descriptions VLM.

    CLIP comprend surtout l'anglais court ; les descriptions VLM, indexées avec
    les embeddings texte multilingues, couvrent les questions longues ou en
    français. On garde la première occurrence de chaque image.
    """
    clip_res = image_store().query(
        VectorStoreQuery(query_embedding=get_clip().encode_text(query), similarity_top_k=k)
    )
    descriptions = search(query, ["image_description"], k)
    explicit = descriptions if _explicit_captions(query, ["image_description"]) else []
    merged: dict[str, Hit] = {}
    for hit in explicit + _to_hits(clip_res) + descriptions:
        key = hit.image_path or hit.text
        if key not in merged:
            merged[key] = hit
    return list(merged.values())[:k]


def search_all(query: str, k: int = 6) -> list[Hit]:
    return search(query, ["text", "table", "image_description"], k)


def format_hits(hits: list[Hit]) -> str:
    """Contexte formaté pour le LLM, chaque passage étant numéroté et sourcé."""
    if not hits:
        return "Aucun passage pertinent trouvé."
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] Source : {hit.reference}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)
