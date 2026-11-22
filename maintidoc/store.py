"""Accès à PGVector (PostgreSQL) via LlamaIndex.

Deux tables vectorielles :
- ``maintidoc_text``   : passages de texte, tableaux et descriptions d'images
                         (embeddings texte multilingues, dimension 384 par défaut) ;
- ``maintidoc_images`` : images encodées par CLIP (dimension 512), pour la
                         recherche texte -> image.
"""

from __future__ import annotations

from functools import lru_cache

from llama_index.core.vector_stores.types import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.postgres import PGVectorStore
from sqlalchemy import create_engine, text

from .config import settings


def _store(table_name: str, embed_dim: int) -> PGVectorStore:
    return PGVectorStore.from_params(
        host=settings.pg_host,
        port=str(settings.pg_port),
        user=settings.pg_user,
        password=settings.pg_password,
        database=settings.pg_database,
        table_name=table_name,
        embed_dim=embed_dim,
        perform_setup=True,
        # Connexions vérifiées avant usage : après un redémarrage de PostgreSQL, les
        # connexions mortes du pool sont remplacées au lieu de provoquer une erreur 500.
        create_engine_kwargs={"pool_pre_ping": True, "pool_recycle": 1800},
    )


@lru_cache(maxsize=1)
def get_text_embedder() -> HuggingFaceEmbedding:
    kwargs = {}
    if "e5" in settings.text_embed_model.lower():
        # Les modèles E5 attendent des préfixes distincts pour requêtes et passages.
        kwargs = {"query_instruction": "query: ", "text_instruction": "passage: "}
    return HuggingFaceEmbedding(model_name=settings.text_embed_model, device="cpu", **kwargs)


@lru_cache(maxsize=1)
def text_store() -> PGVectorStore:
    dim = len(get_text_embedder().get_text_embedding("dimension"))
    return _store(settings.text_table, dim)


@lru_cache(maxsize=1)
def image_store() -> PGVectorStore:
    from .images import get_clip

    return _store(settings.image_table, get_clip().dim)


def file_filter(file_id: str) -> MetadataFilters:
    return MetadataFilters(filters=[MetadataFilter(key="file_id", value=file_id)])


def type_filter(content_types: list[str]) -> MetadataFilters:
    return MetadataFilters(
        filters=[MetadataFilter(key="content_type", value=content_types, operator=FilterOperator.IN)]
    )


def _engine():
    url = (
        f"postgresql+psycopg2://{settings.pg_user}:{settings.pg_password}"
        f"@{settings.pg_host}:{settings.pg_port}/{settings.pg_database}"
    )
    return create_engine(url, pool_pre_ping=True)


def list_documents() -> list[dict]:
    """Documents indexés, avec le nombre d'éléments par type de contenu."""
    table = f"data_{settings.text_table}"
    query = text(
        f"""
        SELECT metadata_->>'source' AS source, metadata_->>'file_id' AS file_id,
               metadata_->>'content_type' AS content_type, COUNT(*) AS n
        FROM {table}
        GROUP BY 1, 2, 3 ORDER BY 1
        """
    )
    docs: dict[str, dict] = {}
    try:
        with _engine().connect() as conn:
            for source, file_id, ctype, n in conn.execute(query):
                entry = docs.setdefault(file_id, {"source": source, "file_id": file_id})
                entry[ctype] = n
    except Exception:  # table pas encore créée
        return []
    return list(docs.values())
