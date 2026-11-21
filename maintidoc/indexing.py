"""Pipeline d'indexation : PDF -> extraction -> embeddings -> PGVector."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from llama_index.core.schema import TextNode

from .config import settings
from .extraction import ExtractedImage, extract_pdf
from .images import describe_image, get_clip
from .store import file_filter, get_text_embedder, image_store, text_store

log = logging.getLogger(__name__)

Progress = Callable[[str], None]


def _image_description_node(image: ExtractedImage, description: str, doc_id: str, source: str) -> TextNode:
    body = f"[Image page {image.page}] {image.caption}\n{description}".strip()
    return TextNode(
        id_=f"desc_{image.image_id}",
        text=body,
        metadata={
            "source": source,
            "file_id": doc_id,
            "page": image.page,
            "content_type": "image_description",
            "image_id": image.image_id,
            "image_path": str(image.path),
            "caption": image.caption,
        },
        excluded_embed_metadata_keys=["file_id", "content_type", "image_id", "image_path"],
        excluded_llm_metadata_keys=["file_id", "image_path"],
    )


def _embed_text_nodes(nodes: list[TextNode], progress: Progress) -> None:
    embedder = get_text_embedder()
    batch = 16
    for start in range(0, len(nodes), batch):
        chunk = nodes[start:start + batch]
        texts = [n.get_content(metadata_mode="embed") for n in chunk]
        for node, emb in zip(chunk, embedder.get_text_embedding_batch(texts)):
            node.embedding = emb
        progress(f"Embeddings texte : {min(start + batch, len(nodes))}/{len(nodes)}")


def index_pdf(pdf_path: str | Path, progress: Progress | None = None) -> dict:
    """Indexe un PDF (réindexation propre si le document existe déjà)."""
    progress = progress or (lambda msg: log.info(msg))
    pdf_path = Path(pdf_path)

    progress(f"Extraction de {pdf_path.name} (texte, tableaux, images)…")
    result = extract_pdf(pdf_path)
    doc_id = result.text_nodes[0].metadata["file_id"] if result.text_nodes else None
    if doc_id is None:
        raise ValueError("Aucun texte extrait : le PDF est peut-être scanné (OCR non géré à cette étape).")
    progress(
        f"{len(result.text_nodes)} passages, {len(result.table_nodes)} tableaux, "
        f"{len(result.images)} images extraits."
    )

    # Réindexation idempotente : on supprime d'abord les anciens vecteurs du document.
    t_store, i_store = text_store(), image_store()
    t_store.delete_nodes(filters=file_filter(doc_id))
    i_store.delete_nodes(filters=file_filter(doc_id))

    # Images : CLIP + description VLM
    description_nodes: list[TextNode] = []
    image_nodes: list[TextNode] = []
    if result.images:
        progress("Encodage CLIP des images…")
        clip_vectors = get_clip().encode_images([img.path for img in result.images])
        for n, (image, vector) in enumerate(zip(result.images, clip_vectors), start=1):
            description = ""
            # Cache sur disque : une réindexation ne relance pas le VLM (coûteux sur CPU).
            cache = image.path.with_suffix(".txt")
            if cache.exists():
                description = cache.read_text(encoding="utf-8").strip()
                progress(f"Description de l'image {n}/{len(result.images)} lue depuis le cache.")
            elif settings.describe_images:
                progress(f"Description de l'image {n}/{len(result.images)} par le VLM ({settings.vlm_model})…")
                try:
                    description = describe_image(image.path, image.caption)
                    cache.write_text(description, encoding="utf-8")
                except Exception as exc:  # le VLM est optionnel : on garde la légende
                    log.warning("Description VLM impossible pour %s : %s", image.image_id, exc)
            if not description:
                description = image.caption or "Image sans légende."
            desc_node = _image_description_node(image, description, doc_id, result.source)
            description_nodes.append(desc_node)

            img_node = TextNode(
                id_=f"clip_{image.image_id}",
                text=desc_node.text,
                metadata={**desc_node.metadata, "content_type": "image"},
            )
            img_node.embedding = vector
            image_nodes.append(img_node)

    all_text_nodes = result.text_nodes + result.table_nodes + description_nodes
    _embed_text_nodes(all_text_nodes, progress)

    progress("Écriture dans PGVector…")
    t_store.add(all_text_nodes)
    if image_nodes:
        i_store.add(image_nodes)

    summary = {
        "file_id": doc_id,
        "source": result.source,
        "text_chunks": len(result.text_nodes),
        "tables": len(result.table_nodes),
        "images": len(result.images),
    }
    progress(f"Indexation terminée : {summary}")
    return summary
