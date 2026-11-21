"""Extraction multimodale d'un PDF : texte, tableaux et images.

- Le texte est lu avec le lecteur PyMuPDF de LlamaIndex (un ``Document`` par page),
  puis découpé en passages avec ``SentenceSplitter``.
- Les tableaux sont détectés avec ``page.find_tables()`` de PyMuPDF et convertis
  en Markdown pour rester lisibles par le LLM.
- Les images matricielles sont extraites et sauvegardées sur disque ; leur légende
  (« Figure N: ... ») est retrouvée sur la page pour enrichir les métadonnées.

Chaque élément conserve des métadonnées vérifiables : fichier source, page,
type de contenu, identifiant, légende.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.readers.file import PyMuPDFReader

from .config import settings

CAPTION_RE = re.compile(r"^\s*(Figure|Fig\.|Table)\s*(\d+)\s*[:.]", re.IGNORECASE)
MIN_IMAGE_SIDE = 64  # ignore les icônes et artefacts trop petits


@dataclass
class ExtractedImage:
    image_id: str
    path: Path
    page: int
    caption: str = ""
    width: int = 0
    height: int = 0


@dataclass
class ExtractionResult:
    source: str
    text_nodes: list[TextNode] = field(default_factory=list)
    table_nodes: list[TextNode] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)


def document_id(pdf_path: Path) -> str:
    """Identifiant stable du document (hash du contenu)."""
    return hashlib.sha1(pdf_path.read_bytes()).hexdigest()[:12]


def _captions(page: pymupdf.Page, kind: str) -> list[tuple[pymupdf.Rect, str]]:
    """Retourne les blocs de légende (« Figure N: » ou « Table N: ») de la page."""
    result = []
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        match = CAPTION_RE.match(text)
        if match and match.group(1).lower().startswith(kind):
            result.append((pymupdf.Rect(x0, y0, x1, y1), " ".join(text.split())))
    return result


def _nearest_caption(bbox: pymupdf.Rect, captions: list[tuple[pymupdf.Rect, str]]) -> str:
    if not captions:
        return ""
    def distance(item):
        rect, _ = item
        return min(abs(rect.y0 - bbox.y1), abs(bbox.y0 - rect.y1))
    return min(captions, key=distance)[1]


def extract_text(pdf_path: Path, doc_id: str) -> list[TextNode]:
    """Texte par page (LlamaIndex) puis découpage en passages."""
    reader = PyMuPDFReader()
    pages: list[Document] = reader.load_data(file_path=pdf_path)
    for doc in pages:
        page_num = int(doc.metadata.get("source", 0))
        doc.metadata = {
            "source": pdf_path.name,
            "file_id": doc_id,
            "page": page_num,
            "content_type": "text",
        }
        doc.excluded_embed_metadata_keys = ["file_id", "content_type"]
        doc.excluded_llm_metadata_keys = ["file_id"]

    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    nodes = splitter.get_nodes_from_documents(pages)
    for i, node in enumerate(nodes):
        node.metadata["chunk_index"] = i
    return nodes


def extract_tables(pdf_path: Path, doc_id: str) -> list[TextNode]:
    """Tableaux détectés par PyMuPDF, convertis en Markdown."""
    nodes: list[TextNode] = []
    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            tables = page.find_tables().tables
            if not tables:
                continue
            captions = _captions(page, "table")
            for t_idx, table in enumerate(tables):
                markdown = table.to_markdown(clean=False).strip()
                if not markdown:
                    continue
                caption = _nearest_caption(pymupdf.Rect(table.bbox), captions)
                text = f"{caption}\n\n{markdown}" if caption else markdown
                nodes.append(
                    TextNode(
                        text=text,
                        metadata={
                            "source": pdf_path.name,
                            "file_id": doc_id,
                            "page": page.number + 1,
                            "content_type": "table",
                            "table_index": t_idx,
                            "caption": caption,
                        },
                        excluded_embed_metadata_keys=["file_id", "content_type", "table_index"],
                        excluded_llm_metadata_keys=["file_id", "table_index"],
                    )
                )
    return nodes


def extract_images(pdf_path: Path, doc_id: str) -> list[ExtractedImage]:
    """Images matricielles du PDF, enregistrées en PNG dans storage/images/."""
    images: list[ExtractedImage] = []
    out_dir = settings.images_dir / doc_id
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[int] = set()

    with pymupdf.open(pdf_path) as pdf:
        for page in pdf:
            captions = _captions(page, "fig")
            for info in page.get_image_info(xrefs=True):
                xref = info.get("xref", 0)
                if not xref or xref in seen:
                    continue
                seen.add(xref)
                pix = pymupdf.Pixmap(pdf, xref)
                if pix.width < MIN_IMAGE_SIDE or pix.height < MIN_IMAGE_SIDE:
                    continue
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
                image_id = f"{doc_id}_p{page.number + 1}_x{xref}"
                path = out_dir / f"{image_id}.png"
                pix.save(path)
                images.append(
                    ExtractedImage(
                        image_id=image_id,
                        path=path,
                        page=page.number + 1,
                        caption=_nearest_caption(pymupdf.Rect(info["bbox"]), captions),
                        width=pix.width,
                        height=pix.height,
                    )
                )
    return images


def extract_pdf(pdf_path: str | Path) -> ExtractionResult:
    pdf_path = Path(pdf_path)
    doc_id = document_id(pdf_path)
    return ExtractionResult(
        source=pdf_path.name,
        text_nodes=extract_text(pdf_path, doc_id),
        table_nodes=extract_tables(pdf_path, doc_id),
        images=extract_images(pdf_path, doc_id),
    )
