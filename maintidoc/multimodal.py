"""Pipeline RAG multimodal V2 (livrable L2.2).

question (+ photo)
  -> analyse de la photo par le VLM (PhotoAnalysis)
  -> recherches parallèles :
       * passages / tableaux / descriptions (embeddings texte) pour chaque requête ;
       * figures par CLIP multilingue (texte -> image) ;
       * figures par similarité CLIP image -> image avec la photo ;
  -> fusion des listes (Reciprocal Rank Fusion)
  -> reranking par cross-encoder (initial ou adapté)
  -> réponse structurée Pydantic : réponse, citations vérifiables, incertitude,
     demande de clarification si la situation est ambiguë.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.core.vector_stores.types import VectorStoreQuery
from PIL import Image
from pydantic import BaseModel, Field

from .images import encode_text_multilingual, get_clip
from .rag import get_llm
from .reranker import default_reranker, rerank
from .retrieval import Hit, _to_hits, search
from .store import image_store
from .vision import PhotoAnalysis, analyze_photo

RRF_K = 60
CANDIDATES = 20  # taille de la liste fusionnée envoyée au reranker
CONTEXT_SIZE = 5  # passages gardés après reranking
# Score minimal (logit du cross-encoder) du meilleur passage pour considérer que la
# documentation couvre la demande. Étalonné sur les 30 questions de test avec le
# reranker adapté : 29/30 ont un meilleur score >= -3.0 (voir docs/RAPPORT_RERANKER.md).
RELEVANCE_THRESHOLD = -3.0
AUTO_CITATIONS = 3


class LLMAnswer(BaseModel):
    """Format demandé au LLM (sortie structurée)."""

    answer: str = Field(description="Réponse en français, fondée uniquement sur les passages")
    cited_passages: list[int] = Field(default_factory=list, description="Numéros des passages utilisés")
    uncertainty: Literal["faible", "moyenne", "élevée"] = Field(description="Niveau d'incertitude")
    uncertainty_reason: str = Field(description="Pourquoi ce niveau d'incertitude")
    needs_clarification: bool = Field(description="Vrai si la demande est ambiguë ou si les passages ne suffisent pas")
    clarification_question: str = Field(default="", description="Question à poser au technicien si besoin")


class Citation(BaseModel):
    ref: int
    source: str
    page: int
    content_type: str
    excerpt: str
    image_path: str = ""
    attribution: Literal["modèle", "automatique"] = "modèle"


class MaintiDocAnswer(BaseModel):
    """Réponse finale renvoyée à l'interface (citations construites par le système)."""

    answer: str
    citations: list[Citation]
    uncertainty: Literal["faible", "moyenne", "élevée"]
    uncertainty_reason: str
    needs_clarification: bool
    clarification_question: str = ""
    relevance_score: float | None = None
    photo_analysis: PhotoAnalysis | None = None


@dataclass
class RetrievalTrace:
    queries: list[str] = field(default_factory=list)
    lists: dict[str, int] = field(default_factory=dict)
    reranker: str | None = None
    ranked: list[tuple[Hit, float]] = field(default_factory=list)


SYSTEM_PROMPT = """Tu es MaintiDoc, assistant documentaire pour techniciens de maintenance.
Réponds UNIQUEMENT à partir des passages numérotés fournis (et de l'analyse de la photo
si elle est présente). Indique dans cited_passages les numéros des passages utilisés.
Évalue ton incertitude honnêtement. Si la demande est ambiguë (photo difficile à
identifier, plusieurs interprétations possibles) ou si les passages ne contiennent pas
la réponse, mets needs_clarification à true et pose une question précise au technicien
au lieu d'inventer. Tu ne pilotes aucun équipement et ne garantis aucun diagnostic."""


def _key(hit: Hit) -> str:
    return hit.image_path or hit.node_id or hit.text[:80]


def _image_hits(vector: list[float], k: int) -> list[Hit]:
    hits = _to_hits(image_store().query(VectorStoreQuery(query_embedding=vector, similarity_top_k=k)))
    for hit in hits:
        hit.content_type = "image_description"
    return hits


def reciprocal_rank_fusion(lists: dict[str, list[Hit]], k: int = RRF_K) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for hits in lists.values():
        for rank, hit in enumerate(hits, start=1):
            key = _key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            # on conserve la version « description » (texte complet) de chaque image
            if key not in best or (hit.node_id or "").startswith("desc_"):
                best[key] = hit
    return [best[key] for key in sorted(scores, key=scores.get, reverse=True)]


def retrieve(question: str, photo: Image.Image | None = None, analysis: PhotoAnalysis | None = None,
             reranker: str | None | Literal["default"] = "default") -> RetrievalTrace:
    trace = RetrievalTrace()
    queries = [q for q in [question] if q.strip()]
    if analysis is not None:
        queries.append(analysis.search_query)
        if analysis.visible_text:
            queries.append(" ".join(analysis.visible_text[:10]))
    trace.queries = queries

    lists: dict[str, list[Hit]] = {}
    for i, query in enumerate(queries):
        lists[f"texte[{i}]"] = search(query, ["text", "table", "image_description"], k=10)
        lists[f"clip_multilingue[{i}]"] = _image_hits(encode_text_multilingual(query), k=3)
    if photo is not None:
        lists["clip_image"] = _image_hits(get_clip().encode_images([photo])[0], k=3)
    trace.lists = {name: len(hits) for name, hits in lists.items()}

    fused = reciprocal_rank_fusion(lists)[:CANDIDATES]
    model = default_reranker() if reranker == "default" else reranker
    trace.reranker = model
    rerank_query = " ".join(queries[:2])
    trace.ranked = rerank(rerank_query, fused, model=model, top_k=CONTEXT_SIZE)
    return trace


def _context(ranked: list[tuple[Hit, float]], analysis: PhotoAnalysis | None) -> str:
    blocks = []
    if analysis is not None:
        blocks.append(
            "ANALYSE DE LA PHOTO (VLM) :\n"
            f"- description : {analysis.description}\n"
            f"- textes visibles : {', '.join(analysis.visible_text) or 'aucun'}\n"
            f"- voyants / états : {', '.join(analysis.indicators) or 'aucun'}\n"
            f"- confiance : {analysis.confidence}"
        )
    for i, (hit, _) in enumerate(ranked, start=1):
        blocks.append(f"[{i}] Source : {hit.reference}\n{hit.text.strip()[:1500]}")
    return "\n\n---\n\n".join(blocks)


def answer(question: str, photo: Image.Image | None = None,
           reranker: str | None | Literal["default"] = "default") -> tuple[MaintiDocAnswer, RetrievalTrace]:
    analysis = analyze_photo(photo, question) if photo is not None else None
    trace = retrieve(question, photo, analysis, reranker)

    llm = get_llm().with_structured_output(LLMAnswer, method="json_schema")
    messages = [
        SystemMessage(SYSTEM_PROMPT),
        HumanMessage(f"{_context(trace.ranked, analysis)}\n\nQUESTION : {question or 'Que montre cette photo ?'}"),
    ]
    raw: LLMAnswer | None = None
    for _ in range(2):  # le petit LLM produit parfois un JSON invalide : un nouvel essai
        try:
            raw = llm.invoke(messages)
            break
        except Exception:  # sortie non conforme au schéma, erreur transitoire d'Ollama
            continue
    if raw is None:
        # Repli prudent : aucune réponse générée, donc aucune affirmation ni citation.
        raw = LLMAnswer(answer="La génération de la réponse a échoué ; consultez les passages proposés.",
                        uncertainty="élevée", uncertainty_reason="Sortie du modèle invalide après deux essais.",
                        needs_clarification=True,
                        clarification_question="Pouvez-vous reformuler votre question plus simplement ?")

    # Les citations sont construites par le système à partir des numéros : une
    # source inventée par le modèle est simplement ignorée.
    def make(ref: int, attribution: str) -> Citation:
        hit = trace.ranked[ref - 1][0]
        return Citation(ref=ref, source=hit.source, page=hit.page, content_type=hit.content_type,
                        excerpt=hit.text.strip()[:300], image_path=hit.image_path, attribution=attribution)

    refs = list(raw.cited_passages) + [int(n) for n in re.findall(r"\[(\d+)\]", raw.answer)]
    refs = [r for r in dict.fromkeys(refs) if 1 <= r <= len(trace.ranked)]
    citations = [make(r, "modèle") for r in refs]

    top_score = trace.ranked[0][1] if trace.ranked and trace.reranker else None
    covered = top_score is None or top_score >= RELEVANCE_THRESHOLD
    if not citations and covered:
        # Le petit LLM omet souvent les numéros : on rattache la réponse aux passages
        # jugés pertinents par le reranker, en le signalant (attribution automatique).
        relevant = [i for i, (_, score) in enumerate(trace.ranked, start=1)
                    if trace.reranker is None or score >= RELEVANCE_THRESHOLD]
        citations = [make(r, "automatique") for r in relevant[:AUTO_CITATIONS]]

    text, uncertainty, reason = raw.answer, raw.uncertainty, raw.uncertainty_reason
    needs_clarification, clarification = raw.needs_clarification, raw.clarification_question
    if not covered:
        # Aucun passage suffisamment pertinent : pas de réponse spéculative.
        text = ("Je n'ai pas trouvé dans la documentation indexée d'élément correspondant de façon "
                "fiable à cette demande.")
        citations, uncertainty = [], "élevée"
        reason = f"Meilleur score de pertinence {top_score:.2f} < seuil {RELEVANCE_THRESHOLD:.2f}."
        needs_clarification = True
        clarification = (
            "Pouvez-vous préciser l'équipement ou le schéma concerné (référence, emplacement), "
            "ou envoyer une photo plus nette et plus proche ?"
        )
    elif not citations:
        uncertainty, reason = "élevée", "Aucun passage cité à l'appui de la réponse."
    elif analysis is not None and analysis.confidence == "faible" and uncertainty == "faible":
        uncertainty, reason = "moyenne", "Le VLM signale une identification incertaine de la photo."

    result = MaintiDocAnswer(
        answer=text,
        citations=citations,
        uncertainty=uncertainty,
        uncertainty_reason=reason,
        needs_clarification=needs_clarification,
        clarification_question=(clarification or "Pouvez-vous préciser votre demande ?") if needs_clarification else "",
        relevance_score=top_score,
        photo_analysis=analysis,
    )
    return result, trace
