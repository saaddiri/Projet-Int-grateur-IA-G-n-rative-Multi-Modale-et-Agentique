"""Tests de l'étape 2 sans base de données ni modèle (fusion, métriques, schémas)."""

import math

import pytest

from maintidoc.multimodal import LLMAnswer, MaintiDocAnswer, reciprocal_rank_fusion
from maintidoc.retrieval import Hit
from maintidoc.reranker import rerank
from maintidoc.vision import PhotoAnalysis
from scripts.eval_reranker import mrr_at_k, ndcg_at_k


def _hit(node_id, image_path=""):
    return Hit(text=node_id, score=0.0, source="doc.pdf", page=1, content_type="text",
               node_id=node_id, image_path=image_path)


def test_rrf_favorise_les_elements_presents_dans_plusieurs_listes():
    fused = reciprocal_rank_fusion({
        "a": [_hit("x"), _hit("y"), _hit("z")],
        "b": [_hit("y"), _hit("w")],
    })
    assert [h.node_id for h in fused][:2] == ["y", "x"]


def test_rrf_fusionne_une_image_vue_par_clip_et_par_sa_description():
    fused = reciprocal_rank_fusion({
        "clip": [_hit("clip_img1", image_path="img1.png")],
        "texte": [_hit("desc_img1", image_path="img1.png"), _hit("t1")],
    })
    assert len(fused) == 2
    assert fused[0].node_id == "desc_img1"  # la version description est conservée


def test_metriques():
    assert ndcg_at_k(1) == 1.0
    assert ndcg_at_k(3) == pytest.approx(1 / math.log2(4))
    assert ndcg_at_k(6) == 0.0 and ndcg_at_k(None) == 0.0
    assert mrr_at_k(2) == 0.5 and mrr_at_k(11) == 0.0


def test_sans_reranker_conserve_l_ordre():
    ranked = rerank("q", ["a", "b", "c"], text_of=str, model=None)
    assert [item for item, _ in ranked] == ["a", "b", "c"]


def test_schemas_pydantic():
    raw = LLMAnswer.model_validate_json(
        '{"answer": "ok", "cited_passages": [1], "uncertainty": "moyenne", '
        '"uncertainty_reason": "r", "needs_clarification": false}'
    )
    assert raw.clarification_question == ""
    with pytest.raises(Exception):
        LLMAnswer.model_validate({"answer": "x", "uncertainty": "inconnue", "uncertainty_reason": "",
                                  "needs_clarification": False})
    analysis = PhotoAnalysis(description="d", search_query="q", confidence="faible")
    result = MaintiDocAnswer(answer="a", citations=[], uncertainty="élevée", uncertainty_reason="r",
                             needs_clarification=True, clarification_question="?", photo_analysis=analysis)
    assert result.model_dump()["photo_analysis"]["confidence"] == "faible"
