"""Tests d'extraction (sans base de données ni modèles)."""

from pathlib import Path

from maintidoc.extraction import extract_pdf

PDF = Path(__file__).resolve().parent.parent / "data" / "attention_is_all_you_need.pdf"


def test_extraction_multimodale():
    result = extract_pdf(PDF)

    assert len(result.text_nodes) > 10
    assert all(1 <= n.metadata["page"] <= 11 for n in result.text_nodes)

    captions = {n.metadata["caption"].split(":")[0] for n in result.table_nodes}
    assert {"Table 1", "Table 2", "Table 3"} <= captions

    assert len(result.images) >= 3
    assert any(img.caption.startswith("Figure 1") for img in result.images)
    assert all(img.path.exists() for img in result.images)
