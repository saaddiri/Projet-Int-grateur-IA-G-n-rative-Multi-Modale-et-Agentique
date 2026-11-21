"""Versionnage des prompts et des modèles (étape 4).

`versions.lock.json` (racine du dépôt) fige l'empreinte de chaque prompt et l'identifiant
de chaque modèle. Un test vérifie que le code correspond au fichier : toute modification de
prompt ou de modèle doit être assumée en régénérant le fichier
(`python -m scripts.lock_versions`) et en incrémentant APP_VERSION.
"""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version

from .config import ROOT_DIR, settings

APP_VERSION = "4.0.0"
LOCK_FILE = ROOT_DIR / "versions.lock.json"


def prompts() -> dict[str, str]:
    from . import agent, images, multimodal, rag, vision, workflow

    return {
        "rag.RAG_SYSTEM_PROMPT": rag.RAG_SYSTEM_PROMPT,
        "agent.AGENT_SYSTEM_PROMPT": agent.AGENT_SYSTEM_PROMPT,
        "images.DESCRIBE_PROMPT": images.DESCRIBE_PROMPT,
        "vision.VISION_PROMPT": vision.VISION_PROMPT,
        "multimodal.SYSTEM_PROMPT": multimodal.SYSTEM_PROMPT,
        "workflow.DRAFT_PROMPT": workflow.DRAFT_PROMPT,
    }


def reranker_card() -> dict:
    path = ROOT_DIR / "models" / "reranker-maintidoc-v1.json"
    if not path.exists():
        return {}
    card = json.loads(path.read_text(encoding="utf-8"))
    return {k: card[k] for k in ("name", "version", "base_model", "train_sha256") if k in card}


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "absent"


def snapshot() -> dict:
    return {
        "app_version": APP_VERSION,
        "prompts": {name: hashlib.sha256(text.encode()).hexdigest()[:16] for name, text in prompts().items()},
        "models": {
            "llm": settings.llm_model,
            "vlm": settings.vlm_model,
            "text_embeddings": settings.text_embed_model,
            "clip": settings.clip_model,
            "clip_multilingual_text": settings.mclip_text_model,
            "reranker": reranker_card(),
        },
    }


def runtime_info() -> dict:
    """Informations exposées par /version (inclut les versions de paquets installées)."""
    return snapshot() | {"packages": {p: _pkg(p) for p in (
        "llama-index-core", "langchain", "langgraph", "mcp", "fastapi", "python-telegram-bot", "torch")}}


def write_lock() -> dict:
    data = snapshot()
    LOCK_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data


def check_lock() -> list[str]:
    """Différences entre le code et le fichier de verrouillage (liste vide = conforme)."""
    if not LOCK_FILE.exists():
        return ["versions.lock.json absent"]
    locked, current = json.loads(LOCK_FILE.read_text(encoding="utf-8")), snapshot()
    diffs = []
    for section in ("prompts", "models"):
        for key in sorted(set(locked[section]) | set(current[section])):
            if locked[section].get(key) != current[section].get(key):
                diffs.append(f"{section}.{key}")
    if locked["app_version"] != current["app_version"]:
        diffs.append("app_version")
    return diffs
