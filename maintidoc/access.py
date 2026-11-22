"""Filtrage des documents selon les autorisations (contrainte transversale).

Chaque document indexé est rattaché à une règle d'accès (data/documents_acl.json, complété
par storage/documents_acl.local.json pour les documents ajoutés depuis l'interface) :
rôles autorisés et sites concernés. Un document absent du fichier suit la règle par
défaut (superviseurs uniquement) : un nouveau document n'est jamais visible de tous
par accident.

Le backend place la liste des documents autorisés pour l'utilisateur dans un
ContextVar (`restricted_to`) ; la recherche vectorielle (texte, tableaux, figures,
CLIP) ajoute alors un filtre de métadonnées `source IN (...)` à chaque requête
PGVector. Un document non autorisé n'est donc jamais récupéré, ni montré au LLM,
ni cité. Sans restriction (scripts locaux, évaluation hors ligne), tout est visible.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Iterator

from .config import ROOT_DIR, settings

ACL_FILE = ROOT_DIR / "data" / "documents_acl.json"  # règles versionnées (documents de démonstration)


def local_acl_file() -> Path:
    """Règles ajoutées depuis l'interface (volume persistant storage/), prioritaires."""
    return settings.storage_dir / "documents_acl.local.json"


_allowed: ContextVar[tuple[str, ...] | None] = ContextVar("maintidoc_allowed_sources", default=None)


def _read(path: Path) -> dict:
    if not path.exists():
        return {"default": {"roles": ["superviseur"], "sites": ["*"]}, "documents": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def load_acl(path: Path | None = None) -> dict:
    if path is not None:
        return _read(Path(path))
    acl = _read(ACL_FILE)
    local = local_acl_file()
    if local.exists():
        acl.setdefault("documents", {}).update(_read(local).get("documents", {}))
    return acl


def save_rule(source: str, roles: list[str], sites: list[str], path: Path | None = None) -> None:
    path = Path(path or local_acl_file())
    path.parent.mkdir(parents=True, exist_ok=True)
    acl = _read(path)
    acl.setdefault("documents", {})[source] = {"roles": sorted(set(roles)), "sites": sorted(set(sites))}
    path.write_text(json.dumps(acl, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def can_read(source: str, role: str, sites: tuple[str, ...] | list[str], acl: dict | None = None) -> bool:
    acl = acl or load_acl()
    rule = acl.get("documents", {}).get(source, acl.get("default", {"roles": ["superviseur"], "sites": ["*"]}))
    if role not in rule.get("roles", []):
        return False
    allowed_sites = rule.get("sites", ["*"])
    return "*" in allowed_sites or bool(set(allowed_sites) & set(sites))


def allowed_sources(role: str, sites: tuple[str, ...] | list[str], sources: list[str] | None = None) -> list[str]:
    """Documents indexés que ce rôle / ces sites peuvent consulter."""
    if sources is None:
        from .store import list_documents

        sources = [doc["source"] for doc in list_documents()]
    acl = load_acl()
    return [s for s in sources if can_read(s, role, sites, acl)]


@contextmanager
def restricted_to(sources: list[str]) -> Iterator[None]:
    token = _allowed.set(tuple(sources))
    try:
        yield
    finally:
        _allowed.reset(token)


def current_restriction() -> tuple[str, ...] | None:
    """None = pas de restriction ; tuple (éventuellement vide) = documents autorisés."""
    return _allowed.get()
