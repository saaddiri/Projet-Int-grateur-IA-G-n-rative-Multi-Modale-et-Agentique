"""Jetons d'approbation humaine (livrable L3.5).

Après validation explicite d'un brouillon par le technicien, l'application émet un
jeton signé (HMAC-SHA256) qui lie l'approbation :
  * à l'utilisateur qui approuve ;
  * à l'équipement ;
  * au contenu exact du ticket (empreinte SHA-256 du titre et de la description) ;
  * à un identifiant d'approbation unique (clé d'idempotence) et à une date d'expiration.

Le serveur MCP vérifie ce jeton avant toute création : un contenu modifié après
approbation, un autre utilisateur, un autre équipement, un jeton expiré ou falsifié
sont refusés. Le LLM ne voit jamais le secret et ne peut donc pas fabriquer de jeton.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
import uuid
from dataclasses import dataclass

from .config import settings

TOKEN_TTL_SECONDS = 15 * 60


class ApprovalError(ValueError):
    """Jeton d'approbation invalide."""


@dataclass(frozen=True)
class Approval:
    approval_id: str
    user_id: str
    equipment_id: str
    content_sha256: str
    issued_at: int
    expires_at: int


def content_hash(title: str, description: str) -> str:
    """Empreinte du contenu exact soumis à approbation."""
    return hashlib.sha256(f"{title.strip()}\x00{description.strip()}".encode()).hexdigest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _sign(payload: str, secret: str) -> str:
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


def issue_token(user_id: str, equipment_id: str, title: str, description: str,
                secret: str | None = None, ttl: int = TOKEN_TTL_SECONDS) -> str:
    """À appeler uniquement après un clic explicite d'approbation par l'utilisateur."""
    now = int(time.time())
    payload = _b64(json.dumps({
        "approval_id": uuid.uuid4().hex,
        "user_id": user_id,
        "equipment_id": equipment_id,
        "content_sha256": content_hash(title, description),
        "issued_at": now,
        "expires_at": now + ttl,
    }, separators=(",", ":")).encode())
    return f"{payload}.{_sign(payload, secret or settings.approval_secret)}"


def verify_token(token: str, *, user_id: str, equipment_id: str, title: str, description: str,
                 secret: str | None = None, now: float | None = None) -> Approval:
    try:
        payload, signature = token.rsplit(".", 1)
    except (AttributeError, ValueError):
        raise ApprovalError("jeton mal formé") from None
    if not hmac.compare_digest(signature, _sign(payload, secret or settings.approval_secret)):
        raise ApprovalError("signature du jeton invalide")
    try:
        approval = Approval(**json.loads(_unb64(payload)))
    except (ValueError, TypeError):
        raise ApprovalError("contenu du jeton illisible") from None
    if (now or time.time()) > approval.expires_at:
        raise ApprovalError("jeton expiré : nouvelle approbation nécessaire")
    if approval.user_id != user_id:
        raise ApprovalError("jeton émis pour un autre utilisateur")
    if approval.equipment_id != equipment_id:
        raise ApprovalError("jeton émis pour un autre équipement")
    if approval.content_sha256 != content_hash(title, description):
        raise ApprovalError("le contenu du ticket diffère du contenu approuvé")
    return approval
