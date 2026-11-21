"""Backend FastAPI de MaintiDoc (livrable L4.2).

Deux façons de s'authentifier, toutes deux vérifiées ici, côté serveur :
  * `Authorization: Bearer <clé API>` : accès direct (client Web, scripts) ;
  * bot de messagerie : `X-Bot-Secret` (secret partagé bot <-> backend) +
    `X-Channel-User-Id` (identifiant Telegram), associé au préalable à une identité
    applicative par un code de liaison à usage unique.

Le backend appelle le serveur MCP au nom de l'utilisateur (clé de service + délégation) ;
le serveur MCP réapplique ses propres contrôles (site, rôle, jeton d'approbation).

Lancement :  uvicorn maintidoc.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import io
import logging
import secrets
import sqlite3
import threading
import time
import uuid
from collections import defaultdict, deque
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import mcp_client
from .approval import content_hash
from .config import settings
from .identity import IdentityStore, Principal

log = logging.getLogger("maintidoc.api")

MAX_QUESTION = 1000
MAX_PHOTO_BYTES = 8 * 1024 * 1024
EQUIPMENT_PATTERN = r"^[A-Z]{3}-\d{3}$"


# --------------------------------------------------------------------------- dépendances partagées

class Services:
    """Objets construits une seule fois (graphe, identités) ; remplaçables dans les tests."""

    def __init__(self, identities: IdentityStore | None = None, graph=None):
        self.identities = identities or IdentityStore()
        self._graph = graph
        self._lock = threading.Lock()

    @property
    def graph(self):
        with self._lock:
            if self._graph is None:
                from langgraph.checkpoint.sqlite import SqliteSaver

                from .workflow import build_graph

                # Checkpoints persistants : une approbation en attente survit à un redémarrage.
                conn = sqlite3.connect(settings.storage_dir / "checkpoints.db", check_same_thread=False)
                self._graph = build_graph(SqliteSaver(conn))
            return self._graph


class RateLimiter:
    """Fenêtre glissante d'une minute, par utilisateur et par catégorie."""

    def __init__(self):
        self.calls: dict[tuple[str, str], deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, user_id: str, bucket: str, limit: int) -> None:
        now = time.monotonic()
        with self.lock:
            calls = self.calls[(user_id, bucket)]
            while calls and now - calls[0] > 60:
                calls.popleft()
            if len(calls) >= limit:
                raise HTTPException(429, f"Trop de requêtes ({limit}/min) : réessayez dans une minute.")
            calls.append(now)


app = FastAPI(title="MaintiDoc API", version="4.0.0",
              description="Assistant documentaire de maintenance : questions sourcées, photos, "
                          "historique et tickets avec approbation humaine.")
app.state.services = None
app.state.limiter = RateLimiter()


def services(request: Request) -> Services:
    if request.app.state.services is None:
        request.app.state.services = Services()
    return request.app.state.services


def current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    x_bot_secret: Annotated[str | None, Header()] = None,
    x_channel_user_id: Annotated[str | None, Header()] = None,
) -> Principal:
    svc = services(request)
    principal = None
    if x_bot_secret is not None:
        if not settings.bot_shared_secret or not secrets.compare_digest(x_bot_secret, settings.bot_shared_secret):
            raise HTTPException(401, "Secret du bot invalide.")
        if not x_channel_user_id:
            raise HTTPException(401, "Identifiant de messagerie manquant.")
        principal = svc.identities.by_channel("telegram", x_channel_user_id)
        if principal is None:
            raise HTTPException(403, "Compte de messagerie non associé : demandez un code de liaison à "
                                     "un administrateur puis envoyez /start <code>.")
    elif authorization and authorization.lower().startswith("bearer "):
        principal = svc.identities.by_api_key(authorization[7:].strip())
    if principal is None:
        raise HTTPException(401, "Authentification requise.")
    request.state.user_id = principal.user_id
    request.app.state.limiter.check(principal.user_id, "all", settings.rate_limit_per_minute)
    return principal


User = Annotated[Principal, Depends(current_user)]


def llm_quota(request: Request, user: User) -> Principal:
    request.app.state.limiter.check(user.user_id, "llm", settings.llm_rate_limit_per_minute)
    return user


LLMUser = Annotated[Principal, Depends(llm_quota)]


@app.middleware("http")
async def access_log(request: Request, call_next):
    # Journalisation minimale : ni question, ni photo, ni jeton, ni contenu de ticket.
    start = time.perf_counter()
    response = await call_next(request)
    route = request.scope.get("route")
    log.info("%s %s user=%s status=%s %.0fms", request.method, getattr(route, "path", request.url.path),
             getattr(request.state, "user_id", "-"), response.status_code, (time.perf_counter() - start) * 1000)
    return response


@app.exception_handler(mcp_client.MCPToolError)
def mcp_refusal(_: Request, exc: mcp_client.MCPToolError) -> JSONResponse:
    return JSONResponse(status_code=403, content={"detail": str(exc)})


# --------------------------------------------------------------------------- modèles

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION)


class LinkRequest(BaseModel):
    code: str = Field(min_length=4, max_length=32)


class DraftRequest(BaseModel):
    equipment_id: str = Field(pattern=EQUIPMENT_PATTERN)
    problem: str = Field(min_length=3, max_length=MAX_QUESTION)


class ApproveRequest(BaseModel):
    thread_id: str | None = Field(default=None, description="Brouillon à approuver (défaut : le dernier en attente)")
    content_sha256: str = Field(min_length=8, max_length=64,
                                description="Empreinte (ou préfixe ≥ 8 caractères) du contenu affiché à l'utilisateur")


class CancelRequest(BaseModel):
    thread_id: str | None = None


# --------------------------------------------------------------------------- routes

@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/version")
def version_info() -> dict:
    from .versioning import runtime_info

    return runtime_info()


@app.post("/link")
def link_account(body: LinkRequest, request: Request,
                 x_bot_secret: Annotated[str | None, Header()] = None,
                 x_channel_user_id: Annotated[str | None, Header()] = None) -> dict:
    if not settings.bot_shared_secret or not x_bot_secret or \
            not secrets.compare_digest(x_bot_secret, settings.bot_shared_secret) or not x_channel_user_id:
        raise HTTPException(401, "Secret du bot invalide.")
    request.app.state.limiter.check(f"link:{x_channel_user_id}", "link", 5)
    principal = services(request).identities.link("telegram", x_channel_user_id, body.code)
    if principal is None:
        raise HTTPException(403, "Code de liaison invalide, expiré ou déjà utilisé.")
    return {"user_id": principal.user_id, "name": principal.name, "role": principal.role}


@app.get("/me")
def me(user: User) -> dict:
    return {"user_id": user.user_id, "name": user.name, "role": user.role, "sites": list(user.sites),
            "can_create_tickets": user.can_create_tickets}


@app.post("/ask")
def ask(body: AskRequest, user: LLMUser) -> dict:
    from .multimodal import answer

    result, trace = answer(body.question)
    return result.model_dump() | {"reranker": trace.reranker}


@app.post("/analyze")
def analyze(user: LLMUser, photo: Annotated[UploadFile, File()],
            question: Annotated[str, Form(max_length=MAX_QUESTION)] = "") -> dict:
    from PIL import Image, UnidentifiedImageError

    data = photo.file.read(MAX_PHOTO_BYTES + 1)
    if len(data) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "Photo trop volumineuse (8 Mo maximum).")
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        image = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError):
        raise HTTPException(415, "Fichier non reconnu comme une image.") from None
    from .multimodal import answer

    result, trace = answer(question, image)
    return result.model_dump() | {"reranker": trace.reranker}


@app.get("/equipment/{equipment_id}/history")
def history(equipment_id: str, user: User) -> dict:
    return mcp_client.get_equipment_history(equipment_id.upper(), on_behalf_of=user.user_id)


@app.get("/equipment/{equipment_id}/tickets")
def open_tickets(equipment_id: str, user: User) -> dict:
    return mcp_client.get_open_tickets(equipment_id.upper(), on_behalf_of=user.user_id)


def _config(user: Principal, thread_id: str) -> dict:
    from .workflow import run_config

    return run_config(thread_id, user.user_id, on_behalf_of=user.user_id)


@app.post("/tickets/draft")
def draft_ticket(body: DraftRequest, request: Request, user: LLMUser) -> dict:
    from .workflow import pending_interrupt

    if not user.can_create_tickets:
        # Contrôle anticipé (confort) ; le serveur MCP refuserait de toute façon la création.
        raise HTTPException(403, f"Le rôle « {user.role} » ne permet pas de créer un ticket.")
    svc = services(request)
    thread_id = uuid.uuid4().hex
    result = svc.graph.invoke({"question": f"Préparer un ticket pour {body.equipment_id} : {body.problem}",
                               "equipment_id": body.equipment_id, "steps": []}, _config(user, thread_id))
    pending = pending_interrupt(result)
    if pending is None:
        raise HTTPException(403 if result.get("error") else 422,
                            result.get("error") or result.get("response") or "Brouillon impossible.")
    draft = pending["draft"]
    digest = content_hash(draft["title"], draft["description"])
    svc.identities.add_pending(thread_id, user.user_id, body.equipment_id, digest)
    return {"thread_id": thread_id, "draft": draft, "content_sha256": digest,
            "similar_tickets": pending["similar_tickets"], "steps": result["steps"],
            "expires_in_seconds": 15 * 60}


def _owned_pending(svc: Services, user: Principal, thread_id: str | None) -> dict:
    pending = svc.identities.get_pending(thread_id) if thread_id else svc.identities.latest_pending(user.user_id)
    # Même réponse si le brouillon n'existe pas ou appartient à un autre utilisateur.
    if pending is None or pending["user_id"] != user.user_id:
        raise HTTPException(404, "Aucun brouillon en attente.")
    if pending["status"] != "en_attente":
        raise HTTPException(409, f"Ce brouillon n'est plus en attente (état : {pending['status']}).")
    if pending["expires_at"] < time.time():
        svc.identities.close_pending(pending["thread_id"], "expiré")
        raise HTTPException(410, "Brouillon expiré : préparez-en un nouveau.")
    return pending


@app.post("/tickets/approve")
def approve(body: ApproveRequest, request: Request, user: User) -> dict:
    from langgraph.types import Command

    svc = services(request)
    pending = _owned_pending(svc, user, body.thread_id)
    # L'utilisateur approuve le contenu exact qui lui a été montré.
    if not pending["content_sha256"].startswith(body.content_sha256.lower()):
        raise HTTPException(409, "Le contenu approuvé ne correspond pas au brouillon en attente.")
    if not svc.identities.close_pending(pending["thread_id"], "en_cours"):
        raise HTTPException(409, "Approbation déjà en cours de traitement.")
    result = svc.graph.invoke(Command(resume={"action": "approve"}), _config(user, pending["thread_id"]))
    if result.get("error") or not result.get("ticket"):
        svc.identities.finalize(pending["thread_id"], "refusé")
        raise HTTPException(403, result.get("error") or "Création refusée.")
    ticket = result["ticket"]
    svc.identities.finalize(pending["thread_id"], "créé", ticket["ticket"]["ticket_id"])
    return {"created": ticket["created"], "duplicate": ticket["duplicate"], "ticket": ticket["ticket"],
            "steps": result["steps"]}


@app.post("/tickets/cancel")
def cancel(body: CancelRequest, request: Request, user: User) -> dict:
    from langgraph.types import Command

    svc = services(request)
    pending = _owned_pending(svc, user, body.thread_id)
    if not svc.identities.close_pending(pending["thread_id"], "annulé"):
        raise HTTPException(409, "Ce brouillon n'est plus en attente.")
    svc.graph.invoke(Command(resume={"action": "cancel"}), _config(user, pending["thread_id"]))
    return {"cancelled": True, "thread_id": pending["thread_id"]}
