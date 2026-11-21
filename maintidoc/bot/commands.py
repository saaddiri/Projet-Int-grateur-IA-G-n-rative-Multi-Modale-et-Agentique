"""Logique des commandes du bot, indépendante de Telegram (testable).

Chaque commande reçoit l'identifiant Telegram de l'expéditeur et ses arguments, appelle le
backend FastAPI et renvoie le texte de réponse. Le bot ne prend aucune décision de sécurité :
identité, droits et approbation sont vérifiés par le backend puis par le serveur MCP.
"""

from __future__ import annotations

import re

import httpx

from ..config import settings

HELP = """MaintiDoc — assistant documentaire de maintenance
/ask <question> — question sur la documentation (réponse sourcée)
/analyze — envoyez une photo (légende = question facultative)
/history <équipement> — historique, ex. /history CMP-101
/tickets <équipement> — tickets ouverts
/create_ticket <équipement> <problème> — préparer un brouillon de ticket
/approve <code> — approuver le contenu exact du brouillon affiché
/cancel — annuler le brouillon en attente
Le bot ne pilote aucun équipement et ne remplace pas un technicien habilité."""

EQUIPMENT_RE = re.compile(r"^[A-Za-z]{3}-?\d{3}$")
UNCERTAINTY = {"faible": "🟢", "moyenne": "🟠", "élevée": "🔴"}


class Backend:
    """Client HTTP du backend, authentifié comme bot pour le compte d'un utilisateur Telegram."""

    def __init__(self, base_url: str | None = None, secret: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, timeout: float = 900):
        self.base_url = base_url or settings.api_base_url
        self.secret = secret if secret is not None else settings.bot_shared_secret
        self.transport = transport
        self.timeout = timeout

    async def request(self, method: str, path: str, telegram_id: int | str, **kwargs) -> tuple[int, dict]:
        headers = {"X-Bot-Secret": self.secret, "X-Channel-User-Id": str(telegram_id)}
        async with httpx.AsyncClient(base_url=self.base_url, transport=self.transport,
                                     timeout=self.timeout) as client:
            response = await client.request(method, path, headers=headers, **kwargs)
        try:
            data = response.json()
        except ValueError:
            data = {"detail": response.text}
        return response.status_code, data


def _error(status: int, data: dict) -> str:
    detail = data.get("detail", "erreur")
    if isinstance(detail, list):  # erreurs de validation FastAPI
        detail = "; ".join(d.get("msg", "") for d in detail)
    return f"⛔ {detail}" if status in (401, 403, 404, 409, 410, 413, 415, 429) else f"⚠️ Erreur ({status}) : {detail}"


def _equipment(arg: str) -> str | None:
    if not EQUIPMENT_RE.match(arg):
        return None
    return f"{arg[:3].upper()}-{arg[-3:]}"


def format_answer(data: dict) -> str:
    lines = []
    if data.get("needs_clarification"):
        lines.append(f"❓ {data['clarification_question']}")
    lines.append(data["answer"])
    if data.get("citations"):
        lines.append("\nSources :")
        for c in data["citations"]:
            auto = " (attribution automatique)" if c.get("attribution") == "automatique" else ""
            lines.append(f"[{c['ref']}] {c['source']}, p. {c['page']} — {c['content_type']}{auto}")
    lines.append(f"\n{UNCERTAINTY.get(data['uncertainty'], '')} Incertitude {data['uncertainty']}")
    return "\n".join(lines)


async def start(api: Backend, telegram_id, args: list[str]) -> str:
    if args:
        status, data = await api.request("POST", "/link", telegram_id, json={"code": args[0]})
        if status != 200:
            return _error(status, data)
        return f"✅ Compte associé à {data['name']} ({data['role']}).\n\n{HELP}"
    status, data = await api.request("GET", "/me", telegram_id)
    if status != 200:
        return _error(status, data)
    return f"Bonjour {data['name']} ({data['role']}).\n\n{HELP}"


async def ask(api: Backend, telegram_id, args: list[str]) -> str:
    question = " ".join(args).strip()
    if not question:
        return "Usage : /ask <question>"
    status, data = await api.request("POST", "/ask", telegram_id, json={"question": question})
    return format_answer(data) if status == 200 else _error(status, data)


async def analyze(api: Backend, telegram_id, photo: bytes, question: str = "") -> str:
    status, data = await api.request("POST", "/analyze", telegram_id,
                                     files={"photo": ("photo.jpg", photo, "image/jpeg")},
                                     data={"question": question})
    if status != 200:
        return _error(status, data)
    text = format_answer(data)
    analysis = data.get("photo_analysis") or {}
    if analysis:
        text = f"📷 {analysis.get('description', '')[:300]} (confiance {analysis.get('confidence')})\n\n" + text
    return text


async def history(api: Backend, telegram_id, args: list[str]) -> str:
    eq = _equipment(args[0]) if args else None
    if eq is None:
        return "Usage : /history <équipement>, ex. /history CMP-101"
    status, data = await api.request("GET", f"/equipment/{eq}/history", telegram_id)
    if status != 200:
        return _error(status, data)
    lines = [f"🗂️ {eq} — {data['equipment']['name']}"]
    lines += [f"• {h['date']} ({h['type']}) : {h['summary']}" for h in data["interventions"]]
    return "\n".join(lines)


async def tickets(api: Backend, telegram_id, args: list[str]) -> str:
    eq = _equipment(args[0]) if args else None
    if eq is None:
        return "Usage : /tickets <équipement>, ex. /tickets CMP-101"
    status, data = await api.request("GET", f"/equipment/{eq}/tickets", telegram_id)
    if status != 200:
        return _error(status, data)
    if not data["tickets"]:
        return f"Aucun ticket ouvert sur {eq}."
    return "\n".join([f"🎫 Tickets ouverts sur {eq} :"] +
                     [f"• {t['ticket_id']} — {t['title']} ({t['created_at'][:10]})" for t in data["tickets"]])


async def create_ticket(api: Backend, telegram_id, args: list[str]) -> str:
    eq = _equipment(args[0]) if args else None
    problem = " ".join(args[1:]).strip()
    if eq is None or not problem:
        return "Usage : /create_ticket <équipement> <problème>, ex. /create_ticket CMP-101 vibrations au démarrage"
    status, data = await api.request("POST", "/tickets/draft", telegram_id,
                                     json={"equipment_id": eq, "problem": problem})
    if status != 200:
        return _error(status, data)
    draft, code = data["draft"], data["content_sha256"][:8]
    lines = ["📝 BROUILLON — rien n'est créé sans votre approbation", f"Équipement : {draft['equipment_id']}",
             f"Titre : {draft['title']}", f"Description :\n{draft['description']}"]
    for t in data.get("similar_tickets", []):
        lines.append(f"⚠️ Ticket ouvert similaire : {t['ticket_id']} — {t['title']}")
    lines.append(f"\nPour créer CE contenu : /approve {code}\nPour abandonner : /cancel  (expire dans 15 min)")
    return "\n".join(lines)


async def approve(api: Backend, telegram_id, args: list[str]) -> str:
    if not args:
        return "Usage : /approve <code affiché sous le brouillon>"
    status, data = await api.request("POST", "/tickets/approve", telegram_id, json={"content_sha256": args[0]})
    if status != 200:
        return _error(status, data)
    t = data["ticket"]
    if data["duplicate"]:
        return f"ℹ️ Ticket déjà existant : {t['ticket_id']} — {t['title']} (aucun doublon créé)."
    return f"✅ Ticket {t['ticket_id']} créé sur {t['equipment_id']} : {t['title']}"


async def cancel(api: Backend, telegram_id, args: list[str]) -> str:
    status, data = await api.request("POST", "/tickets/cancel", telegram_id, json={})
    return "🗑️ Brouillon annulé : aucun ticket créé." if status == 200 else _error(status, data)
