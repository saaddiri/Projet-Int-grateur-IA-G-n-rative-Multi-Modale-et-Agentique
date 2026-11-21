"""Serveur MCP de démonstration simulant le service de maintenance (livrable L3.2).

Outils exposés :
  * get_equipment_history(equipment_id)
  * get_open_tickets(equipment_id)
  * create_ticket(equipment_id, title, description, approval_token)

Contrôles côté serveur (L3.5) — le client et le LLM ne sont pas considérés comme fiables :
  * identité : déterminée par la clé API reçue au lancement du serveur
    (MAINTIDOC_API_KEY), jamais par un argument d'outil que le LLM pourrait choisir ;
  * autorisation : l'utilisateur doit avoir accès au site de l'équipement ; seuls les
    rôles « technicien » et « superviseur » peuvent créer un ticket ;
  * approbation : jeton signé lié à l'utilisateur, à l'équipement et au contenu exact ;
  * idempotence : un même jeton (approval_id) ne crée jamais deux tickets, et un
    contenu identique déjà ouvert sur l'équipement renvoie le ticket existant ;
  * journalisation minimale des appels (sans le contenu des jetons).

Lancement (stdio) :  python -m maintidoc.mcp_server
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from .approval import ApprovalError, content_hash, verify_token
from .config import settings

log = logging.getLogger("maintidoc.mcp")

CAN_CREATE = {"technicien", "superviseur"}
MAX_TITLE, MAX_DESCRIPTION = 120, 2000

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (user_id TEXT PRIMARY KEY, name TEXT, role TEXT, sites TEXT, api_key_sha256 TEXT UNIQUE);
CREATE TABLE IF NOT EXISTS equipment (equipment_id TEXT PRIMARY KEY, name TEXT, site TEXT);
CREATE TABLE IF NOT EXISTS history (id INTEGER PRIMARY KEY, equipment_id TEXT, date TEXT, type TEXT, summary TEXT, technician TEXT);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY, equipment_id TEXT, title TEXT, description TEXT, status TEXT,
    created_by TEXT, created_at TEXT, approval_id TEXT UNIQUE, content_sha256 TEXT
);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, at TEXT, user_id TEXT, tool TEXT, equipment_id TEXT, outcome TEXT);
"""


def init_db(db_path: Path, seed_path: Path) -> None:
    """Crée la base SQLite et charge les données fictives si elle est vide."""
    with sqlite3.connect(db_path) as db:
        db.executescript(SCHEMA)
        if db.execute("SELECT COUNT(*) FROM users").fetchone()[0]:
            return
        seed = json.loads(seed_path.read_text(encoding="utf-8"))
        db.executemany("INSERT INTO users VALUES (?,?,?,?,?)", [
            (u["user_id"], u["name"], u["role"], json.dumps(u["sites"]), u["api_key_sha256"]) for u in seed["users"]])
        db.executemany("INSERT INTO equipment VALUES (?,?,?)", [
            (e["equipment_id"], e["name"], e["site"]) for e in seed["equipment"]])
        db.executemany("INSERT INTO history (equipment_id, date, type, summary, technician) VALUES (?,?,?,?,?)", [
            (h["equipment_id"], h["date"], h["type"], h["summary"], h["technician"]) for h in seed["history"]])
        db.executemany("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,NULL,?)", [
            (t["ticket_id"], t["equipment_id"], t["title"], t["description"], t["status"], t["created_by"],
             t["created_at"], content_hash(t["title"], t["description"])) for t in seed["tickets"]])


def create_server(api_key: str | None = None, db_path: Path | None = None,
                  seed_path: Path | None = None, approval_secret: str | None = None,
                  service_key: str | None = None, on_behalf_of: str | None = None) -> MCPServer:
    """Identité : soit la clé API de l'utilisateur, soit (étape 4) la clé de service du
    backend accompagnée de l'utilisateur pour le compte duquel il agit. Dans les deux cas
    les droits appliqués sont ceux de l'utilisateur final."""
    db_path = Path(db_path or settings.maintenance_db)
    init_db(db_path, Path(seed_path or settings.maintenance_seed))
    secret = approval_secret or settings.approval_secret

    def connect() -> sqlite3.Connection:
        db = sqlite3.connect(db_path)
        db.row_factory = sqlite3.Row
        return db

    key_hash = hashlib.sha256((api_key or "").encode()).hexdigest()
    with connect() as db:
        if service_key and on_behalf_of:
            trusted = bool(settings.mcp_service_key) and hmac.compare_digest(service_key, settings.mcp_service_key)
            row = db.execute("SELECT * FROM users WHERE user_id = ?", (on_behalf_of,)).fetchone() if trusted else None
        else:
            row = db.execute("SELECT * FROM users WHERE api_key_sha256 = ?", (key_hash,)).fetchone()
    user = dict(row) | {"sites": json.loads(row["sites"])} if row else None

    def audit(tool: str, equipment_id: str, outcome: str) -> None:
        with connect() as db:
            db.execute("INSERT INTO audit (at, user_id, tool, equipment_id, outcome) VALUES (?,?,?,?,?)",
                       (datetime.now().isoformat(timespec="seconds"), user and user["user_id"], tool,
                        equipment_id, outcome))
        log.info("%s %s %s -> %s", user and user["user_id"], tool, equipment_id, outcome)

    def authorize(tool: str, equipment_id: str) -> dict:
        if user is None:
            audit(tool, equipment_id, "refus: non authentifié")
            raise ToolError("Accès refusé : clé API absente ou inconnue.")
        with connect() as db:
            equipment = db.execute("SELECT * FROM equipment WHERE equipment_id = ?", (equipment_id,)).fetchone()
        if equipment is None:
            audit(tool, equipment_id, "refus: équipement inconnu")
            raise ToolError(f"Équipement inconnu : {equipment_id}.")
        if equipment["site"] not in user["sites"]:
            audit(tool, equipment_id, "refus: site non autorisé")
            raise ToolError(f"Accès refusé : {user['user_id']} n'est pas habilité sur le site {equipment['site']}.")
        return dict(equipment)

    server = MCPServer(
        name="maintidoc-maintenance",
        instructions="Service de maintenance simulé : historique, tickets ouverts, création de tickets approuvés.",
    )

    @server.tool()
    def get_equipment_history(equipment_id: str) -> dict:
        """Historique des interventions d'un équipement (le plus récent en premier)."""
        equipment = authorize("get_equipment_history", equipment_id)
        with connect() as db:
            rows = db.execute("SELECT date, type, summary, technician FROM history WHERE equipment_id = ? "
                              "ORDER BY date DESC", (equipment_id,)).fetchall()
        audit("get_equipment_history", equipment_id, "ok")
        return {"equipment": equipment, "interventions": [dict(r) for r in rows]}

    @server.tool()
    def get_open_tickets(equipment_id: str) -> dict:
        """Tickets encore ouverts sur un équipement."""
        authorize("get_open_tickets", equipment_id)
        with connect() as db:
            rows = db.execute("SELECT ticket_id, title, description, status, created_by, created_at FROM tickets "
                              "WHERE equipment_id = ? AND status = 'ouvert' ORDER BY created_at DESC",
                              (equipment_id,)).fetchall()
        audit("get_open_tickets", equipment_id, "ok")
        return {"equipment_id": equipment_id, "tickets": [dict(r) for r in rows]}

    @server.tool()
    def create_ticket(equipment_id: str, title: str, description: str, approval_token: str) -> dict:
        """Crée un ticket. Exige un jeton d'approbation humaine lié au contenu exact."""
        authorize("create_ticket", equipment_id)
        if user["role"] not in CAN_CREATE:
            audit("create_ticket", equipment_id, "refus: rôle")
            raise ToolError(f"Accès refusé : le rôle « {user['role']} » ne permet pas de créer un ticket.")
        title, description = title.strip(), description.strip()
        if not title or len(title) > MAX_TITLE or len(description) > MAX_DESCRIPTION:
            audit("create_ticket", equipment_id, "refus: contenu invalide")
            raise ToolError("Titre vide ou contenu trop long.")
        try:
            approval = verify_token(approval_token, user_id=user["user_id"], equipment_id=equipment_id,
                                    title=title, description=description, secret=secret)
        except ApprovalError as exc:
            audit("create_ticket", equipment_id, f"refus: {exc}")
            raise ToolError(f"Approbation invalide : {exc}.") from None

        digest = content_hash(title, description)
        with connect() as db:
            # Idempotence : même approbation, ou même contenu déjà ouvert sur l'équipement.
            existing = db.execute(
                "SELECT * FROM tickets WHERE approval_id = ? OR (equipment_id = ? AND content_sha256 = ? "
                "AND status = 'ouvert')", (approval.approval_id, equipment_id, digest)).fetchone()
            if existing:
                audit("create_ticket", equipment_id, f"idempotent: {existing['ticket_id']}")
                return {"created": False, "duplicate": True, "ticket": _public(existing)}
            number = db.execute("SELECT COUNT(*) FROM tickets").fetchone()[0] + 1
            ticket_id = f"T-{number:04d}"
            db.execute("INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?,?)", (
                ticket_id, equipment_id, title, description, "ouvert", user["user_id"],
                datetime.now().isoformat(timespec="seconds"), approval.approval_id, digest))
            row = db.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
        audit("create_ticket", equipment_id, f"créé: {ticket_id}")
        return {"created": True, "duplicate": False, "ticket": _public(row)}

    return server


def _public(row: sqlite3.Row) -> dict:
    return {k: row[k] for k in ("ticket_id", "equipment_id", "title", "description", "status",
                                "created_by", "created_at")}


def main() -> None:
    # Les logs vont sur stderr : stdout est réservé au protocole MCP.
    logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s %(message)s")
    create_server(api_key=os.environ.get("MAINTIDOC_API_KEY"),
                  service_key=os.environ.get("MAINTIDOC_SERVICE_KEY"),
                  on_behalf_of=os.environ.get("MAINTIDOC_ON_BEHALF_OF")).run("stdio")


if __name__ == "__main__":
    main()
