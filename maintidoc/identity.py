"""Identités applicatives et actions en attente du backend (livrable L4.2).

* Les utilisateurs et leurs rôles proviennent du référentiel (data/maintenance/seed.json).
* Un compte de messagerie (Telegram) n'est associé à un utilisateur qu'avec un code de
  liaison à usage unique, émis par un administrateur (`python -m scripts.link_account`) et
  valable 10 minutes : connaître l'identifiant Telegram d'un collègue ne suffit pas.
* Les actions en attente (brouillons de tickets) sont rattachées à leur propriétaire, à
  l'empreinte du contenu présenté et à une date d'expiration.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .config import settings

LINK_CODE_TTL = 10 * 60
PENDING_TTL = 15 * 60

SCHEMA = """
CREATE TABLE IF NOT EXISTS identities (channel TEXT, external_id TEXT, user_id TEXT, linked_at REAL,
                                       PRIMARY KEY (channel, external_id));
CREATE TABLE IF NOT EXISTS link_codes (code_sha256 TEXT PRIMARY KEY, user_id TEXT, expires_at REAL, used INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS pending_actions (
    thread_id TEXT PRIMARY KEY, user_id TEXT, equipment_id TEXT, content_sha256 TEXT,
    status TEXT, created_at REAL, expires_at REAL, ticket_id TEXT
);
"""


@dataclass(frozen=True)
class Principal:
    user_id: str
    name: str
    role: str
    sites: tuple[str, ...]
    channel: str  # "api" ou "telegram"

    @property
    def can_create_tickets(self) -> bool:
        return self.role in ("technicien", "superviseur")


class IdentityStore:
    def __init__(self, db_path: Path | None = None, seed_path: Path | None = None):
        self.db_path = Path(db_path or settings.storage_dir / "backend.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        seed = json.loads(Path(seed_path or settings.maintenance_seed).read_text(encoding="utf-8"))
        self.users = {u["user_id"]: u for u in seed["users"]}
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        db.row_factory = sqlite3.Row
        return db

    def _principal(self, user_id: str, channel: str) -> Principal | None:
        user = self.users.get(user_id)
        if user is None:
            return None
        return Principal(user_id, user["name"], user["role"], tuple(user["sites"]), channel)

    # --- authentification
    def by_api_key(self, api_key: str) -> Principal | None:
        digest = hashlib.sha256(api_key.encode()).hexdigest()
        for user in self.users.values():
            if secrets.compare_digest(user["api_key_sha256"], digest):
                return self._principal(user["user_id"], "api")
        return None

    def by_channel(self, channel: str, external_id: str) -> Principal | None:
        with self._db() as db:
            row = db.execute("SELECT user_id FROM identities WHERE channel = ? AND external_id = ?",
                             (channel, external_id)).fetchone()
        return self._principal(row["user_id"], channel) if row else None

    # --- liaison des comptes de messagerie
    def issue_link_code(self, user_id: str) -> str:
        if user_id not in self.users:
            raise KeyError(user_id)
        code = secrets.token_hex(4).upper()
        with self._db() as db:
            db.execute("INSERT INTO link_codes VALUES (?,?,?,0)",
                       (hashlib.sha256(code.encode()).hexdigest(), user_id, time.time() + LINK_CODE_TTL))
        return code

    def link(self, channel: str, external_id: str, code: str) -> Principal | None:
        digest = hashlib.sha256(code.strip().upper().encode()).hexdigest()
        with self._db() as db:
            row = db.execute("SELECT * FROM link_codes WHERE code_sha256 = ?", (digest,)).fetchone()
            if row is None or row["used"] or row["expires_at"] < time.time():
                return None
            db.execute("UPDATE link_codes SET used = 1 WHERE code_sha256 = ?", (digest,))
            db.execute("INSERT OR REPLACE INTO identities VALUES (?,?,?,?)",
                       (channel, external_id, row["user_id"], time.time()))
        return self._principal(row["user_id"], channel)

    # --- actions en attente d'approbation
    def add_pending(self, thread_id: str, user_id: str, equipment_id: str, content_sha256: str) -> None:
        now = time.time()
        with self._db() as db:
            db.execute("UPDATE pending_actions SET status = 'remplacé' WHERE user_id = ? AND status = 'en_attente'",
                       (user_id,))
            db.execute("INSERT INTO pending_actions VALUES (?,?,?,?,?,?,?,NULL)",
                       (thread_id, user_id, equipment_id, content_sha256, "en_attente", now, now + PENDING_TTL))

    def get_pending(self, thread_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM pending_actions WHERE thread_id = ?", (thread_id,)).fetchone()
        return dict(row) if row else None

    def latest_pending(self, user_id: str) -> dict | None:
        with self._db() as db:
            row = db.execute("SELECT * FROM pending_actions WHERE user_id = ? AND status = 'en_attente' "
                             "ORDER BY created_at DESC LIMIT 1", (user_id,)).fetchone()
        return dict(row) if row else None

    def close_pending(self, thread_id: str, status: str, ticket_id: str | None = None) -> bool:
        """Passe l'action à l'état final ; renvoie False si elle n'était plus en attente."""
        with self._db() as db:
            cursor = db.execute("UPDATE pending_actions SET status = ?, ticket_id = ? "
                                "WHERE thread_id = ? AND status = 'en_attente'", (status, ticket_id, thread_id))
        return cursor.rowcount == 1

    def finalize(self, thread_id: str, status: str, ticket_id: str | None = None) -> None:
        with self._db() as db:
            db.execute("UPDATE pending_actions SET status = ?, ticket_id = ? WHERE thread_id = ?",
                       (status, ticket_id, thread_id))
