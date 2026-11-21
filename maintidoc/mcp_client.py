"""Client MCP intégré à MaintiDoc (livrable L3.3).

Lance le serveur de maintenance en sous-processus (transport stdio) avec la clé API
de l'utilisateur courant, appelle un outil et renvoie son résultat structuré.
Les refus du serveur (accès, approbation) sont remontés sous forme de MCPToolError.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from typing import Any

from mcp import Client
from mcp.client.stdio import StdioServerParameters

from .config import ROOT_DIR, settings


class MCPToolError(RuntimeError):
    """Refus ou erreur renvoyé par un outil du serveur MCP."""


def server_parameters(api_key: str | None = None, on_behalf_of: str | None = None) -> StdioServerParameters:
    env = {"MAINTENANCE_DB": str(settings.maintenance_db), "PYTHONIOENCODING": "utf-8"}
    if on_behalf_of:
        # Backend (étape 4) : clé de service + utilisateur final ; le serveur applique
        # les droits de cet utilisateur.
        env |= {"MAINTIDOC_SERVICE_KEY": settings.mcp_service_key, "MAINTIDOC_ON_BEHALF_OF": on_behalf_of}
    else:
        env["MAINTIDOC_API_KEY"] = api_key or settings.mcp_api_key
    for name in ("MCP_SERVICE_KEY",):
        if name in os.environ:
            env[name] = os.environ[name]
    # Le secret d'approbation est lu par le serveur depuis .env ; on ne le propage que
    # s'il a été défini dans l'environnement du processus (tests, déploiement).
    if "APPROVAL_SECRET" in os.environ:
        env["APPROVAL_SECRET"] = os.environ["APPROVAL_SECRET"]
    return StdioServerParameters(command=sys.executable, args=["-m", "maintidoc.mcp_server"],
                                 env=env, cwd=str(ROOT_DIR))


def _result(result: Any) -> dict:
    text = " ".join(getattr(block, "text", "") for block in result.content).strip()
    if result.is_error:
        raise MCPToolError(re.sub(r"^Error executing tool \w+:\s*", "", text) or "erreur inconnue du serveur MCP")
    if isinstance(result.structured_content, dict):
        content = result.structured_content
        return content.get("result", content) if set(content) == {"result"} else content
    try:
        return json.loads(text)
    except ValueError:
        return {"text": text}


async def acall(tool: str, arguments: dict, api_key: str | None = None, server: Any = None,
                on_behalf_of: str | None = None) -> dict:
    """Appelle un outil. `server` permet de passer un MCPServer en mémoire (tests)."""
    target = server if server is not None else server_parameters(api_key, on_behalf_of)
    async with Client(target, read_timeout_seconds=60) as client:
        result = await client.call_tool(tool, arguments)
    # Interprété hors du contexte : une exception levée dedans serait enveloppée
    # dans un ExceptionGroup par le groupe de tâches du transport.
    return _result(result)


async def alist_tools(api_key: str | None = None) -> list[str]:
    async with Client(server_parameters(api_key)) as client:
        return [tool.name for tool in (await client.list_tools()).tools]


def call(tool: str, arguments: dict, api_key: str | None = None, on_behalf_of: str | None = None) -> dict:
    """Version synchrone (nœuds LangGraph, Streamlit, FastAPI)."""
    return asyncio.run(acall(tool, arguments, api_key, on_behalf_of=on_behalf_of))


def get_equipment_history(equipment_id: str, api_key: str | None = None, on_behalf_of: str | None = None) -> dict:
    return call("get_equipment_history", {"equipment_id": equipment_id}, api_key, on_behalf_of)


def get_open_tickets(equipment_id: str, api_key: str | None = None, on_behalf_of: str | None = None) -> dict:
    return call("get_open_tickets", {"equipment_id": equipment_id}, api_key, on_behalf_of)


def create_ticket(equipment_id: str, title: str, description: str, approval_token: str,
                  api_key: str | None = None, on_behalf_of: str | None = None) -> dict:
    return call("create_ticket", {"equipment_id": equipment_id, "title": title,
                                  "description": description, "approval_token": approval_token},
                api_key, on_behalf_of)
