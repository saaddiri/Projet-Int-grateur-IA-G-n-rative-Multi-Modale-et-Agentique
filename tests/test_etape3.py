"""Tests de l'étape 3 : jetons d'approbation, contrôles du serveur MCP, workflow LangGraph.

Le serveur MCP est testé en mémoire (sans sous-processus) ; le workflow est testé avec
les nœuds LLM et documentaires remplacés, pour vérifier les transitions et l'interruption.
"""

from __future__ import annotations

import asyncio

import pytest
from langgraph.types import Command

from maintidoc import mcp_client, workflow
from maintidoc.approval import ApprovalError, issue_token, verify_token
from maintidoc.config import settings
from maintidoc.mcp_server import create_server

SECRET = "secret-de-test"


def _token(user="tech.dupont", eq="CMP-101", title="Fuite", desc="Fuite au raccord.", **kw):
    return issue_token(user, eq, title, desc, secret=SECRET, **kw)


def test_token_lie_au_contenu_utilisateur_equipement():
    token = _token()
    assert verify_token(token, user_id="tech.dupont", equipment_id="CMP-101", title="Fuite",
                        description="Fuite au raccord.", secret=SECRET)
    for kwargs in ({"user_id": "chef.martin"}, {"equipment_id": "PMP-204"}, {"description": "Autre."}):
        args = {"user_id": "tech.dupont", "equipment_id": "CMP-101", "title": "Fuite",
                "description": "Fuite au raccord."} | kwargs
        with pytest.raises(ApprovalError):
            verify_token(token, secret=SECRET, **args)
    with pytest.raises(ApprovalError):
        verify_token(token, user_id="tech.dupont", equipment_id="CMP-101", title="Fuite",
                     description="Fuite au raccord.", secret="autre-secret")
    with pytest.raises(ApprovalError, match="expiré"):
        verify_token(_token(ttl=-1), user_id="tech.dupont", equipment_id="CMP-101", title="Fuite",
                     description="Fuite au raccord.", secret=SECRET)


@pytest.fixture
def server_for(tmp_path):
    db = tmp_path / "maintenance.db"

    def make(api_key):
        return create_server(api_key=api_key, db_path=db, seed_path=settings.maintenance_seed,
                             approval_secret=SECRET)
    return make


def call(server, tool, **arguments):
    return asyncio.run(mcp_client.acall(tool, arguments, server=server))


def test_serveur_consultation_et_droits_par_site(server_for):
    tech = server_for("demo-key-dupont")
    assert call(tech, "get_equipment_history", equipment_id="CMP-101")["interventions"]
    assert [t["ticket_id"] for t in call(tech, "get_open_tickets", equipment_id="CMP-101")["tickets"]] == ["T-0001"]
    with pytest.raises(mcp_client.MCPToolError, match="site"):
        call(tech, "get_open_tickets", equipment_id="CNV-310")
    with pytest.raises(mcp_client.MCPToolError, match="clé API"):
        call(server_for("cle-inconnue"), "get_equipment_history", equipment_id="CMP-101")


def test_creation_exige_approbation_et_reste_idempotente(server_for):
    tech = server_for("demo-key-dupont")
    args = {"equipment_id": "CMP-101", "title": "Fuite", "description": "Fuite au raccord."}
    with pytest.raises(mcp_client.MCPToolError, match="Approbation invalide"):
        call(tech, "create_ticket", approval_token="faux.jeton", **args)
    token = _token()
    first = call(tech, "create_ticket", approval_token=token, **args)
    again = call(tech, "create_ticket", approval_token=token, **args)
    assert first["created"] and not again["created"] and again["duplicate"]
    assert first["ticket"]["ticket_id"] == again["ticket"]["ticket_id"]
    with pytest.raises(mcp_client.MCPToolError, match="contenu"):
        call(tech, "create_ticket", approval_token=token, **(args | {"description": "Modifié."}))
    with pytest.raises(mcp_client.MCPToolError, match="lecteur"):
        call(server_for("demo-key-leroy"), "create_ticket",
             approval_token=_token(user="stagiaire.leroy"), **args)


@pytest.fixture
def graph(monkeypatch):
    created = []

    monkeypatch.setattr(workflow.mcp_client, "get_equipment_history",
                        lambda eq, key=None: {"equipment": {"equipment_id": eq, "name": "Compresseur"},
                                              "interventions": [{"date": "2026-11-15", "type": "observation",
                                                                 "summary": "Vibrations"}]})
    monkeypatch.setattr(workflow.mcp_client, "get_open_tickets", lambda eq, key=None: {"tickets": []})

    def fake_create(eq, title, description, token, key=None):
        created.append((eq, title, description, token))
        return {"created": True, "duplicate": False,
                "ticket": {"ticket_id": "T-0099", "equipment_id": eq, "title": title}}

    monkeypatch.setattr(workflow.mcp_client, "create_ticket", fake_create)
    monkeypatch.setattr(workflow, "search_docs", lambda state, config: {"doc_answer": {"answer": "doc", "citations": []},
                                                                       "steps": ["rechercher_documentation"]})
    monkeypatch.setattr(workflow, "draft_ticket", lambda state: {
        "draft": {"equipment_id": state["equipment_id"], "title": "Vibrations", "description": "Constat."},
        "similar_tickets": [], "steps": ["rediger_brouillon"]})
    return workflow.build_graph(), created


def _config(name):
    return workflow.run_config(name, "tech.dupont", "demo-key-dupont")


def test_workflow_interrompt_avant_creation_puis_cree_apres_approbation(graph):
    g, created = graph
    config = _config("t1")
    result = g.invoke({"question": "Préparer un ticket pour CMP-101 qui vibre", "steps": []}, config)
    request = workflow.pending_interrupt(result)
    assert request and request["draft"]["title"] == "Vibrations"
    assert created == []  # rien n'est créé tant que l'humain n'a pas répondu
    result = g.invoke(Command(resume={"action": "approve", "title": "Vibrations",
                                      "description": "Constat modifié."}), config)
    assert len(created) == 1 and created[0][2] == "Constat modifié."
    verify_token(created[0][3], user_id="tech.dupont", equipment_id="CMP-101", title="Vibrations",
                 description="Constat modifié.")
    assert "T-0099" in result["response"]


def test_workflow_annulation_et_clarification(graph):
    g, created = graph
    config = _config("t2")
    g.invoke({"question": "Ouvrir un ticket pour CMP-101", "steps": []}, config)
    result = g.invoke(Command(resume={"action": "cancel"}), config)
    assert created == [] and "aucun ticket" in result["response"]
    result = g.invoke({"question": "Crée un ticket pour la machine qui fuit", "steps": []}, _config("t3"))
    assert "identifiant" in result["response"] and workflow.pending_interrupt(result) is None


def test_workflow_historique_sans_llm(graph):
    g, _ = graph
    result = g.invoke({"question": "Historique du cmp101 ?", "steps": []}, _config("t4"))
    assert result["equipment_id"] == "CMP-101" and "Vibrations" in result["response"]
