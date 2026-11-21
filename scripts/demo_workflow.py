"""Démonstration de bout en bout de l'étape 3 (livrable L3.6).

Scénarios :
  1. consultation de l'historique et des tickets ouverts (MCP) ;
  2. demande de ticket -> brouillon -> interruption -> approbation humaine -> création ;
     puis rejeu du même jeton : aucun doublon (idempotence) ;
  3. le même scénario annulé par l'humain : aucun ticket créé ;
  4. utilisateur au rôle « lecteur » qui approuve : création refusée par le serveur ;
  5. jeton approuvé mais contenu modifié ensuite : création refusée par le serveur.

Usage :
  python -m scripts.demo_workflow            # approbations demandées au clavier
  python -m scripts.demo_workflow --auto     # approbations simulées (démonstration non interactive)
"""

from __future__ import annotations

import argparse
import json
import time
import uuid

from langgraph.types import Command

from maintidoc import mcp_client
from maintidoc.approval import issue_token
from maintidoc.workflow import build_graph, pending_interrupt, run_config

USERS = {
    "tech.dupont": "demo-key-dupont",
    "stagiaire.leroy": "demo-key-leroy",
}

TICKET_QUESTION = ("Le compresseur CMP-101 vibre fortement au démarrage depuis ce matin, "
                   "peux-tu préparer un ticket ?")


def show(result: dict) -> None:
    for step in result.get("steps", []):
        print(f"   · {step}")
    if result.get("response"):
        print("   → " + result["response"].replace("\n", "\n     "))


def ask_human(request: dict, auto: str | None) -> dict:
    draft = request["draft"]
    print("\n   ┌─ APPROBATION HUMAINE REQUISE ─────────────────────────")
    print(f"   │ Équipement : {draft['equipment_id']}")
    print(f"   │ Titre      : {draft['title']}")
    print("   │ Description: " + draft["description"].replace("\n", "\n   │              "))
    for t in request.get("similar_tickets", []):
        print(f"   │ ⚠ Ticket ouvert similaire : {t['ticket_id']} « {t['title']} »")
    print("   └───────────────────────────────────────────────────────")
    choice = auto or input("   Approuver ce contenu exact ? [o = oui / n = annuler] ").strip().lower()
    print(f"   Décision : {'approuvé' if choice in ('o', 'oui', 'approve') else 'annulé'}")
    return {"action": "approve" if choice in ("o", "oui", "approve") else "cancel"}


def run(graph, question: str, user_id: str, decision: str | None, label: str) -> dict:
    print(f"\n=== {label}\n   utilisateur : {user_id}\n   demande : {question}")
    config = run_config(uuid.uuid4().hex, user_id, USERS[user_id])
    start = time.time()
    result = graph.invoke({"question": question, "steps": []}, config)
    request = pending_interrupt(result)
    if request:
        show(result)
        print("   (workflow interrompu : en attente de l'humain)")
        result = graph.invoke(Command(resume=ask_human(request, decision)), config)
    show({**result, "steps": result["steps"][-3:] if request else result["steps"]})
    print(f"   durée : {time.time() - start:.0f} s")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto", action="store_true", help="simule les décisions humaines")
    args = parser.parse_args()
    auto = (lambda choice: choice) if args.auto else (lambda choice: None)

    print("Outils MCP disponibles :", mcp_client.asyncio.run(mcp_client.alist_tools(USERS["tech.dupont"])))
    graph = build_graph()

    run(graph, "Quel est l'historique du CMP-101 ?", "tech.dupont", None, "1. Historique (MCP)")

    result = run(graph, TICKET_QUESTION, "tech.dupont", auto("o"), "2. Ticket approuvé")
    decision = result.get("decision", {})
    if decision.get("action") == "approve":
        print("\n   Rejeu du même jeton d'approbation (double clic, nouvelle tentative réseau) :")
        replay = mcp_client.create_ticket("CMP-101", decision["title"], decision["description"],
                                          decision["approval_token"], USERS["tech.dupont"])
        print(f"   → created={replay['created']} duplicate={replay['duplicate']} "
              f"ticket={replay['ticket']['ticket_id']}")

        print("\n=== 5. Contenu modifié après approbation")
        try:
            mcp_client.create_ticket("CMP-101", decision["title"], decision["description"] + " Remplacer le moteur.",
                                     decision["approval_token"], USERS["tech.dupont"])
            print("   → ACCEPTÉ (anomalie)")
        except mcp_client.MCPToolError as exc:
            print(f"   → refusé par le serveur : {exc}")

    run(graph, TICKET_QUESTION.replace("vibre fortement", "fait un bruit de claquement"), "tech.dupont",
        auto("n"), "3. Ticket annulé par l'humain")

    run(graph, "Signaler une fuite d'huile sur CMP-101 avec un ticket.", "stagiaire.leroy", auto("o"),
        "4. Rôle lecteur : création interdite")

    tickets = mcp_client.get_open_tickets("CMP-101", USERS["tech.dupont"])["tickets"]
    print("\nTickets ouverts sur CMP-101 :", json.dumps([t["ticket_id"] + " " + t["title"] for t in tickets],
                                                      ensure_ascii=False))
    # Vérification indépendante : jeton émis pour un autre utilisateur
    token = issue_token("chef.martin", "CMP-101", "x", "y")
    try:
        mcp_client.create_ticket("CMP-101", "x", "y", token, USERS["tech.dupont"])
    except mcp_client.MCPToolError as exc:
        print(f"Jeton d'un autre utilisateur → refusé : {exc}")


if __name__ == "__main__":
    main()
