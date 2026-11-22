"""Client Web Streamlit de MaintiDoc.

L'interface ne contient aucune logique métier : chaque action appelle le backend
FastAPI (maintidoc/api.py) avec la clé API de l'utilisateur connecté. Identités,
droits, filtrage des documents, limites d'appels et approbations sont donc
appliqués côté serveur, exactement comme pour le bot Telegram.

Lancement :  uvicorn maintidoc.api:app --port 8000   puis   streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import httpx
import streamlit as st

from maintidoc.config import settings

st.set_page_config(page_title="MaintiDoc", page_icon="🛠️", layout="wide")

UNCERTAINTY_ICON = {"faible": "🟢", "moyenne": "🟠", "élevée": "🔴"}
TIMEOUT = 900  # les requêtes avec IA prennent plusieurs minutes sur CPU


# --------------------------------------------------------------------------- accès au backend

def api(method: str, path: str, **kwargs) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {st.session_state.get('api_key', '')}"}
    try:
        response = httpx.request(method, f"{settings.api_base_url}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    except httpx.HTTPError as exc:
        return 0, {"detail": f"Backend injoignable ({settings.api_base_url}) : {exc}"}
    try:
        return response.status_code, response.json()
    except ValueError:
        return response.status_code, {"detail": response.text}


def show_error(status: int, data: dict) -> None:
    detail = data.get("detail", "erreur")
    if isinstance(detail, list):
        detail = "; ".join(d.get("msg", "") for d in detail)
    st.error(f"{detail}" + (f" (code {status})" if status else ""))


# --------------------------------------------------------------------------- connexion

def login() -> dict | None:
    if st.session_state.get("me"):
        return st.session_state["me"]
    st.title("🛠️ MaintiDoc")
    st.caption("Assistant documentaire de maintenance — connexion requise.")
    with st.form("login"):
        key = st.text_input("Clé API", type="password",
                            help="Comptes de démonstration : demo-key-dupont, demo-key-martin, demo-key-leroy")
        if st.form_submit_button("Se connecter", type="primary"):
            st.session_state["api_key"] = key
            status, data = api("GET", "/me")
            if status == 200:
                st.session_state["me"] = data
                st.rerun()
            show_error(status, data)
    return None


def sidebar(me: dict) -> str:
    st.sidebar.markdown(f"**{me['name']}**  \n{me['role']} · {', '.join(me['sites'])}")
    if st.sidebar.button("Se déconnecter"):
        st.session_state.clear()
        st.rerun()

    st.sidebar.header("📄 Documents autorisés")
    status, data = api("GET", "/documents")
    if status == 200 and data["documents"]:
        st.sidebar.dataframe(
            [{"Document": d["source"], "Texte": d.get("text", 0), "Tableaux": d.get("table", 0),
              "Figures": d.get("image_description", 0)} for d in data["documents"]],
            hide_index=True, use_container_width=True)
    elif status == 200:
        st.sidebar.info("Aucun document accessible.")
    if me["role"] == "superviseur":
        with st.sidebar.expander("Ajouter un PDF"):
            with st.form("upload"):
                pdf = st.file_uploader("PDF", type=["pdf"])
                roles = st.multiselect("Rôles autorisés", ["technicien", "superviseur", "lecteur"],
                                       default=["technicien", "superviseur"])
                sites = st.text_input("Sites (séparés par des virgules, * = tous)", "*")
                if st.form_submit_button("Indexer") and pdf is not None:
                    with st.spinner("Indexation (descriptions d'images par le VLM : plusieurs minutes)…"):
                        status, result = api("POST", "/documents", files={"pdf": (pdf.name, pdf.getvalue(), "application/pdf")},
                                             data={"roles": ",".join(roles), "sites": sites})
                    st.success(f"Indexé : {result['indexed']}") if status == 200 else show_error(status, result)

    st.sidebar.divider()
    return st.sidebar.radio("Mode", ["Assistant (texte + photo)", "Workflow ticket (LangGraph + MCP)",
                                     "Historique et tickets", "Agent LangChain (outils)", "RAG simple"])


# --------------------------------------------------------------------------- modes

def render_citations(citations: list[dict]) -> None:
    with st.expander(f"📚 Citations ({len(citations)})", expanded=True):
        for c in citations:
            note = " — attribution automatique (reranker)" if c.get("attribution") == "automatique" else ""
            st.markdown(f"**[{c['ref']}] {c['source']}, p. {c['page']}** ({c['content_type']}){note}")
            if c.get("image_path") and Path(c["image_path"]).exists():
                st.image(c["image_path"], width=320)
            st.text(c["excerpt"])


def render_assistant() -> None:
    with st.form("assistant"):
        photo = st.file_uploader("Photo de l'équipement (optionnelle)", type=["png", "jpg", "jpeg", "webp"])
        question = st.text_area("Question", placeholder="Ex. : Que faire en cas de vibrations au démarrage du CMP-101 ?")
        submitted = st.form_submit_button("Analyser", type="primary")
    if not submitted or (not question.strip() and photo is None):
        return
    with st.spinner("Recherche, reranking et génération (1 à 7 min sur CPU)…"):
        if photo is not None:
            status, data = api("POST", "/analyze", files={"photo": (photo.name, photo.getvalue(), photo.type)},
                               data={"question": question})
        else:
            status, data = api("POST", "/ask", json={"question": question})
    if status != 200:
        show_error(status, data)
        return
    cols = st.columns([1, 2]) if photo is not None else [None, st.container()]
    if photo is not None:
        cols[0].image(photo, caption="Photo envoyée", use_container_width=True)
    with cols[1]:
        if data["needs_clarification"]:
            st.warning(f"**Clarification demandée :** {data['clarification_question']}")
        st.markdown(data["answer"])
        st.caption(f"{UNCERTAINTY_ICON[data['uncertainty']]} Incertitude {data['uncertainty']} — {data['uncertainty_reason']}")
    if data.get("photo_analysis"):
        with st.expander("🖼️ Analyse de la photo (VLM)"):
            st.json(data["photo_analysis"])
    render_citations(data["citations"])
    with st.expander("{ } Réponse structurée (JSON)"):
        st.json(data)


def render_workflow() -> None:
    st.caption("Le ticket n'est créé qu'après votre approbation explicite du contenu affiché.")
    with st.form("draft"):
        c1, c2 = st.columns([1, 3])
        equipment = c1.text_input("Équipement", "CMP-101")
        problem = c2.text_input("Problème constaté", placeholder="vibrations au démarrage depuis ce matin")
        submitted = st.form_submit_button("Préparer un brouillon", type="primary")
    if submitted and problem.strip():
        with st.spinner("Historique (MCP), documentation, rédaction du brouillon…"):
            status, data = api("POST", "/tickets/draft", json={"equipment_id": equipment.strip().upper(), "problem": problem})
        if status != 200:
            show_error(status, data)
        else:
            st.session_state["draft"] = data

    draft = st.session_state.get("draft")
    if not draft:
        return
    d = draft["draft"]
    st.subheader(f"📝 Brouillon — {d['equipment_id']}")
    for t in draft.get("similar_tickets", []):
        st.warning(f"Ticket ouvert similaire : **{t['ticket_id']}** — {t['title']}")
    st.markdown(f"**{d['title']}**")
    st.text(d["description"])
    st.caption(f"Code du contenu : `{draft['content_sha256'][:8]}` — expire dans 15 min")
    approve, cancel = st.columns(2)
    if approve.button("✅ Approuver ce contenu exact", type="primary"):
        status, data = api("POST", "/tickets/approve", json={"thread_id": draft["thread_id"],
                                                            "content_sha256": draft["content_sha256"]})
        if status == 200:
            t = data["ticket"]
            st.success(f"Ticket {t['ticket_id']} {'créé' if data['created'] else 'déjà existant (aucun doublon)'}.")
            st.session_state.pop("draft")
        else:
            show_error(status, data)
    if cancel.button("✖️ Annuler"):
        status, data = api("POST", "/tickets/cancel", json={"thread_id": draft["thread_id"]})
        st.info("Brouillon annulé : aucun ticket créé.") if status == 200 else show_error(status, data)
        st.session_state.pop("draft", None)
    with st.expander("🧭 Étapes du graphe LangGraph"):
        for step in draft.get("steps", []):
            st.markdown(f"- {step}")


def render_history() -> None:
    equipment = st.text_input("Équipement", "CMP-101").strip().upper()
    if not st.button("Consulter", type="primary"):
        return
    c1, c2 = st.columns(2)
    status, data = api("GET", f"/equipment/{equipment}/history")
    with c1:
        st.subheader("🗂️ Historique")
        if status != 200:
            show_error(status, data)
        else:
            st.caption(data["equipment"]["name"])
            for h in data["interventions"]:
                st.markdown(f"- **{h['date']}** ({h['type']}) — {h['summary']}")
    status, data = api("GET", f"/equipment/{equipment}/tickets")
    with c2:
        st.subheader("🎫 Tickets ouverts")
        if status != 200:
            show_error(status, data)
        else:
            for t in data["tickets"] or []:
                st.markdown(f"- **{t['ticket_id']}** — {t['title']} ({t['created_at'][:10]})")
            if not data["tickets"]:
                st.info("Aucun ticket ouvert.")


def render_chat(path: str) -> None:
    key = f"messages{path}"
    st.session_state.setdefault(key, [])
    if path == "/agent" and st.button("Nouvelle conversation"):
        api("POST", "/agent/reset")
        st.session_state[key] = []
    for msg in st.session_state[key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            for s in msg.get("sources", []):
                st.caption(f"{s['source']}, p. {s['page']} ({s['content_type']})")
    question = st.chat_input("Posez une question sur la documentation…")
    if not question:
        return
    st.session_state[key].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération…"):
            status, data = api("POST", path, json={"question": question})
        if status != 200:
            show_error(status, data)
            return
        st.markdown(data["answer"])
        if data.get("tool_calls"):
            with st.expander("🔧 Outils appelés par l'agent"):
                for call in data["tool_calls"]:
                    st.code(call, language="text")
        with st.expander(f"📚 Sources ({len(data['sources'])})"):
            for s in data["sources"]:
                st.markdown(f"**{s['source']}, p. {s['page']}** ({s['content_type']}) — score {s['score']}")
                if s.get("image_path") and Path(s["image_path"]).exists():
                    st.image(s["image_path"], width=320)
                st.text(s["excerpt"])
    st.session_state[key].append({"role": "assistant", "content": data["answer"], "sources": data["sources"]})


def main() -> None:
    me = login()
    if me is None:
        return
    mode = sidebar(me)
    st.title("🛠️ MaintiDoc — Assistant documentaire multimodal")
    st.caption("Réponses fondées sur les documents autorisés pour votre compte. MaintiDoc ne pilote aucun "
               "équipement et ne remplace pas un technicien habilité.")
    if mode.startswith("Assistant"):
        render_assistant()
    elif mode.startswith("Workflow"):
        render_workflow()
    elif mode.startswith("Historique"):
        render_history()
    elif mode.startswith("Agent"):
        render_chat("/agent")
    else:
        render_chat("/rag")


main()
