"""Interface Streamlit de MaintiDoc (étape 1).

Lancement :  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from maintidoc.config import ROOT_DIR, settings

st.set_page_config(page_title="MaintiDoc — RAG multimodal", page_icon="🛠️", layout="wide")

UPLOAD_DIR = ROOT_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


@st.cache_resource(show_spinner="Chargement des modèles (embeddings, CLIP, LLM)…")
def load_agent():
    from maintidoc.agent import MaintiDocAgent

    return MaintiDocAgent()


def sidebar() -> str:
    from maintidoc.indexing import index_pdf
    from maintidoc.store import list_documents

    st.sidebar.header("📄 Documents")
    uploaded = st.sidebar.file_uploader("Charger un PDF", type=["pdf"])
    if uploaded is not None and st.sidebar.button("Indexer le document", type="primary"):
        target = UPLOAD_DIR / Path(uploaded.name).name
        target.write_bytes(uploaded.getbuffer())
        with st.sidebar.status("Indexation en cours…", expanded=True) as status:
            try:
                summary = index_pdf(target, progress=status.write)
                status.update(label=f"✅ {summary['source']} indexé", state="complete")
            except Exception as exc:
                status.update(label="❌ Échec de l'indexation", state="error")
                st.sidebar.error(str(exc))

    docs = list_documents()
    if docs:
        st.sidebar.caption("Index PGVector")
        st.sidebar.dataframe(
            [
                {
                    "Document": d["source"],
                    "Passages": d.get("text", 0),
                    "Tableaux": d.get("table", 0),
                    "Images": d.get("image_description", 0),
                }
                for d in docs
            ],
            hide_index=True,
        )
    else:
        st.sidebar.info("Aucun document indexé pour l'instant.")

    st.sidebar.divider()
    mode = st.sidebar.radio(
        "Mode de réponse",
        ["Multimodal V2 (texte + photo)", "Agent LangChain (outils)", "RAG simple"],
        help="V2 : analyse de photo par VLM, fusion des recherches, reranking et réponse structurée. "
        "Agent : le LLM choisit lui-même les outils de recherche.",
    )
    st.session_state["reranker_choice"] = st.sidebar.selectbox(
        "Reranker (mode V2)", ["default", "adapted", "base", "none"],
        help="default = reranker adapté s'il existe, sinon le reranker initial.",
    )
    if st.sidebar.button("Nouvelle conversation"):
        st.session_state.messages = []
        load_agent().reset()
    st.sidebar.caption(f"LLM : `{settings.llm_model}` · VLM : `{settings.vlm_model}`")
    st.sidebar.caption(f"Embeddings : `{settings.text_embed_model}` · CLIP : `{settings.clip_model}`")
    return mode


def render_sources(answer) -> None:
    if answer.get("tool_calls"):
        with st.expander("🔧 Outils appelés par l'agent"):
            for call in answer["tool_calls"]:
                st.code(call, language="text")
    sources = answer.get("sources") or []
    if sources:
        with st.expander(f"📚 Sources ({len(sources)})"):
            for i, hit in enumerate(sources, start=1):
                st.markdown(f"**[{i}] {hit.reference}** — score {hit.score:.3f}")
                if hit.image_path and Path(hit.image_path).exists():
                    st.image(hit.image_path, caption=hit.caption or None, width=360)
                st.text(hit.text[:1200])


UNCERTAINTY_ICON = {"faible": "🟢", "moyenne": "🟠", "élevée": "🔴"}


def render_v2() -> None:
    """Mode V2 : question + photo -> réponse structurée (Pydantic)."""
    from PIL import Image

    from maintidoc.multimodal import answer as multimodal_answer
    from maintidoc.reranker import resolve

    with st.form("v2"):
        photo_file = st.file_uploader("Photo de l'équipement (optionnelle)", type=["png", "jpg", "jpeg", "webp"])
        question = st.text_area("Question", placeholder="Ex. : Que représente ce schéma ? À quoi sert ce bloc ?")
        submitted = st.form_submit_button("Analyser", type="primary")
    if not submitted:
        return
    if not question.strip() and photo_file is None:
        st.warning("Saisissez une question ou ajoutez une photo.")
        return

    photo = Image.open(photo_file) if photo_file is not None else None
    choice = st.session_state.get("reranker_choice", "default")
    reranker = "default" if choice == "default" else resolve(choice)
    with st.spinner("Analyse de la photo, recherche, reranking et génération…"):
        try:
            result, trace = multimodal_answer(question, photo, reranker=reranker)
        except Exception as exc:
            st.error(f"Erreur : {exc}. Vérifiez que PostgreSQL (docker compose) et Ollama sont démarrés.")
            return

    cols = st.columns([1, 2]) if photo is not None else [None, st.container()]
    if photo is not None:
        cols[0].image(photo, caption="Photo envoyée", use_container_width=True)
    with cols[1]:
        if result.needs_clarification:
            st.warning(f"**Clarification demandée :** {result.clarification_question}")
        st.markdown(result.answer)
        st.caption(f"{UNCERTAINTY_ICON[result.uncertainty]} Incertitude {result.uncertainty} — {result.uncertainty_reason}")

    if result.photo_analysis is not None:
        with st.expander("🖼️ Analyse de la photo (VLM)"):
            st.json(result.photo_analysis.model_dump())
    with st.expander(f"📚 Citations ({len(result.citations)})", expanded=True):
        for c in result.citations:
            note = " — attribution automatique (reranker)" if c.attribution == "automatique" else ""
            st.markdown(f"**[{c.ref}] {c.source}, p. {c.page}** ({c.content_type}){note}")
            if c.image_path and Path(c.image_path).exists():
                st.image(c.image_path, width=320)
            st.text(c.excerpt)
    with st.expander("🔎 Trace de recherche (fusion + reranking)"):
        st.write({"requêtes": trace.queries, "listes fusionnées": trace.lists, "reranker": trace.reranker or "aucun",
                  "score de pertinence (meilleur passage)": result.relevance_score})
        for hit, score in trace.ranked:
            st.markdown(f"- {hit.reference} — score reranker {score:.3f}")
    with st.expander("{ } Sortie structurée (JSON)"):
        st.json(result.model_dump(exclude={"photo_analysis"}))


def main() -> None:
    st.title("🛠️ MaintiDoc — Assistant documentaire multimodal")
    st.caption(
        "V2 : RAG multimodal (LlamaIndex · PGVector · CLIP multilingue · Qwen2.5-VL · reranker adapté · LangChain). "
        "Les réponses sont fondées sur les documents indexés et ne remplacent pas un technicien habilité."
    )
    mode = sidebar()
    if mode.startswith("Multimodal"):
        render_v2()
        return

    st.session_state.setdefault("messages", [])
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                render_sources(msg)

    question = st.chat_input("Posez une question sur le document…")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Recherche et génération…"):
            try:
                if mode.startswith("Agent"):
                    answer = load_agent().ask(question)
                else:
                    from maintidoc.rag import answer_with_rag

                    answer = answer_with_rag(question)
            except Exception as exc:
                st.error(f"Erreur : {exc}. Vérifiez que PostgreSQL (docker compose) et Ollama sont démarrés.")
                return
        msg = {
            "role": "assistant",
            "content": answer.text,
            "sources": answer.sources,
            "tool_calls": answer.tool_calls,
        }
        st.markdown(answer.text)
        render_sources(msg)
    st.session_state.messages.append(msg)


main()
