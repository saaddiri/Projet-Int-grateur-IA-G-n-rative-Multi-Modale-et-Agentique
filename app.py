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
        ["Agent LangChain (outils)", "RAG simple"],
        help="L'agent choisit lui-même les outils de recherche (texte, tableaux, figures).",
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


def main() -> None:
    st.title("🛠️ MaintiDoc — Assistant documentaire multimodal")
    st.caption(
        "Étape 1 : RAG agentique multimodal (LlamaIndex · PGVector · CLIP · Qwen/Ollama · LangChain). "
        "Les réponses sont fondées sur les documents indexés et ne remplacent pas un technicien habilité."
    )
    mode = sidebar()

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
