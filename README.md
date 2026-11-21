# MaintiDoc — Assistant documentaire multimodal et agentique

MaintiDoc est un assistant destiné aux techniciens de maintenance : il exploite des
manuels, procédures, schémas et photographies pour répondre à des questions techniques
avec des **références vérifiables** (document, page, type de contenu).

> Le système ne pilote aucun équipement et ne remplace pas un technicien habilité.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| **1** | RAG agentique multimodal (LlamaIndex, PGVector, CLIP, Qwen/Ollama, LangChain, Streamlit) | ✅ livrée |
| 2 | Fine-tuning (reranker adapté), VLM, sorties structurées | à venir |
| 3 | Orchestration LangGraph, serveur/client MCP, tickets avec validation humaine | à venir |
| 4 | Client Web, bot Telegram/Discord, sécurité, déploiement | à venir |

## Étape 1 — RAG agentique multimodal

### Architecture

```
                 ┌──────────────────── Indexation ────────────────────┐
  PDF ──► LlamaIndex (PyMuPDFReader + SentenceSplitter) ──► passages texte ─┐
      ├─► PyMuPDF find_tables() ──► tableaux en Markdown ───────────────────┤
      └─► images extraites ──► CLIP ─────────────────► PGVector (images, 512d)
                            └─► VLM Qwen2.5-VL ──► descriptions ────────────┤
                                                                            ▼
                                        Embeddings multilingues ──► PGVector (texte)

                 ┌───────────────────── Requête ──────────────────────┐
  Streamlit ──► Agent LangChain (Qwen2.5 via Ollama)
                   ├─ search_text    ──► PGVector (passages)
                   ├─ search_tables  ──► PGVector (tableaux)
                   └─ search_figures ──► CLIP + descriptions VLM
                ──► réponse citée [1], [2]… + sources (page, image)
```

| Brique | Choix | Rôle |
|---|---|---|
| Extraction | LlamaIndex `PyMuPDFReader`, PyMuPDF | texte par page, tableaux, images + légendes |
| Embeddings texte | `intfloat/multilingual-e5-small` (open source, multilingue ; `BAAI/bge-m3` possible via `.env`) | questions en français sur un document en anglais |
| Images | `openai/clip-vit-base-patch32` | recherche texte → image |
| Descriptions d'images | `qwen2.5vl:3b` via Ollama | rendre les figures interrogeables en texte |
| Base vectorielle | PostgreSQL 16 + pgvector (`PGVectorStore` de LlamaIndex) | vecteurs + métadonnées (source, page, type, légende) |
| LLM | `qwen2.5:3b` via Ollama | génération et appel d'outils |
| Agent | LangChain `create_agent` | choix des outils, gestion du contexte (historique glissant) |
| Interface | Streamlit | chargement du PDF, questions, réponses et sources |

### Structure du dépôt

```
maintidoc/
  config.py       configuration (.env)
  extraction.py   extraction texte / tableaux / images (LlamaIndex + PyMuPDF)
  images.py       encodeur CLIP et descriptions VLM
  store.py        tables PGVector (LlamaIndex PGVectorStore)
  indexing.py     pipeline d'indexation PDF -> PGVector
  retrieval.py    recherche texte, tableaux, images (fusion CLIP + descriptions)
  rag.py          pipeline RAG simple (référence et repli)
  agent.py        agent LangChain avec outils de recherche
app.py            interface Streamlit
scripts/          ingestion et questions en ligne de commande
tests/            tests d'extraction
data/             PDF de démonstration (« Attention Is All You Need »)
docs/             note technique
```

### Installation

Prérequis : Python 3.11+, Docker, [Ollama](https://ollama.com).

```bash
# 1. Base PGVector
docker compose up -d

# 2. Modèles Ollama
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b

# 3. Environnement Python
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS : source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# 4. Configuration
copy .env.example .env            # Linux/macOS : cp .env.example .env
```

### Utilisation

```bash
# Indexer le PDF de démonstration
python -m scripts.ingest data/attention_is_all_you_need.pdf

# Poser des questions en ligne de commande
python -m scripts.ask "Que montre la figure 1 ?" "Quel score BLEU obtient le Transformer (big) en anglais-allemand ?"

# Interface web
streamlit run app.py
```

Dans l'interface : charger un PDF dans la barre latérale, cliquer sur **Indexer le document**,
puis poser des questions. Chaque réponse affiche les outils appelés par l'agent et les sources
(page, extrait, image le cas échéant).

### Tests

```bash
pytest -q
```

Voir la [note technique](docs/NOTE_TECHNIQUE.md) pour les choix de modèles, les limites et des
exemples de questions/réponses.
