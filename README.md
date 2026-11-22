# MaintiDoc — Assistant documentaire multimodal et agentique

MaintiDoc est un assistant destiné aux techniciens de maintenance : il exploite des
manuels, procédures, schémas et photographies pour répondre à des questions techniques
avec des **références vérifiables** (document, page, type de contenu).

> Le système ne pilote aucun équipement et ne remplace pas un technicien habilité.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| **1** | RAG agentique multimodal (LlamaIndex, PGVector, CLIP, Qwen/Ollama, LangChain, Streamlit) | ✅ livrée |
| **2** | Multimodalité (photo + VLM, CLIP multilingue), fusion + reranking, reranker adapté, sorties Pydantic | ✅ livrée |
| **3** | Orchestration LangGraph, serveur/client MCP, tickets avec validation humaine | ✅ livrée |
| **4** | Client Web, bot Telegram, sécurité, déploiement Docker Compose, évaluation | ✅ livrée |

## Étape 4 — Bot Telegram, sécurité et déploiement

Un technicien utilise MaintiDoc depuis Telegram : question, photo, historique, tickets,
approbation et annulation. Le bot passe par un backend FastAPI qui authentifie le compte,
applique les limites d'appels et agit pour le compte de l'utilisateur auprès du serveur MCP,
lequel réapplique ses propres contrôles.

| Livrable | Réalisation |
|---|---|
| L4.1 Bot Telegram | [`maintidoc/bot/`](maintidoc/bot/) (python-telegram-bot) : `/start /ask /analyze /history /tickets /create_ticket /approve /cancel` |
| L4.2 Backend et identités | [`maintidoc/api.py`](maintidoc/api.py), [`maintidoc/identity.py`](maintidoc/identity.py) : clé API ou bot + liaison par code à usage unique, délégation MCP, limites, journal minimal |
| L4.3 Docker Compose et installation | [`docker-compose.yml`](docker-compose.yml), [`Dockerfile`](Dockerfile), [`docs/INSTALLATION.md`](docs/INSTALLATION.md) ; versions figées : `requirements.txt`, [`versions.lock.json`](versions.lock.json) (prompts, modèles) |
| L4.4 Tests et sécurité | [`tests/test_etape4.py`](tests/test_etape4.py) (intégration, injection, accès, approbation) |
| L4.5 Évaluation | [`scripts/evaluate.py`](scripts/evaluate.py) → [`docs/RAPPORT_EVALUATION.md`](docs/RAPPORT_EVALUATION.md) |
| L4.6 Documentation | [`docs/GUIDE_UTILISATEUR.md`](docs/GUIDE_UTILISATEUR.md), [`docs/DOCUMENTATION_TECHNIQUE.md`](docs/DOCUMENTATION_TECHNIQUE.md) |
| L4.7 Démonstration et soutenance | [`docs/SOUTENANCE.md`](docs/SOUTENANCE.md) |

```bash
cp .env.example .env                          # puis renseigner les secrets et TELEGRAM_BOT_TOKEN
docker compose --profile telegram up -d --build
docker compose exec api python -m scripts.link_account tech.dupont   # code pour /start <code>
pytest                                        # tous les tests
python -m scripts.evaluate                    # évaluation complète (≈ 1 h sur CPU)
```

## Étape 3 — Orchestration agentique et MCP

Un graphe LangGraph consulte le service de maintenance via MCP, interroge la documentation,
rédige un brouillon de ticket puis **s'interrompt** : le ticket n'est créé qu'après approbation
explicite, avec un jeton signé lié à l'utilisateur, à l'équipement et au contenu exact.
Détails, schéma du graphe et résultats : [`docs/WORKFLOW_ETAPE3.md`](docs/WORKFLOW_ETAPE3.md).

| Livrable | Réalisation |
|---|---|
| L3.1 Workflow LangGraph documenté | [`maintidoc/workflow.py`](maintidoc/workflow.py) + [`docs/WORKFLOW_ETAPE3.md`](docs/WORKFLOW_ETAPE3.md) |
| L3.2 Serveur MCP de démonstration | [`maintidoc/mcp_server.py`](maintidoc/mcp_server.py) — `get_equipment_history`, `get_open_tickets`, `create_ticket` ; données [`data/maintenance/seed.json`](data/maintenance/seed.json) |
| L3.3 Client MCP intégré | [`maintidoc/mcp_client.py`](maintidoc/mcp_client.py) (stdio, clé API de l'utilisateur) |
| L3.4 Brouillons de tickets | nœud `rediger_brouillon` (sortie Pydantic `TicketDraft`, références documentaires, tickets similaires) |
| L3.5 Validation humaine et contrôles d'accès | `interrupt()` LangGraph + [`maintidoc/approval.py`](maintidoc/approval.py) (jeton HMAC) + contrôles côté serveur (clé API, site, rôle, idempotence, audit) |
| L3.6 Démonstration de bout en bout | [`scripts/demo_workflow.py`](scripts/demo_workflow.py) et mode « Workflow ticket » de [`app.py`](app.py) |

```bash
python -m scripts.demo_workflow          # approbations demandées au clavier (--auto : simulées)
pytest tests/test_etape3.py              # contrôles d'accès, approbation, idempotence, transitions
streamlit run app.py                     # mode « Workflow ticket (LangGraph + MCP) » (backend requis)
```

## Étape 2 — Multimodalité et fine-tuning

```
 question (FR/EN) ─┐                       ┌─► passages / tableaux / descriptions (embeddings texte)
                   ├─► requêtes ───────────┼─► figures : CLIP multilingue (texte -> image)
 photo ─► VLM Qwen2.5-VL ─► PhotoAnalysis ─┘
   └──────────────────────────────────────────► figures : CLIP image -> image
                 ─► fusion RRF ─► reranker (initial ou adapté) ─► top 5
                 ─► LLM, sortie structurée Pydantic : réponse, citations, incertitude, clarification
```

| Livrable | Réalisation |
|---|---|
| L2.1 Module d'analyse d'images (VLM) | [`maintidoc/vision.py`](maintidoc/vision.py) — `PhotoAnalysis` (description, textes visibles, composants, voyants, requête, confiance) |
| L2.2 Pipeline RAG multimodal | [`maintidoc/multimodal.py`](maintidoc/multimodal.py) — recherches parallèles, fusion RRF, reranking, réponse `MaintiDocAnswer` |
| L2.3 Jeu de données d'adaptation | [`scripts/build_dataset.py`](scripts/build_dataset.py) → [`data/reranker/`](data/reranker/) (questions FR/EN générées puis vérifiées, négatifs difficiles) |
| L2.4 Reranker adapté et versionné | [`scripts/train_reranker.py`](scripts/train_reranker.py) → `models/reranker-maintidoc-v1/` + fiche [`models/reranker-maintidoc-v1.json`](models/reranker-maintidoc-v1.json) |
| L2.5 Rapport comparatif | [`docs/RAPPORT_RERANKER.md`](docs/RAPPORT_RERANKER.md) — nDCG@5, MRR@10 sans reranker / initial / adapté |
| L2.6 Application V2 texte + photo | mode « Assistant (texte + photo) » de [`app.py`](app.py) |

### Reproduire l'étape 2

```bash
# modèles supplémentaires (téléchargés automatiquement au premier usage) :
#   cross-encoder/mmarco-mMiniLMv2-L12-H384-v1, sentence-transformers/clip-ViT-B-32-multilingual-v1
python -m scripts.build_dataset --per-lang 4 --negatives 5   # jeu de données (≈ 45 min sur CPU)
python -m scripts.train_reranker                               # fine-tuning (≈ 20 min sur CPU)
python -m scripts.eval_reranker                                # métriques -> data/reranker/eval_test.json
streamlit run app.py                                           # mode « Assistant (texte + photo) » (backend requis)
```

## Étape 1 — RAG agentique multimodal

| Livrable | Réalisation |
|---|---|
| L1.1 Note de cadrage et architecture | [`docs/NOTE_CADRAGE.md`](docs/NOTE_CADRAGE.md) (dont le choix PGVector plutôt que Qdrant) |
| L1.2 Pipeline d'ingestion | [`maintidoc/extraction.py`](maintidoc/extraction.py), [`maintidoc/indexing.py`](maintidoc/indexing.py), [`scripts/ingest.py`](scripts/ingest.py) |
| L1.3 Index et métadonnées | PGVector (source, page, type, légende, droits d'accès par document : [`data/documents_acl.json`](data/documents_acl.json)) |
| L1.4 API et chaîne RAG | [`maintidoc/rag.py`](maintidoc/rag.py), [`maintidoc/agent.py`](maintidoc/agent.py), exposés par [`maintidoc/api.py`](maintidoc/api.py) (`/rag`, `/agent`, `/ask`) |
| L1.5 Interface Streamlit | [`app.py`](app.py) (client du backend, connexion par clé API) |
| L1.6 Rapport de tests documentaires et d'accès | [`docs/RAPPORT_TESTS_ETAPE1.md`](docs/RAPPORT_TESTS_ETAPE1.md) |

Corpus de démonstration : l'article « Attention Is All You Need » (recommandé par le sujet)
et un manuel de maintenance **fictif** du compresseur CMP-101
([`scripts/make_demo_manual.py`](scripts/make_demo_manual.py)), réservé aux techniciens et
superviseurs de l'atelier A, qui relie la documentation aux équipements simulés.

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
  Streamlit ──► FastAPI ──► Agent LangChain (Qwen2.5 via Ollama)
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
| Interface | Streamlit (via le backend FastAPI) | chargement du PDF (superviseurs), questions, réponses et sources |

### Structure du dépôt

```
maintidoc/
  config.py       configuration (.env)
  access.py       filtrage des documents selon le rôle et le site
  extraction.py   extraction texte / tableaux / images (LlamaIndex + PyMuPDF)
  images.py       encodeur CLIP et descriptions VLM
  store.py        tables PGVector (LlamaIndex PGVectorStore)
  indexing.py     pipeline d'indexation PDF -> PGVector
  retrieval.py    recherche texte, tableaux, images (fusion CLIP + descriptions)
  rag.py          pipeline RAG simple (référence et repli)
  agent.py        agent LangChain avec outils de recherche
  vision.py       analyse de photo par VLM, sortie Pydantic (étape 2)
  multimodal.py   pipeline V2 : fusion RRF, reranking, réponse structurée (étape 2)
  reranker.py     cross-encoder multilingue (initial / adapté) (étape 2)
  workflow.py     graphe LangGraph avec interruption pour approbation (étape 3)
  mcp_server.py   serveur MCP du service de maintenance simulé (étape 3)
  mcp_client.py   client MCP (étape 3)
  approval.py     jetons d'approbation liés au contenu et à l'utilisateur (étape 3)
  api.py          backend FastAPI : identités, limites, routes (étape 4)
  identity.py     comptes, liaison Telegram, brouillons en attente (étape 4)
  bot/            bot Telegram (étape 4)
  versioning.py   versions des prompts et modèles (étape 4)
app.py            client Web Streamlit (appelle le backend)
scripts/          ingestion, questions, jeu de données, entraînement, démonstrations, évaluation
tests/            tests d'extraction, du pipeline V2, des étapes 3 et 4
data/             PDF de démonstration, règles d'accès, données de maintenance fictives, résultats
docs/             cadrage, notes, rapports, guides, soutenance
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

# Backend puis interface web
uvicorn maintidoc.api:app --port 8000
streamlit run app.py
```

Dans l'interface : se connecter avec une clé API (comptes de démonstration
`demo-key-dupont`, `demo-key-martin`, `demo-key-leroy`), puis poser des questions (modes
« Agent LangChain », « RAG simple » ou « Assistant »). Un superviseur charge un PDF depuis la
barre latérale en choisissant qui peut le consulter. Chaque réponse affiche les outils appelés
par l'agent et les sources (page, extrait, image le cas échéant).

### Tests

```bash
pytest -q
```

Voir la [note technique](docs/NOTE_TECHNIQUE.md) pour les choix de modèles, les limites et des
exemples de questions/réponses.
