# Guide d'installation et de déploiement (L4.3)

## 1. Prérequis

| Élément | Version testée | Remarque |
|---|---|---|
| Docker + Docker Compose | Docker Desktop (Compose v2) | base PGVector, backend, interface, bot |
| Ollama | sur l'hôte, ou conteneur `--profile ollama` | modèles `qwen2.5:3b` (LLM) et `qwen2.5vl:3b` (VLM) |
| Machine | 4 cœurs, 16 Go de RAM, sans GPU | latences mesurées dans [RAPPORT_EVALUATION.md](RAPPORT_EVALUATION.md) |
| Python (hors Docker) | 3.12 | développement, tests, évaluation |

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5vl:3b
```

## 2. Configuration

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(32))"   # à faire 3 fois
```

Renseigner dans `.env` : `APPROVAL_SECRET`, `MCP_SERVICE_KEY`, `BOT_SHARED_SECRET` (valeurs
générées ci-dessus) et, pour Telegram, `TELEGRAM_BOT_TOKEN` (créé avec @BotFather).
`.env` n'est jamais versionné et n'entre pas dans l'image Docker (liste blanche `.dockerignore`).

## 3. Déploiement avec Docker Compose

```bash
docker compose up -d --build                 # pgvector + api (8000) + web (8501)
docker compose --profile telegram up -d      # + bot Telegram
docker compose --profile ollama up -d        # + Ollama en conteneur (mettre OLLAMA_BASE_URL_DOCKER=http://ollama:11434)
```

| Service | Rôle | Port (lié à 127.0.0.1) |
|---|---|---|
| `pgvector` | index vectoriel (PostgreSQL 16 + pgvector) | 5433 |
| `api` | backend FastAPI (identités, RAG, workflow, client MCP) | 8000 (`/docs` : documentation OpenAPI) |
| `web` | interface Streamlit | 8501 |
| `bot` | bot Telegram (profil `telegram`) | — (connexion sortante vers Telegram) |
| `ollama` | LLM / VLM (profil `ollama`, facultatif) | 11434 |

Le serveur MCP n'est pas un service séparé : le backend le lance en sous-processus (stdio) à
chaque appel, dans le même conteneur, avec sa propre base `storage/maintenance.db`.

Volumes : `pgdata` (index), `storage` (bases SQLite : maintenance, identités, checkpoints
LangGraph), `uploads`, `hf-cache` (modèles Hugging Face, téléchargés au premier usage),
`./models` monté en lecture seule (reranker adapté).

## 4. Premier démarrage

```bash
# indexer le document de démonstration (dans le conteneur ou en local)
docker compose exec api python -m scripts.ingest data/attention_is_all_you_need.pdf
# reranker adapté : poids non versionnés, à régénérer (≈ 20 min CPU) ou RERANKER=base
python -m scripts.train_reranker
# associer un compte Telegram à un utilisateur (code à usage unique, 10 min)
docker compose exec api python -m scripts.link_account tech.dupont
#   puis, dans Telegram :  /start <code>
```

Vérifications : `curl http://localhost:8000/health` puis `curl http://localhost:8000/version`
(versions des prompts, modèles et paquets).

## 5. Installation locale (sans Docker)

```bash
python -m venv .venv && .venv\Scripts\activate           # Windows
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
docker compose up -d pgvector
python -m scripts.ingest data/attention_is_all_you_need.pdf
uvicorn maintidoc.api:app --port 8000                     # backend
python -m maintidoc.bot.telegram_bot                      # bot
streamlit run app.py                                      # interface
pytest                                                    # tests
python -m scripts.evaluate --quick                        # évaluation sans LLM (≈ 5 min)
```

## 6. Versionnage

- Dépendances : versions figées dans `requirements.txt` (et PyTorch dans le `Dockerfile`).
- Prompts et modèles : `versions.lock.json` ; un test échoue si un prompt ou un modèle change
  sans régénération (`python -m scripts.lock_versions`) ; exposés par `/version`.
- Reranker adapté : fiche `models/reranker-maintidoc-v1.json` (version, hyperparamètres,
  empreinte du jeu d'entraînement).
- Image Docker : `maintidoc:4.0.0`.
