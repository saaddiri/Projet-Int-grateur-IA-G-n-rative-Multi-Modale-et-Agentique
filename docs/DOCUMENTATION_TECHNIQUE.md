# Documentation technique — MaintiDoc (L4.6)

Vue d'ensemble de l'application intégrée (étapes 1 à 4). Détails par étape :
[NOTE_TECHNIQUE.md](NOTE_TECHNIQUE.md) (RAG), [RAPPORT_RERANKER.md](RAPPORT_RERANKER.md)
(multimodalité, reranker), [WORKFLOW_ETAPE3.md](WORKFLOW_ETAPE3.md) (LangGraph, MCP).

## 1. Architecture

```
 Telegram ──► bot (python-telegram-bot) ──┐  X-Bot-Secret + identifiant Telegram
 Navigateur ─► client Web Streamlit ──────┤  Authorization: Bearer <clé API> (connexion)
 Scripts / autres clients ────────────────┤  Authorization: Bearer <clé API>
                                          ▼
                         Backend FastAPI (maintidoc/api.py)
   identités · filtrage des documents · limites d'appels · journal · validation
               │                       │                          │
     RAG multimodal V2          Graphe LangGraph            Client MCP
  (étapes 1-2 : PGVector,   (étape 3 : brouillon,      (clé de service +
   CLIP, VLM, reranker)      interruption, reprise)     utilisateur délégué)
               │                       │ checkpoints SQLite       │ stdio
               ▼                       ▼                          ▼
      PostgreSQL + pgvector      storage/checkpoints.db   Serveur MCP maintenance
      Ollama (LLM, VLM)                                   (droits, jetons, idempotence,
                                                           audit) → storage/maintenance.db
```

## 2. Modules

| Module | Rôle |
|---|---|
| `maintidoc/api.py` | backend FastAPI : routes, authentification, limites, journal |
| `maintidoc/access.py` | filtrage des documents selon le rôle et le site (règles `data/documents_acl.json`) |
| `app.py` | client Web Streamlit : connexion par clé API, toutes les actions via le backend |
| `maintidoc/identity.py` | utilisateurs (référentiel), liaison des comptes Telegram, brouillons en attente |
| `maintidoc/bot/commands.py` | logique des commandes (testable sans Telegram) |
| `maintidoc/bot/telegram_bot.py` | adaptateur Telegram (polling, photos, découpage des messages) |
| `maintidoc/workflow.py` | graphe LangGraph (étape 3) |
| `maintidoc/mcp_server.py` / `mcp_client.py` | service de maintenance simulé et son client |
| `maintidoc/approval.py` | jetons d'approbation HMAC |
| `maintidoc/multimodal.py`, `vision.py`, `reranker.py` | pipeline V2 (étape 2) |
| `maintidoc/versioning.py` | empreintes des prompts, modèles, `/version` |

## 3. API (OpenAPI complète sur `/docs`)

| Méthode | Route | Authentification | Limite LLM | Description |
|---|---|---|---|---|
| GET | `/health`, `/version` | aucune | — | état, versions |
| POST | `/link` | secret du bot | 5 essais/min | associe un compte Telegram avec un code |
| GET | `/me` | oui | — | identité et rôle |
| POST | `/ask` | oui | oui | réponse structurée sourcée (pipeline V2) |
| POST | `/rag` | oui | oui | RAG simple de l'étape 1 |
| POST | `/agent`, `/agent/reset` | oui | oui | agent LangChain de l'étape 1 (historique par utilisateur) |
| GET | `/documents` | oui | — | documents que l'utilisateur peut consulter |
| POST | `/documents` | superviseur | — | ajout d'un PDF avec sa règle d'accès, puis indexation |
| POST | `/analyze` | oui | oui | photo (≤ 8 Mo, image vérifiée) + question |
| GET | `/equipment/{id}/history` | oui | — | historique (MCP) |
| GET | `/equipment/{id}/tickets` | oui | — | tickets ouverts (MCP) |
| POST | `/tickets/draft` | oui | oui | brouillon ; renvoie `thread_id` et `content_sha256` |
| POST | `/tickets/approve` | oui | — | approuve le contenu dont l'empreinte est fournie |
| POST | `/tickets/cancel` | oui | — | annule le brouillon en attente |

## 4. Sécurité

| Menace | Mesure | Vérifiée par |
|---|---|---|
| Usurpation d'un compte Telegram | secret partagé bot/backend ; liaison par code à usage unique (10 min, haché, 5 essais/min) | tests, évaluation |
| Accès à un équipement d'un autre site | contrôle par le serveur MCP sur l'utilisateur délégué | matrice d'accès (évaluation) |
| Création sans approbation | aucune route ni outil LLM ne crée directement ; reprise LangGraph explicite ; jeton HMAC | tests, évaluation |
| Approbation d'un autre contenu | empreinte du contenu affiché exigée ; jeton lié au contenu exact | tests, évaluation |
| Approbation du brouillon d'autrui | propriétaire vérifié ; réponse 404 identique (pas de divulgation) | tests, évaluation |
| Doublons (double clic, rejeu) | `approval_id` unique + contenu identique ouvert ; verrou d'état « en_cours » | tests, évaluation |
| Injection de prompt (question, historique, documents) | le LLM n'a ni outil d'action ni secret dans son contexte ; contenus présentés comme données ; contrôles hors LLM | évaluation (15 injections) |
| Délégation forgée | clé de service comparée en temps constant par le serveur MCP | évaluation |
| Accès à un document non autorisé | filtre `source IN (documents autorisés)` ajouté à chaque requête PGVector (texte, tableaux, figures, CLIP) ; document sans règle réservé aux superviseurs ; règle enregistrée avant indexation | tests, évaluation (19 scénarios) |
| Contournement par l'interface Web | Streamlit sans logique métier : tout passe par le backend authentifié | revue, test manuel |
| Base redémarrée (connexions mortes) | `pool_pre_ping` sur les connexions PostgreSQL | incident observé et corrigé |
| Abus / surcharge | 20 requêtes/min et 4 requêtes LLM/min par utilisateur ; tailles maximales | tests, évaluation |
| Fuite de secrets | `.env` hors dépôt et hors image ; jetons jamais journalisés ; logs `httpx` du bot réduits (URL contenant le jeton) | revue |
| Exposition réseau | ports liés à 127.0.0.1 ; utilisateur non privilégié dans l'image | revue |

Journalisation minimale : backend (méthode, route, utilisateur, statut, durée), serveur MCP
(table `audit` : utilisateur, outil, équipement, résultat). Ni questions, ni photos, ni jetons.

## 5. Tests

| Fichier | Contenu |
|---|---|
| `tests/test_extraction.py` | extraction texte / tableaux / images |
| `tests/test_v2.py` | fusion RRF, reranking, sorties structurées |
| `tests/test_etape3.py` | jetons, serveur MCP, transitions LangGraph |
| `tests/test_etape4.py` | intégration bot → API → LangGraph → MCP : identités, accès, approbation, injection, limites, versions, filtrage des documents |

## 6. Limites connues

- CPU uniquement : 1 à 7 minutes par requête avec IA ; un GPU ou un modèle quantifié plus
  petit est nécessaire pour un usage réel à plusieurs.
- Corpus de démonstration d'un seul document, sans rapport avec les équipements simulés.
- Filtrage des documents : par document entier (pas par page ni par passage).
- Limites d'appels en mémoire (une seule instance du backend).
- Le serveur MCP est relancé à chaque appel (≈ 2,5 s).
- Bot : démarré avec un vrai bot Telegram, liaison de compte vérifiée ; les autres commandes
  restent à tester manuellement depuis Telegram (leur logique est testée jusqu'au backend).
