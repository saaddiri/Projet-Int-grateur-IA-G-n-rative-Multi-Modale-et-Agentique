# Démonstration et soutenance finale (L4.7)

## 1. Plan de présentation (≈ 15 min + démonstration)

| # | Partie | Contenu | Support |
|---|---|---|---|
| 1 | Problème | techniciens, manuels dispersés, photos de terrain, traçabilité des tickets | README |
| 2 | Architecture | schéma d'ensemble, choix open source (Qwen, CLIP, PGVector, LangGraph, MCP) | DOCUMENTATION_TECHNIQUE §1 |
| 3 | Étape 1 — RAG multimodal | texte, tableaux, figures ; citations | NOTE_TECHNIQUE |
| 4 | Étape 2 — multimodalité et reranker | VLM, CLIP multilingue, fusion, nDCG@5 0,611 → 0,846 | RAPPORT_RERANKER |
| 5 | Étape 3 — agentique et MCP | graphe, interruption, jeton lié au contenu, idempotence | WORKFLOW_ETAPE3 |
| 6 | Étape 4 — bot, sécurité, déploiement | identités, contrôles côté serveur, Docker Compose | INSTALLATION |
| 7 | Évaluation | scénarios, latences, qualité des citations, sécurité, ressources | RAPPORT_EVALUATION |
| 8 | Limites et suites | CPU, petit corpus, VLM 3B, GPU, corpus réel | RAPPORT_EVALUATION §6 |

## 2. Scénario de démonstration (critère de validation de l'étape 4)

Préparation : `docker compose --profile telegram up -d`, document indexé, compte Telegram lié
(`python -m scripts.link_account tech.dupont` puis `/start <code>`).

| # | Action dans Telegram | Attendu |
|---|---|---|
| 1 | `/start` | « Bonjour Technicien Dupont (technicien) » + aide |
| 2 | photo `data/photos_test/schema_photo.jpg`, légende « Que représente ce schéma ? » | réponse sourcée (figure 1, p. 3) |
| 3 | photo `photo_floue.jpg`, légende « Que signifie ce voyant ? » | demande de clarification, aucune réponse inventée |
| 4 | `/history CMP-101` | 3 interventions (dont une note piégée, sans effet) |
| 5 | `/tickets CMP-101` | T-0001 ouvert |
| 6 | `/create_ticket CMP-101 vibrations au démarrage depuis ce matin` | brouillon + T-0001 signalé comme similaire |
| 7 | `/approve 00000000` | refusé : contenu différent |
| 8 | `/approve <code affiché>` | ticket créé |
| 9 | `/approve <même code>` | refusé : plus de brouillon en attente (aucun doublon) |
| 10 | `/tickets CNV-310` | refusé : site non autorisé |
| 11 | depuis un 2e compte non lié : `/history CMP-101` | refusé : compte non associé |
| 12 | compte lié à `stagiaire.leroy` : `/create_ticket CMP-101 fuite` | refusé : rôle lecteur |

Plan B sans Telegram (réseau indisponible) : mêmes étapes avec `pytest tests/test_etape4.py -v`
(chaîne bot → API → LangGraph → MCP), `python -m scripts.demo_workflow` et l'interface
Streamlit (mode « Workflow agentique »).

## 3. Questions probables et éléments de réponse

- **Pourquoi l'approbation ne peut-elle pas être contournée par le LLM ?** Le LLM n'a aucun outil
  de création ; seule la reprise humaine du graphe émet un jeton, signé avec un secret absent
  du contexte du LLM, et vérifié par le serveur MCP sur le contenu exact.
- **Pourquoi la délégation d'identité ?** Le bot ne connaît pas les clés API des utilisateurs ;
  le backend agit pour le compte de l'utilisateur authentifié, et le serveur MCP réapplique
  les droits de cet utilisateur.
- **Le gain du reranker adapté est-il significatif ?** Non établi (30 questions de test) ; le
  gain principal vient du reranking lui-même.
- **Passage à l'échelle ?** GPU ou modèles quantifiés, MCP en HTTP persistant, limites
  d'appels partagées (Redis), checkpoints PostgreSQL, corpus réel avec filtrage par autorisation.
