# Note de cadrage et architecture (L1.1)

## 1. Contexte et besoin

Les techniciens de maintenance consultent des manuels, procédures, schémas et historiques
dispersés, souvent sur le terrain, avec une photo de l'équipement comme seul point de départ.
MaintiDoc doit leur fournir des **réponses sourcées** (document, page), exploiter les
**photographies**, consulter l'**historique** des équipements et préparer des **tickets**
soumis à validation humaine.

## 2. Périmètre

| Dans le périmètre | Hors périmètre |
|---|---|
| Questions sur la documentation indexée (texte, tableaux, figures) | Pilotage d'équipements |
| Analyse de photos pour orienter la recherche | Diagnostic garanti |
| Consultation de l'historique et des tickets (service simulé, MCP) | Création de ticket sans approbation humaine |
| Brouillons de tickets avec approbation explicite | Remplacement d'un technicien habilité |
| Accès par Telegram et interface Web, avec contrôle des droits | Données réelles d'une GMAO (service simulé) |

## 3. Utilisateurs et droits

| Rôle | Consultation documentaire | Historique / tickets | Création de ticket | Ajout de document |
|---|---|---|---|---|
| lecteur | documents publics de ses sites | ses sites | non | non |
| technicien | documents de ses sites | ses sites | oui, après approbation | non |
| superviseur | tous les documents de ses sites (et documents sans règle) | ses sites | oui, après approbation | oui |

Règles d'accès aux documents : [`data/documents_acl.json`](../data/documents_acl.json) ;
utilisateurs et sites : [`data/maintenance/seed.json`](../data/maintenance/seed.json).

## 4. Contraintes

- Modèles **open source** exécutés localement (Ollama, Hugging Face) ; aucune donnée envoyée à un service tiers.
- Machine de développement **sans GPU** (4 cœurs, 16 Go) : modèles de 3 milliards de paramètres.
- Citations vérifiables, filtrage des documents selon les autorisations, approbation humaine
  obligatoire, secrets protégés, entrées non fiables, limites d'appels, journalisation minimale.
- Mesures présentées comme résultats observés, jamais comme garanties.

## 5. Architecture cible

```
 Telegram ─► bot ─┐                        ┌─► RAG multimodal : LlamaIndex · PGVector · CLIP · VLM · reranker
 Web (Streamlit) ─┼─► Backend FastAPI ─────┼─► Graphe LangGraph (brouillon, interruption, reprise)
 Clients API ─────┘   identités, droits,   └─► Client MCP ─► Serveur MCP maintenance (droits, jetons, audit)
                      filtrage documentaire,
                      limites, journal         Ollama : Qwen2.5 3B (LLM), Qwen2.5-VL 3B (VLM)
```

Incréments : étape 1 RAG multimodal agentique → étape 2 photos, fusion, reranker adapté →
étape 3 LangGraph, MCP, approbation → étape 4 bot, backend, sécurité, Docker Compose.

## 6. Choix techniques

| Besoin | Choix | Raison |
|---|---|---|
| Extraction PDF | LlamaIndex (PyMuPDFReader) + PyMuPDF | texte, tableaux en Markdown, images avec légendes |
| Base vectorielle | **PGVector** (PostgreSQL 16) | imposée par le cahier des charges de la Partie I ; métadonnées JSONB filtrables (type de contenu, document, droits) ; une seule base à exploiter |
| Embeddings texte | `intfloat/multilingual-e5-small` | multilingue (questions FR, documents EN), léger sur CPU |
| Images | CLIP ViT-B/32 + CLIP multilingue + descriptions VLM | recherche texte → image en français et image → image |
| LLM / VLM | Qwen2.5 3B / Qwen2.5-VL 3B via Ollama | open source, appel d'outils, sortie JSON, exécutable sur CPU |
| Reranking | cross-encoder mMiniLM multilingue, adapté au corpus | meilleur classement pour un coût faible |
| Orchestration | LangChain (agent), LangGraph (workflow avec interruption) | outils de recherche ; état persistant et reprise humaine |
| Outils métier | MCP (SDK officiel) | interface standard entre l'assistant et le service de maintenance |
| Backend / clients | FastAPI, python-telegram-bot, Streamlit | API documentée, bot asynchrone, interface rapide |

**PGVector ou Qdrant.** Le découpage pédagogique mentionne un « index Qdrant » (L1.3), alors
que le cahier des charges de la Partie I demande PGVector. Nous avons retenu PGVector : il est
explicitement demandé, il couvre le besoin (recherche vectorielle + filtres de métadonnées,
utilisés pour le filtrage par autorisations) et il évite d'exploiter une seconde base. Le
passage à Qdrant ne toucherait que [`maintidoc/store.py`](../maintidoc/store.py) (LlamaIndex
fournit un `QdrantVectorStore` équivalent).

**API FastAPI (L1.4).** La chaîne RAG de l'étape 1 est exposée par le backend (`/rag` pour le
RAG simple, `/agent` pour l'agent LangChain, `/ask` pour le pipeline complet). Le backend a été
livré à l'étape 4 avec la gestion des identités ; l'interface Streamlit l'utilise pour toutes
ses actions.

## 7. Risques

| Risque | Mesure |
|---|---|
| Hallucination du LLM | réponse limitée aux passages retrouvés, citations construites par le système, seuil de pertinence et demande de clarification |
| Photo ambiguë ou floue | confiance du VLM, seuil de pertinence, clarification |
| Action non voulue (ticket) | aucun outil d'action pour le LLM, approbation explicite liée au contenu exact |
| Fuite d'information | filtrage documentaire côté serveur, droits par site, secrets hors dépôt |
| Latence sur CPU | modèles 3B, reranking sur 20 candidats ; GPU recommandé en production |

## 8. Critères de réussite

Repris des critères de validation de chaque étape ; résultats dans
[RAPPORT_TESTS_ETAPE1.md](RAPPORT_TESTS_ETAPE1.md), [RAPPORT_RERANKER.md](RAPPORT_RERANKER.md),
[WORKFLOW_ETAPE3.md](WORKFLOW_ETAPE3.md) et [RAPPORT_EVALUATION.md](RAPPORT_EVALUATION.md).
