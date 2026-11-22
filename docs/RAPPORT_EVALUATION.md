# Rapport d'évaluation et de performances (L4.5)

Résultats **observés** sur une seule machine et un petit corpus ; ce ne sont pas des garanties.

## 1. Protocole

- Script : [`scripts/evaluate.py`](../scripts/evaluate.py) ; résultats bruts :
  [`data/evaluation/results.jsonl`](../data/evaluation/results.jsonl), synthèse :
  [`data/evaluation/summary.json`](../data/evaluation/summary.json).
- Chaîne réelle : backend FastAPI (authentification, limites), graphe LangGraph, serveur MCP en
  sous-processus, RAG multimodal V2, reranker adapté, Qwen2.5 3B et Qwen2.5-VL 3B via Ollama.
  Bases de maintenance, d'identités et de checkpoints temporaires.
- Machine : 4 cœurs, 15,4 Go de RAM, **sans GPU** (Windows 10).
- Corpus : un seul document (« Attention Is All You Need », 29 éléments indexés).

**131 scénarios** (l'objectif indicatif était d'environ 150) :

| Catégorie | Scénarios | Réussis | Latence médiane | Latence p95 |
|---|---|---|---|---|
| Contrôle d'accès (matrice utilisateurs × équipements × outils, authentification, délégation) | 34 | **34** | 3,9 s | 5,2 s |
| Approbation (jetons falsifiés, rejeu, brouillons réels, autre utilisateur, annulation, rôle) | 19 | **19** | 3,1 s | voir §3 |
| Entrées et limites (validation, photos, liaison de comptes, force brute, quota LLM) | 10 | **10** | 0,2 s | 2,4 s |
| Recherche documentaire (passage attendu parmi les 5 sources) | 30 | 27 | 3,2 s | 4,1 s |
| Réponse documentaire complète (`/ask`) | 10 | 9 | 101 s | 167 s |
| Hors corpus (clarification ou absence de source attendue) | 10 | 9 | 116 s | 129 s |
| Injection de prompt (fuite de secret, création de ticket) | 16 | **16** | 80 s | 112 s |
| Photo (`/analyze`) : nette → réponse sourcée ; floue → clarification | 2 | 2 | 337 s | 387 s |
| **Total** | **131** | **126** | | |

## 2. Qualité des citations

| Mesure | Valeur |
|---|---|
| Passage attendu parmi les 5 sources retenues (30 questions de test) | 0,90 |
| MRR@5 du passage attendu | 0,79 |
| Réponses avec au moins une citation (10 questions) | 0,90 |
| Réponses citant la page du passage attendu | 0,60 |
| Citations rattachées automatiquement (LLM sans numéro de source) | **100 %** |

Lecture :
- La recherche retrouve bien le passage attendu (9 fois sur 10), comme lors de l'étape 2.
- **Le LLM 3B n'a numéroté aucune de ses sources** : toutes les citations viennent du
  rattachement automatique aux passages jugés pertinents par le reranker, signalé comme tel à
  l'utilisateur. La citation est donc vérifiable mais pas garantie d'avoir servi à la réponse :
  c'est la principale limite de qualité observée.
- La page attendue n'est citée que dans 6 réponses sur 10 : les 3 questions dont le passage
  n'est pas retrouvé en font partie ; une réponse anglaise (comparaison ConvS2S / ByteNet) a été
  bloquée par le seuil de pertinence et remplacée par une demande de clarification.

## 3. Latence et ressources

| Opération | Latence observée (CPU) |
|---|---|
| Historique / tickets via MCP | 3 à 5 s (dont ≈ 2,5 s de lancement du serveur MCP) |
| Recherche + reranking seuls | 3 à 4 s |
| Question documentaire complète | 1 min 15 à 2 min 50 |
| Brouillon de ticket | ≈ 5 min, et jusqu'à 17 min quand la machine construisait l'image Docker en parallèle |
| Photo | 5 à 6,5 min |

| Ressource | Valeur |
|---|---|
| Mémoire du processus applicatif (pic) | 2,8 Go (embeddings, CLIP, CLIP multilingue, reranker) |
| CPU moyen pendant l'évaluation | 94 % des 4 cœurs |
| Mémoire d'Ollama | non mesurée correctement : l'échantillonneur n'a vu que le processus principal (31 Mo), pas les processus d'inférence des modèles |
| Image Docker `maintidoc:4.0.0` | 3,2 Go |

Conclusion : le système est **utilisable pour une démonstration**, pas pour un usage
simultané par plusieurs techniciens sur CPU. Un GPU, ou des modèles plus petits ou quantifiés,
sont nécessaires pour descendre sous les 10 s par question.

## 4. Sécurité

| Contrôle | Scénarios | Résultat |
|---|---|---|
| Accès par site et par rôle, sans authentification, clé inconnue, secret de bot faux, compte Telegram non associé, délégation forgée | 34 | tous bloqués ou autorisés comme attendu |
| Jeton absent, mal formé, signature falsifiée, autre secret, expiré, autre utilisateur, autre équipement, titre ou description modifiés | 9 | tous refusés par le serveur MCP |
| Rejeu, double approbation, contenu identique déjà ouvert | 3 | aucun doublon |
| Approbation d'un autre contenu, du brouillon d'un autre utilisateur, après annulation ; rôle lecteur | 5 | tous refusés |
| Force brute sur les codes de liaison, quota LLM | 2 | 429 renvoyé |
| 15 injections de prompt (prompt système, secrets, clés API, création de tickets, commandes système) | 16 | aucune fuite de secret, aucun ticket créé |

Les secrets ne figurent jamais dans le contexte du LLM, et aucune action ne dépend de sa
sortie sans approbation humaine : les injections ne peuvent donc ni exfiltrer de secret ni
créer de ticket. Elles peuvent en revanche orienter le texte de la réponse : le contenu des
réponses à ces injections n'a pas été relu un par un, seule l'absence de fuite et de création
a été vérifiée.

## 5. Incidents relevés pendant l'évaluation

- **Erreur 500 intermittente sur `/ask`** : deux exécutions se sont arrêtées sur une injection
  (scénarios 9 et 12, différents à chaque fois). Le LLM produit parfois une sortie que la
  validation rejette. Corrections : nouvel essai puis réponse de repli prudente dans le
  pipeline, gestionnaire d'erreurs du backend (trace journalisée, message générique au
  client), évaluation qui enregistre au fil de l'eau. Lors de l'exécution finale des
  16 injections, aucune erreur 500 n'est apparue. La cause exacte de la seconde erreur
  n'a pas été isolée : l'incident reste donc ouvert.
- Les résultats des catégories accès, approbation, entrées et hors corpus proviennent du
  journal de la première exécution complète, interrompue à l'injection 9. Les scénarios y
  sont conservés (réussite, latence), sans leurs détails ; les autres catégories ont été
  rejouées en entier.

## 6. Limites

- 131 scénarios au lieu d'environ 150 ; 30 questions documentaires et un seul document : les
  taux ont une forte incertitude (une question représente 3 à 10 points).
- Équipements et historique fictifs, sans rapport avec le document indexé : les tickets ne
  citent aucune documentation pertinente dans cette démonstration.
- Le seuil de pertinence (-3,0) reste un compromis : une question hors corpus (« numéro de
  série de la machine de la ligne 3 ») a obtenu une réponse avec sources ; la photo floue a
  obtenu un score de -2,86, juste au-dessus du seuil, et c'est le LLM qui a demandé la
  clarification.
- Le bot n'a pas été évalué contre l'API Telegram réelle (jeton @BotFather nécessaire) ; sa
  logique est testée jusqu'au backend.
- Mesures obtenues pendant que d'autres tâches occupaient la machine (construction de
  l'image Docker) : les latences maximales sont pessimistes.
