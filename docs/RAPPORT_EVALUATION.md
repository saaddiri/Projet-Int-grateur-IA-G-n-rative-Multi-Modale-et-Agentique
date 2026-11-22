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
- Corpus : « Attention Is All You Need » (29 éléments, accessible à tous) et le manuel fictif
  du compresseur CMP-101 (4 éléments, réservé aux techniciens et superviseurs de l'atelier A).

**150 scénarios** :

| Catégorie | Scénarios | Réussis | Latence médiane | Latence p95 |
|---|---|---|---|---|
| Contrôle d'accès (matrice utilisateurs × équipements × outils, authentification, délégation) | 34 | **34** | 3,9 s | 5,2 s |
| Approbation (jetons falsifiés, rejeu, brouillons réels, autre utilisateur, annulation, rôle) | 19 | **19** | 3,1 s | voir §3 |
| Entrées et limites (validation, photos, liaison de comptes, force brute, quota LLM) | 10 | **10** | 0,2 s | 2,4 s |
| Recherche documentaire (passage attendu parmi les 5 sources) | 30 | 27 | 4,2 s | 5,0 s |
| Accès aux documents (filtrage par rôle et site : recherche, liste, ajout, réponses) | 19 | **19** | 0,1 s | 227 s |
| Réponse documentaire complète (`/ask`) | 10 | 9 | 119 s | 167 s |
| Hors corpus (clarification ou absence de source attendue) | 10 | 8 | 101 s | 126 s |
| Injection de prompt (fuite de secret, création de ticket) | 16 | **16** | 80 s | 112 s |
| Photo (`/analyze`) : nette → réponse sourcée ; floue → clarification | 2 | 2 | 337 s | 387 s |
| **Total** | **150** | **144** | | |

## 2. Qualité des citations

| Mesure | Valeur |
|---|---|
| Passage attendu parmi les 5 sources retenues (30 questions de test) | 0,90 |
| Page attendue parmi les 5 sources | 0,93 |
| MRR@5 du passage attendu | 0,81 |
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
| Filtrage des documents : sources hors droits jamais renvoyées, manuel réservé invisible pour un lecteur, ajout réservé aux superviseurs | 19 | tous conformes |

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
- Les catégories accès, approbation et entrées proviennent du journal de la première
  exécution complète (21/11), interrompue à l'injection 9 : scénarios conservés (réussite,
  latence) sans leurs détails. Injections et photos ont été rejouées le 21/11 ; recherche,
  réponses documentaires, hors corpus et accès aux documents le 22/11, après l'ajout du
  manuel CMP-101 et du filtrage documentaire.
- **Base PostgreSQL redémarrée** (22/11) : une connexion perdue a réinitialisé le serveur, et
  le backend conservait des connexions mortes (erreur 500 puis requête bloquée). Correction :
  vérification des connexions avant usage (`pool_pre_ping`). C'est peut-être la cause des
  erreurs 500 de la nuit précédente, sans certitude.

## 6. Limites

- 30 questions documentaires et deux documents : les taux ont une forte incertitude (une
  question représente 3 à 10 points).
- Le manuel CMP-101 est fictif et court (4 éléments) : il démontre le lien documentation /
  équipements et le filtrage, pas la qualité sur un vrai corpus de maintenance.
- Le seuil de pertinence (-3,0) reste un compromis : deux questions hors corpus ont obtenu une
  réponse avec source, avec un score de -2,96 (« numéro de série de la machine de la ligne 3 »,
  « fréquence de remplacement du filtre à air du compresseur », que le manuel ne traite pas) ;
  la photo floue a obtenu -2,86 et c'est le LLM qui a demandé la clarification. Il faudrait
  relever le seuil ou vérifier que la réponse figure dans les passages cités.
- Test réel depuis Telegram (22/11, un seul compte, `tech.dupont`) — voir §7.
- Mesures obtenues pendant que d'autres tâches occupaient la machine (construction de
  l'image Docker) : les latences maximales sont pessimistes.

## 7. Test réel depuis Telegram (critère de validation de l'étape 4)

Bot créé avec @BotFather, backend et bot lancés en local, compte Telegram lié à `tech.dupont`
par code à usage unique. Observé dans les journaux du backend et du serveur MCP :

| Heure | Action dans Telegram | Résultat |
|---|---|---|
| 14:11 | `/start <code>` | compte associé (`POST /link` 200, code marqué utilisé) |
| 14:34 | `/history CMP-101` | historique renvoyé via MCP (200) |
| 14:3x | photo + légende | photo téléchargée par le bot, `/analyze` 200 |
| 14:42 | `/create_ticket CMP-101 vibrations au démarrage depuis ce matin` | historique + tickets consultés (MCP), brouillon renvoyé (200), aucun ticket créé |
| 14:47 | `/approve <code affiché>` | ticket T-0003 créé par le serveur MCP après l'approbation |
| 14:48 | `/tickets CNV-310` | refusé par le serveur MCP : « tech.dupont n'est pas habilité sur le site ATELIER-B » |

Critère « depuis Telegram, un technicien transmet une photo, reçoit une réponse, consulte un
historique et approuve un ticket ; les actions non autorisées sont bloquées » : vérifié sur ce
parcours. Le contenu de la réponse à la photo n'a pas été relevé dans les journaux (journal
minimal, sans contenu) ; le refus d'un compte lecteur et d'un compte non associé n'a pas été
rejoué depuis Telegram (couvert par les tests automatisés et l'évaluation).
