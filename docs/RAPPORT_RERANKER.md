# Rapport — Étape 2 : multimodalité et reranker adapté

## 1. Objectif

Étendre MaintiDoc aux photographies et figures, fusionner les résultats de plusieurs
recherches, appliquer un reranking, puis mesurer l'intérêt d'adapter un petit reranker
multilingue au corpus.

## 2. Pipeline multimodal V2

| Étape | Composant |
|---|---|
| Analyse de photo | Qwen2.5-VL 3B (Ollama), sortie JSON validée par Pydantic (`PhotoAnalysis`) |
| Recherche texte | embeddings `multilingual-e5-small` sur passages, tableaux et descriptions de figures |
| Recherche texte → image | `clip-ViT-B-32-multilingual-v1` (texte multilingue aligné sur l'espace CLIP ViT-B/32) |
| Recherche image → image | CLIP ViT-B/32 sur la photo envoyée |
| Fusion | Reciprocal Rank Fusion (k = 60), 20 candidats |
| Reranking | cross-encoder `mmarco-mMiniLMv2-L12-H384-v1` initial ou adapté, top 5 gardés |
| Génération | Qwen2.5 3B, sortie structurée `LLMAnswer` (réponse, passages cités, incertitude, clarification) |
| Garde-fous | citations reconstruites par le système à partir des numéros (pas de source inventée) ; si le LLM n'en donne aucune, rattachement aux passages jugés pertinents par le reranker, marqué « attribution automatique » ; meilleur score du reranker < seuil → pas de réponse spéculative, demande de clarification |

## 3. Jeu de données d'adaptation (L2.3)

Construit automatiquement par [`scripts/build_dataset.py`](../scripts/build_dataset.py) à
partir des 29 éléments indexés (23 passages, 3 tableaux, 3 descriptions de figures).

| Étape | Résultat |
|---|---|
| Questions générées par Qwen2.5 3B (FR + EN) | 237 |
| Rejetées par le juge LLM (passage ne contenant pas la réponse) | 78 (33 %) |
| Questions retenues après déduplication | **156** (124 sur le texte, 15 sur les tableaux, 17 sur les figures ; 75 FR / 81 EN) |
| Découpage train / test (stratifié par passage, graine 42) | 126 / 30 questions |
| Négatifs difficiles par question (top de la recherche vectorielle hors positif) | 5 |
| **Paires (question, passage, label)** | **936** (756 train, 180 test) |

Vérification :
- **automatique** : un juge LLM valide chaque question contre son passage (vérification groupée par passage) ;
- **manuelle** : un échantillon de 12 questions retenues a été relu ; elles sont pertinentes, mais
  quelques-unes restent vagues (« le modèle décrit », « notre recherche ») et certaines portent sur
  les références bibliographiques, peu utiles pour un technicien. Le juge rejette aussi quelques
  questions correctes (il est strict).

Limite importante : les négatifs difficiles sont choisis automatiquement ; un passage voisin
peut parfois contenir lui aussi la réponse (faux négatif).

## 4. Reranker adapté (L2.4)

| Paramètre | Valeur |
|---|---|
| Modèle de base | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (multilingue, 118 M paramètres) |
| Perte | BCE sur logit (positif = 1, négatif difficile = 0) |
| Époques / batch / lr | 1 / 16 / 3e-5 (warmup 10 % puis décroissance linéaire) |
| Longueur max | 320 tokens |
| Durée | 17 min sur CPU |
| Sortie | `models/reranker-maintidoc-v1/` (poids, non versionnés dans git) |
| Version | fiche [`models/reranker-maintidoc-v1.json`](../models/reranker-maintidoc-v1.json) : version 1.0.0, hyperparamètres, empreinte SHA-256 du jeu d'entraînement |

Le modèle est reproductible : `python -m scripts.train_reranker` avec la même graine et le même
fichier `pairs_train.jsonl` (empreinte vérifiable dans la fiche).

## 5. Résultats comparatifs (L2.5)

Protocole : pour chaque question de test, les 20 candidats de la recherche vectorielle sont
réordonnés ; un seul passage est pertinent. Le passage positif figure dans les 20 candidats pour
30 questions sur 30. Script : [`scripts/eval_reranker.py`](../scripts/eval_reranker.py),
résultats bruts : [`data/reranker/eval_test.json`](../data/reranker/eval_test.json).

| Configuration | nDCG@5 | MRR@10 | Recall@5 | nDCG@5 FR | nDCG@5 EN | Temps / question (CPU) |
|---|---|---|---|---|---|---|
| Sans reranker | 0,611 | 0,600 | 0,733 | 0,466 | 0,722 | — |
| Reranker initial | 0,822 | 0,799 | 0,900 | 0,761 | 0,868 | ≈ 6,1 s |
| **Reranker adapté** | **0,846** | **0,821** | **0,933** | 0,741 | **0,927** | ≈ 5,6 s |

### Interprétation (résultats observés, pas des garanties)

- **Le reranking apporte l'essentiel du gain** : +0,21 de nDCG@5 et +0,20 de MRR@10 par rapport à
  la recherche vectorielle seule. Le gain est le plus fort pour les questions en français
  (0,47 → 0,76), où l'embedding léger est le plus faible.
- **L'adaptation apporte un gain supplémentaire modeste** : +0,025 de nDCG@5, +0,023 de MRR@10.
  Il porte sur les questions en anglais (0,87 → 0,93) ; en français, le reranker adapté est
  légèrement en dessous du reranker initial (0,74 contre 0,76).
- **Ces écarts ne sont pas statistiquement établis** : 30 questions de test, un seul document. Une
  question de plus ou de moins bien classée change le nDCG@5 d'environ 0,03. Il faudrait plus de
  questions et plusieurs documents pour conclure.
- **Coût** : le reranking ajoute environ 6 s par question sur CPU (20 paires question-passage), ce
  qui reste faible devant la génération par le LLM.

## 6. Seuil de pertinence et clarification

Le meilleur score du reranker adapté (logit) sert de signal « la documentation couvre-t-elle la
demande ? ». Distribution sur les 30 questions de test (toutes couvertes par le corpus) :

| Seuil | Questions valides au-dessus du seuil |
|---|---|
| 0,0 | 10 / 30 |
| -1,0 | 15 / 30 |
| -2,0 | 25 / 30 |
| **-3,0 (retenu)** | **29 / 30** |
| -4,0 | 30 / 30 |

Avec -3,0, une question valide sur 30 serait traitée à tort comme hors documentation (réponse
remplacée par une demande de clarification). Les deux distributions (questions couvertes / non
couvertes) n'ont pas été mesurées sur un jeu dédié de demandes hors sujet : ce seuil est un
compromis observé sur peu d'exemples, à réévaluer avec plus de données.

## 7. Démonstration V2 (L2.6) — cas observés

Exécutés avec `python -m scripts.ask_photo` (reranker adapté, CPU) ; photos de test dans
[`data/photos_test/`](../data/photos_test/).

| Cas | Photo | Résultat observé | Durée |
|---|---|---|---|
| Schéma net + « Que représente ce schéma et à quoi sert le bloc Multi-Head Attention ? » | photo inclinée de la figure 1 | figure 1 classée 1re (score 1,74) ; réponse décrivant l'architecture et le rôle du Multi-Head Attention ; 3 citations (figures 1 et 2, attribution automatique) ; incertitude moyenne (VLM peu confiant) ; pas de clarification | 6 min 30 |
| Photo floue + « Que signifie ce voyant ? » | figure 1 très floutée | meilleur score -4,91 < -3,0 ; aucune réponse inventée ; incertitude élevée ; **clarification demandée** (préciser l'équipement ou envoyer une photo plus nette) | 2 min 10 |

Critère de validation de l'étape : « une photographie d'équipement donne lieu à une réponse
documentaire sourcée ou à une clarification explicite en cas d'ambiguïté » — vérifié sur ces deux cas.

## 8. Limites observées

- **VLM 3B sur CPU** : sur la photo floue, Qwen2.5-VL a décrit un « panneau de commande » avec des
  voyants qui n'existent pas, influencé par la question ; sur le schéma net, il répète en boucle les
  étiquettes et tronque son JSON (repli : description brute, confiance faible). C'est le seuil de
  pertinence du reranker, et non le VLM, qui empêche la réponse spéculative.
- **Citations** : le LLM 3B omet souvent les numéros de passages ; l'attribution automatique par
  le reranker est signalée comme telle dans la sortie.
- **Corpus de démonstration** : un seul document (29 éléments) ; les métriques et le seuil doivent
  être réévalués sur le corpus de maintenance réel (30 à 50 documents prévus).
- **Durée** : 2 à 7 minutes par requête avec photo sur CPU (dont ≈ 3 min pour le VLM).
