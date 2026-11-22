# Rapport de tests documentaires et d'accès (L1.6)

Critère de validation de l'étape 1 : « une question technique produit une réponse fondée sur
des passages accessibles, avec références vérifiables aux documents et pages ».

Résultats **observés** le 22/11/2026 (CPU, Qwen2.5 3B), extraits de l'évaluation complète
([`data/evaluation/`](../data/evaluation/), [RAPPORT_EVALUATION.md](RAPPORT_EVALUATION.md)).
Corpus : « Attention Is All You Need » (accessible à tous) et le manuel fictif du compresseur
CMP-101 (techniciens et superviseurs de l'atelier A).

## 1. Tests automatisés

| Fichier | Ce qui est vérifié |
|---|---|
| `tests/test_extraction.py` | extraction du texte par page, des 3 tableaux (Markdown + légende) et des figures (images + légendes) |
| `tests/test_v2.py` | fusion des résultats, reranking, sorties structurées |
| `tests/test_etape4.py` | règles d'accès (rôle, site, document sans règle), filtre ajouté aux requêtes PGVector, liste des documents par utilisateur, ajout réservé aux superviseurs |

## 2. Tests documentaires

| Test | Scénarios | Résultat |
|---|---|---|
| Passage attendu parmi les 5 sources retenues (questions FR/EN du jeu de test) | 30 | 27 (0,90) ; MRR@5 0,81 ; page attendue retrouvée 0,93 |
| Réponse complète avec au moins une référence (document, page) | 10 | 9 |
| Réponse citant la page du passage attendu | 10 | 6 |
| Demande hors documentation : clarification ou absence de source | 10 | 8 |

Exemples vérifiés à la main lors de l'étape 1 (voir [NOTE_TECHNIQUE.md](NOTE_TECHNIQUE.md)) :
texte (« Quel score BLEU… » → tableau 2, p. 8), tableau, figure (« Que montre la figure 1 ? »
→ figure 1, p. 3, avec l'image affichée).

Échecs et limites observés :
- 3 questions françaises vagues (« le modèle décrit dans le codage décrit ») ne retrouvent pas
  leur passage : questions générées automatiquement, peu précises.
- Le LLM 3B ne numérote pas ses sources : les références sont rattachées automatiquement aux
  passages jugés pertinents par le reranker, et signalées comme telles.
- Deux demandes hors documentation obtiennent une réponse avec source, avec un score de
  pertinence juste au-dessus du seuil (-2,96 pour -3,0) : « numéro de série de la machine de la
  ligne 3 », et « fréquence de remplacement du filtre à air du compresseur » (le manuel
  CMP-101 traite du compresseur mais pas de ce filtre). Le seuil devrait être relevé ou
  complété par une vérification que la réponse est bien contenue dans les passages.

## 3. Tests d'accès aux documents

| Test | Scénarios | Résultat |
|---|---|---|
| Recherche : seules les sources autorisées sont renvoyées (3 utilisateurs × 4 questions) | 12 | 12 |
| Liste des documents visibles par utilisateur | 3 | 3 |
| Ajout de document refusé hors superviseur | 2 | 2 |
| Réponse du technicien fondée sur le manuel réservé | 1 | 1 (citations du manuel CMP-101) |
| Même question par le stagiaire (lecteur) : manuel invisible | 1 | 1 (aucune citation du manuel, clarification) |

Exemple réel, même question « Que faire en cas de vibrations anormales au démarrage du
compresseur CMP-101 ? » :
- `tech.dupont` (technicien, atelier A) : vérifier les silentblocs, l'accouplement, mesurer le
  niveau vibratoire ; sources : manuel_CMP-101.pdf p. 1 et p. 3 ;
- `stagiaire.leroy` (lecteur) : « Je n'ai pas trouvé dans la documentation indexée d'élément
  correspondant de façon fiable… » et demande de précision (score -4,44).

Contrôle complémentaire dans l'interface Web (connexion de `stagiaire.leroy`) : la barre
latérale ne liste que l'article ; la création de ticket est refusée (403).

## 4. Conclusion

Le critère de validation est atteint sur le corpus de démonstration : les réponses
s'appuient sur des passages que l'utilisateur est autorisé à consulter, avec document et page.
Les limites (citations rattachées automatiquement, seuil de pertinence perfectible, petit
corpus) sont détaillées ci-dessus et dans le rapport d'évaluation.
