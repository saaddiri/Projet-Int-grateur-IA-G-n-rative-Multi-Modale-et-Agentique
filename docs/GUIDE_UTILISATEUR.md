# Guide utilisateur — MaintiDoc (L4.6)

MaintiDoc répond à vos questions à partir de la documentation technique indexée, en citant
ses sources (document, page). Il consulte l'historique des équipements et prépare des tickets
de maintenance que **vous seul pouvez approuver**.

> MaintiDoc ne pilote aucun équipement et ne garantit aucun diagnostic. Il ne remplace pas
> un technicien habilité : vérifiez toujours les sources citées.

## 1. Se connecter au bot Telegram

1. Demandez un **code de liaison** à votre administrateur (valable 10 minutes, usage unique).
2. Ouvrez le bot MaintiDoc dans Telegram et envoyez : `/start VOTRECODE`
3. Le bot confirme votre nom et votre rôle. Sans liaison, toutes les commandes sont refusées.

## 2. Commandes

| Commande | Exemple | Effet |
|---|---|---|
| `/start` | `/start` | aide et rappel de votre identité |
| `/ask` | `/ask À quoi sert le Multi-Head Attention ?` | réponse sourcée, niveau d'incertitude |
| `/analyze` | envoyer une **photo** avec la question en légende | analyse de la photo + réponse sourcée, ou demande de précision |
| `/history` | `/history CMP-101` | historique des interventions |
| `/tickets` | `/tickets CMP-101` | tickets ouverts |
| `/create_ticket` | `/create_ticket CMP-101 vibrations au démarrage` | **brouillon** de ticket (rien n'est créé) |
| `/approve` | `/approve 3fa94c1b` | crée le ticket **exactement** tel qu'affiché |
| `/cancel` | `/cancel` | abandonne le brouillon en attente |

## 3. Créer un ticket

1. `/create_ticket CMP-101 vibrations au démarrage depuis ce matin`
2. Le bot consulte l'historique, la documentation, puis affiche un **brouillon** : titre,
   description, priorité proposée et éventuels tickets ouverts similaires.
3. Relisez. Si le contenu convient, envoyez `/approve` suivi du **code affiché** sous le
   brouillon. Le code identifie ce contenu précis : un autre code est refusé.
4. Sinon `/cancel`, puis reformulez avec `/create_ticket`.

Bon à savoir :
- un brouillon expire après 15 minutes ; un nouveau brouillon remplace le précédent ;
- approuver deux fois ne crée pas deux tickets ;
- vous ne pouvez approuver que vos propres brouillons ;
- le rôle « lecteur » peut consulter mais pas créer de ticket ;
- vous ne voyez que les équipements de vos sites.

## 4. Lire une réponse

- **Sources** `[n] document, p. X` : vérifiez-les avant d'agir. « Attribution automatique »
  signifie que le modèle n'a pas numéroté ses sources et que le système a rattaché la réponse
  aux passages jugés les plus pertinents.
- **Incertitude** 🟢 faible, 🟠 moyenne, 🔴 élevée.
- **❓ Clarification** : la documentation ne couvre pas la demande de façon fiable, ou la photo
  est ambiguë. Précisez l'équipement ou envoyez une photo plus nette et plus proche.

## 5. Temps de réponse

Sur le serveur de démonstration (CPU, sans GPU) : consultation d'historique en quelques
secondes, question documentaire en 1 à 2 minutes, photo en 3 à 7 minutes. Le bot indique
qu'il travaille ; inutile de renvoyer la commande. Au-delà de 4 requêtes avec IA par minute,
le bot vous demande de patienter.

## 6. Interface Web

`http://<serveur>:8501` : mêmes fonctions (question, photo, workflow de ticket avec
formulaire d'approbation), pour la démonstration et le poste fixe.
