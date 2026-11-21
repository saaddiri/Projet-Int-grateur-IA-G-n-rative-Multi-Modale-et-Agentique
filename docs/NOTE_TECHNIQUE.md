# Note technique — Étape 1 : RAG multimodal agentique

## 1. Objectif

Application de questions-réponses sur des documents PDF contenant du texte, des images et
des tableaux. Les contenus sont indexés de manière multimodale dans PGVector, et les réponses
sont produites par un LLM open source orchestré par un agent LangChain. Chaque réponse cite
ses sources (document, page, type de contenu).

Document de démonstration : *Attention Is All You Need* (Vaswani et al., 2017), 11 pages,
3 images matricielles (figures 1 et 2) et 3 tableaux.

## 2. Chaîne d'indexation

| Étape | Outil | Détail |
|---|---|---|
| Lecture du texte | LlamaIndex `PyMuPDFReader` | un `Document` par page, métadonnées `source`, `page`, `doc_id` |
| Découpage | LlamaIndex `SentenceSplitter` | 512 tokens, recouvrement 64 |
| Tableaux | PyMuPDF `page.find_tables()` | conversion en Markdown, légende « Table N » rattachée par proximité |
| Images | PyMuPDF (`get_image_info`, `Pixmap`) | export PNG, légende « Figure N » rattachée par proximité, filtre < 64 px |
| Représentation des images | CLIP `openai/clip-vit-base-patch32` | vecteur 512d normalisé, table `data_maintidoc_images` |
| Description des images | Qwen2.5-VL 3B via Ollama | image réduite à 672 px, description factuelle guidée par la légende |
| Embeddings texte | `intfloat/multilingual-e5-small` | préfixes `query:` / `passage:`, table `data_maintidoc_text` |
| Stockage | PostgreSQL 16 + pgvector, `PGVectorStore` (LlamaIndex) | vecteurs + métadonnées JSON (`content_type`, `page`, `caption`, `image_path`) |

La réindexation d'un document est **idempotente** : les vecteurs existants du même `doc_id`
(hash SHA-1 du fichier) sont supprimés avant insertion.

## 3. Chaîne de requête

1. **Interface Streamlit** : chargement du PDF, déclenchement de l'indexation, saisie des questions.
2. **Agent LangChain** (`create_agent`) avec le LLM `qwen2.5:3b` (Ollama), qui choisit parmi trois outils :
   - `search_text` : passages de texte (filtre `content_type = text`) ;
   - `search_tables` : tableaux (filtre `content_type = table`) ;
   - `search_figures` : fusion de la recherche CLIP texte → image et de la recherche sur les descriptions VLM.
3. Les passages retournés par les outils sont **numérotés globalement** ([1], [2]…) pour que la réponse
   finale les cite ; l'interface affiche les sources (page, score, extrait, image).
4. **Gestion du contexte** : historique glissant des 3 derniers échanges transmis à l'agent.
5. **Repli** : si le modèle ne déclenche aucun outil, un RAG classique (recherche globale + génération) est utilisé.

## 4. Choix des modèles

| Besoin | Modèle | Justification |
|---|---|---|
| LLM | `qwen2.5:3b` | open source, supporte l'appel d'outils, exécutable sur CPU via Ollama |
| VLM | `qwen2.5vl:3b` | open source, bonne lecture des schémas, même famille que le LLM |
| Embeddings | `multilingual-e5-small` (384d) | multilingue (questions FR, document EN), léger (470 Mo) ; `BAAI/bge-m3` configurable dans `.env` |
| Image ↔ texte | CLIP ViT-B/32 | standard pour la recherche texte → image |

Tous les modèles s'exécutent localement : aucune donnée ne quitte la machine.

## 5. Limites observées

- **Exécution sur CPU** : la description d'une figure prend 2 à 4 minutes avec Qwen2.5-VL 3B ; les
  réponses de l'agent prennent de quelques dizaines de secondes à quelques minutes. Un GPU réduit
  fortement ces durées. `DESCRIBE_IMAGES=0` permet d'indexer sans description (légende seule).
- **Figures vectorielles** : seules les images matricielles sont extraites ; un schéma dessiné en
  vectoriel n'est pas capturé à cette étape (rendu de zone de page envisageable).
- **Tableaux** : la détection PyMuPDF dépend de la mise en page ; un tableau sans bordures peut
  être manqué ou découpé (ici, les 3 tableaux du document sont détectés). Le texte des tableaux
  reste aussi présent dans les passages de la page.
- **CLIP** comprend surtout des requêtes courtes en anglais : la recherche sur les descriptions VLM
  compense pour les questions en français.
- **Petit LLM (3B)** : il peut ignorer un outil pertinent ou mal citer une source ; un modèle plus
  grand (`qwen2.5:7b`) améliore la qualité au prix de la latence.
- **PDF scannés** : pas d'OCR à cette étape.

## 6. Exemples de questions / réponses

Résultats observés sur le document de démonstration (CPU, `qwen2.5:3b`), via
`python -m scripts.ask`. Les réponses sont résumées ; l'outil choisi par l'agent et les
sources retournées sont reportés tels quels.

| Question | Outil choisi par l'agent | Réponse (résumé) | Sources |
|---|---|---|---|
| Qu'est-ce que le mécanisme de Multi-Head Attention ? | `search_text("Multi-Head Attention mechanism")` | plusieurs têtes d'attention en parallèle sur des projections différentes de Q, K, V, permettant d'attendre à différents sous-espaces de représentation ; trois usages dans le Transformer (encodeur-décodeur, self-attention de l'encodeur, self-attention masquée du décodeur) | p. 4, 5, 4, 2 (texte) |
| Quel score BLEU obtient le Transformer (big) sur WMT 2014 anglais-allemand ? | `search_tables("Transformer (big) WMT 2014 BLEU score")` | 28.4 BLEU (27.3 pour le modèle de base) — valeurs exactes du tableau 2 | p. 8 (tableau 2), p. 9 (tableau 3) |
| Que montre la figure 1 du document ? | `search_figures("figure 1")` | architecture du Transformer : piles encodeur/décodeur, Multi-Head Attention, Add & Norm, Feed Forward, encodage positionnel, Softmax de sortie | p. 3 (image, figure 1), p. 4 (image, figure 2) |

Observations :
- l'agent choisit le bon outil pour chaque type de question (texte, tableau, figure) ;
- la valeur chiffrée issue du tableau est exacte ;
- la description de la figure 1 contient quelques détails approximatifs produits par le VLM
  (couleurs, notation « N* ») : les descriptions automatiques doivent être relues ;
- durée observée : environ 1 à 4 minutes par question sur CPU partagé.

## 7. Difficultés rencontrées et corrections

| Problème observé | Cause | Correction |
|---|---|---|
| Description d'image tronquée (« The ») après 10 min | figure 1520×2239 px : trop de tokens visuels pour la fenêtre de 4096 | réduction à 672 px avant envoi au VLM (≈ 3 min par image) |
| Réponse vide de l'agent, aucun outil appelé | appel d'outil mal formé par le petit modèle, rejeté par Ollama | middleware LangChain `wrap_model_call` qui relance le modèle avec une consigne explicite |
| « Que montre la figure 1 ? » ne retrouvait pas la figure 1 | la similarité sémantique capture mal un numéro de figure | détection « figure N / tableau N » et recherche exacte sur la légende, placée en tête |
| Réindexation : doublons et liste de documents incohérente | `PGVectorStore` écrase la métadonnée `doc_id` avec l'identifiant interne LlamaIndex | métadonnée renommée `file_id` ; suppression préalable par `file_id` (idempotence vérifiée) |
| Réindexation lente | description VLM recalculée à chaque fois | cache des descriptions sur disque (`storage/images/*.txt`) |
