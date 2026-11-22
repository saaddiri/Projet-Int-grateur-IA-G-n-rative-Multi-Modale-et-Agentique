"""Orchestration agentique avec LangChain.

L'agent (LLM Qwen via Ollama) choisit lui-même les outils de recherche à
appeler selon la question : passages de texte, tableaux ou figures. Les
passages retrouvés sont numérotés globalement pour que la réponse finale
puisse les citer ([1], [2], ...). L'historique de conversation est conservé
(fenêtre glissante) pour gérer le contexte entre les questions.
"""

from __future__ import annotations

from contextvars import ContextVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.errors import GraphRecursionError

from .rag import Answer, answer_with_rag, get_llm
from .retrieval import Hit, search_images, search_tables, search_text

MAX_HISTORY_MESSAGES = 6  # 3 derniers échanges question/réponse
# Nombre maximal d'étapes du graphe de l'agent (appels LLM + exécutions d'outils) : au-delà,
# le petit modèle tourne en rond et chaque étape coûte 1 à 2 min sur CPU. 7 étapes permettent
# jusqu'à 3 recherches suivies d'une réponse ; sinon, repli sur le RAG classique.
MAX_AGENT_STEPS = 7

AGENT_SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire pour techniciens.
Tu disposes d'outils de recherche dans les documents indexés :
- search_text : passages de texte (explications, définitions, procédures) ;
- search_tables : tableaux (valeurs chiffrées, comparaisons, résultats, hyperparamètres) ;
- search_figures : figures et schémas (architecture, diagrammes, photos).

Règles :
1. Appelle TOUJOURS au moins un outil avant de répondre à une question sur les documents.
   Choisis l'outil adapté ; combine-les si la question mélange texte, chiffres et figures.
   Formule la requête de recherche en anglais si le document est en anglais.
2. Réponds uniquement à partir des résultats des outils et cite chaque information
   avec le numéro du passage entre crochets, placé après la phrase.
   Exemple : « Le modèle atteint 28.4 BLEU [1]. » (et non « [28.4] »).
3. Si les résultats ne permettent pas de répondre, dis-le clairement ; n'invente rien.
4. Réponds dans la langue de la question, de façon concise et structurée."""

# Passages retrouvés pendant la question en cours (isolés par requête).
_collected: ContextVar[list[Hit] | None] = ContextVar("_collected", default=None)


def _register(hits: list[Hit]) -> str:
    collected = _collected.get()
    if collected is None:
        collected = []
        _collected.set(collected)
    if not hits:
        return "Aucun résultat pertinent."
    blocks = []
    for hit in hits:
        collected.append(hit)
        blocks.append(f"[{len(collected)}] Source : {hit.reference}\n{hit.text.strip()}")
    return "\n\n---\n\n".join(blocks)


@tool("search_text")
def search_text_tool(query: str) -> str:
    """Recherche des passages de texte pertinents dans les documents indexés."""
    return _register(search_text(query, k=4))


@tool("search_tables")
def search_tables_tool(query: str) -> str:
    """Recherche des tableaux (chiffres, scores, comparaisons, hyperparamètres)."""
    return _register(search_tables(query, k=2))


@tool("search_figures")
def search_figures_tool(query: str) -> str:
    """Recherche des figures / schémas / images et renvoie leur description."""
    return _register(search_images(query, k=2))


TOOLS = [search_text_tool, search_tables_tool, search_figures_tool]


RETRY_HINT = (
    "Ta réponse précédente était vide ou mal formée. Appelle maintenant un des outils "
    "search_text, search_tables ou search_figures avec une requête courte en anglais."
)


@wrap_model_call
def retry_empty_response(request: ModelRequest, handler) -> ModelResponse:
    """Relance le modèle quand il renvoie une réponse vide.

    Les petits modèles produisent parfois un appel d'outil mal formé, rejeté
    par Ollama : le message obtenu n'a alors ni contenu ni appel d'outil. On
    relance en ajoutant une consigne explicite.
    """
    response = handler(request)
    for _ in range(2):
        message = response.result[-1] if response.result else None
        if isinstance(message, AIMessage) and (message.tool_calls or str(message.content).strip()):
            break
        response = handler(request.override(messages=[*request.messages, HumanMessage(RETRY_HINT)]))
    return response


class MaintiDocAgent:
    def __init__(self):
        self.graph = create_agent(
            model=get_llm(),
            tools=TOOLS,
            system_prompt=AGENT_SYSTEM_PROMPT,
            middleware=[retry_empty_response],
        )
        self.history: list[BaseMessage] = []

    def reset(self) -> None:
        self.history = []

    def ask(self, question: str) -> Answer:
        collected: list[Hit] = []
        token = _collected.set(collected)
        try:
            messages = self.history[-MAX_HISTORY_MESSAGES:] + [HumanMessage(question)]
            limit_reached = False
            try:
                result = self.graph.invoke({"messages": messages}, {"recursion_limit": MAX_AGENT_STEPS})
            except GraphRecursionError:
                result, limit_reached = {"messages": messages}, True  # trop d'étapes : repli ci-dessous
        finally:
            _collected.reset(token)

        new_messages = result["messages"][len(messages):]
        tool_calls = [
            f"{call['name']}({call['args'].get('query', '')!r})"
            for msg in new_messages if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]
        final = next((m for m in reversed(new_messages) if isinstance(m, AIMessage) and m.content), None)

        if not tool_calls or final is None:
            # Repli : le petit modèle n'a pas utilisé d'outil -> RAG classique.
            answer = answer_with_rag(question)
            reason = "limite d'étapes de l'agent atteinte" if limit_reached else "aucun outil utilisé"
            answer.tool_calls = [f"(repli : {reason}) recherche globale texte + tableaux + images"]
        else:
            answer = Answer(text=str(final.content), sources=collected, tool_calls=tool_calls)

        self.history += [HumanMessage(question), AIMessage(answer.text)]
        return answer

