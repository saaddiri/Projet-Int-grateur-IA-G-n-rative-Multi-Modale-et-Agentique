"""Pipeline RAG classique (non agentique) : recherche -> contexte -> LLM.

Sert de référence et de repli lorsque l'agent n'appelle aucun outil.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from .config import settings
from .retrieval import Hit, format_hits, search_all

RAG_SYSTEM_PROMPT = """Tu es MaintiDoc, un assistant documentaire technique.
Réponds UNIQUEMENT à partir du contexte fourni. Chaque affirmation doit citer
sa source avec le numéro du passage entre crochets, placé après la phrase
(exemple : « Le modèle atteint 28.4 BLEU [1]. »).
Si le contexte ne permet pas de répondre, dis-le clairement au lieu d'inventer.
Réponds dans la langue de la question."""


@dataclass
class Answer:
    text: str
    sources: list[Hit] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    return ChatOllama(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_ctx=8192,
    )


def answer_with_rag(question: str, k: int = 6) -> Answer:
    hits = search_all(question, k=k)
    messages = [
        SystemMessage(RAG_SYSTEM_PROMPT),
        HumanMessage(f"Contexte :\n{format_hits(hits)}\n\nQuestion : {question}"),
    ]
    response = get_llm().invoke(messages)
    return Answer(text=str(response.content), sources=hits)
