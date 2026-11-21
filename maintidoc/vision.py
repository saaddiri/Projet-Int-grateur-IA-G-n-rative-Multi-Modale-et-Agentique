"""Analyse de photographies d'équipement par un VLM (livrable L2.1).

Le VLM (Qwen2.5-VL via Ollama) décrit la photo envoyée par le technicien et
renvoie une sortie structurée (Pydantic) : description, textes visibles,
composants, voyants, requête de recherche et niveau de confiance. Cette analyse
alimente ensuite la recherche documentaire multimodale.
"""

from __future__ import annotations

import base64
import io
from typing import Literal

import ollama
from PIL import Image
from pydantic import BaseModel, Field, ValidationError

from .config import settings
from .images import VLM_MAX_SIDE


class PhotoAnalysis(BaseModel):
    description: str = Field(description="Description factuelle de ce que montre la photo")
    visible_text: list[str] = Field(default_factory=list, description="Textes, étiquettes, codes lisibles")
    components: list[str] = Field(default_factory=list, description="Composants ou éléments identifiés")
    indicators: list[str] = Field(default_factory=list, description="Voyants, couleurs, états visibles")
    search_query: str = Field(description="Requête courte en anglais pour chercher dans la documentation")
    confidence: Literal["faible", "moyenne", "élevée"] = Field(
        description="Confiance dans l'identification de ce que montre la photo"
    )


VISION_PROMPT = """Tu analyses une photo envoyée par un technicien de maintenance.
Question du technicien : « {question} »

Décris uniquement ce qui est réellement visible : composants, schéma, étiquettes,
textes lisibles, voyants et leur couleur. Description en 3 phrases maximum, sans
répéter les étiquettes ; au plus 10 éléments par liste. N'invente rien ; si l'image est floue ou
ambiguë, indique une confiance faible. Propose une requête de recherche courte
en anglais pour retrouver la documentation pertinente.
Réponds en JSON conforme au schéma demandé."""


def _image_to_base64(image: Image.Image, max_side: int = VLM_MAX_SIDE) -> str:
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def analyze_photo(image: Image.Image, question: str = "") -> PhotoAnalysis:
    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.vlm_model,
        messages=[{
            "role": "user",
            "content": VISION_PROMPT.format(question=question or "(aucune question, analyser la photo)"),
            "images": [_image_to_base64(image)],
        }],
        format=PhotoAnalysis.model_json_schema(),
        options={"temperature": 0.1, "num_predict": 900, "num_ctx": 4096},
    )
    content = response["message"]["content"]
    try:
        return PhotoAnalysis.model_validate_json(content)
    except ValidationError:
        # Sortie tronquée ou invalide : on conserve le texte brut avec une confiance faible
        # plutôt que d'interrompre le pipeline.
        return PhotoAnalysis(
            description=content[:600],
            search_query=question or "equipment photo",
            confidence="faible",
        )
