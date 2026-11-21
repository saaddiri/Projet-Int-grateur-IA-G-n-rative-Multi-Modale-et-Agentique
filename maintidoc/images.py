"""Traitement des images : encodeur CLIP et descriptions générées par un VLM.

- CLIP projette images et textes dans le même espace vectoriel : on peut
  retrouver une figure à partir d'une question en langage naturel.
- Un modèle vision-langage (Qwen2.5-VL via Ollama) produit une description
  textuelle de chaque image, indexée ensuite comme un passage ordinaire.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

import ollama
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .config import settings

DESCRIBE_PROMPT = (
    "You are analysing a figure extracted from a technical document. "
    "Describe precisely what the figure shows: components, labels, arrows, "
    "flows, axes and any visible text. Be factual, do not invent details. "
    "Answer in 5 to 10 sentences."
)
VLM_MAX_SIDE = 672  # côté maximal (px) des images envoyées au VLM


class ClipEncoder:
    """Encodeur CLIP (images et textes) — vecteurs normalisés L2."""

    def __init__(self, model_name: str = settings.clip_model):
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

    @property
    def dim(self) -> int:
        return self.model.config.projection_dim

    @staticmethod
    def _features(output) -> torch.Tensor:
        # Selon la version de transformers, get_*_features renvoie un tenseur
        # ou un objet contenant pooler_output.
        if isinstance(output, torch.Tensor):
            return output
        return output.pooler_output

    @torch.no_grad()
    def encode_images(self, paths: list[Path]) -> list[list[float]]:
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt")
        feats = self._features(self.model.get_image_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats.tolist()

    @torch.no_grad()
    def encode_text(self, text: str) -> list[float]:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77
        )
        feats = self._features(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, dim=-1)
        return feats[0].tolist()


@lru_cache(maxsize=1)
def get_clip() -> ClipEncoder:
    return ClipEncoder()


def _encode_for_vlm(path: Path, max_side: int = VLM_MAX_SIDE) -> str:
    """Réduit l'image avant l'envoi au VLM.

    Le nombre de tokens visuels croît avec la surface de l'image : une figure
    pleine page (~1500x2200 px) dépasse la fenêtre de contexte et rend le
    calcul très lent sur CPU.
    """
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def describe_image(path: Path, caption: str = "") -> str:
    """Description textuelle d'une image par le VLM (Ollama)."""
    prompt = DESCRIBE_PROMPT
    if caption:
        prompt += f"\nThe caption printed under the figure is: \"{caption}\"."
    client = ollama.Client(host=settings.ollama_base_url)
    response = client.chat(
        model=settings.vlm_model,
        messages=[{"role": "user", "content": prompt, "images": [_encode_for_vlm(Path(path))]}],
        options={"temperature": 0.1, "num_predict": 350, "num_ctx": 4096},
    )
    return response["message"]["content"].strip()
