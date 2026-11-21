"""Configuration centralisée, lue depuis les variables d'environnement (.env)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    # PostgreSQL / PGVector
    pg_host: str = _env("PG_HOST", "localhost")
    pg_port: int = int(_env("PG_PORT", "5433"))
    pg_user: str = _env("PG_USER", "maintidoc")
    pg_password: str = _env("PG_PASSWORD", "maintidoc")
    pg_database: str = _env("PG_DATABASE", "maintidoc")

    # Tables PGVector (LlamaIndex préfixe automatiquement par "data_")
    text_table: str = "maintidoc_text"
    image_table: str = "maintidoc_images"

    # Ollama
    ollama_base_url: str = _env("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = _env("LLM_MODEL", "qwen2.5:3b")
    vlm_model: str = _env("VLM_MODEL", "qwen2.5vl:3b")

    # Embeddings
    text_embed_model: str = _env("TEXT_EMBED_MODEL", "intfloat/multilingual-e5-small")
    clip_model: str = _env("CLIP_MODEL", "openai/clip-vit-base-patch32")

    # Découpage
    chunk_size: int = int(_env("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "64"))

    describe_images: bool = _env("DESCRIBE_IMAGES", "1") == "1"

    # Stockage local des images extraites (servies dans l'interface)
    storage_dir: Path = field(default_factory=lambda: ROOT_DIR / "storage")

    @property
    def images_dir(self) -> Path:
        path = self.storage_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
