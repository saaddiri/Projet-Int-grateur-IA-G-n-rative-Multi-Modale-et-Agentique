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
    # Encodeur de texte multilingue aligné sur l'espace CLIP ViT-B/32 (questions en français)
    mclip_text_model: str = _env("MCLIP_TEXT_MODEL", "sentence-transformers/clip-ViT-B-32-multilingual-v1")
    # Reranker : vide = adapté s'il existe sinon initial ; "none", "base", "adapted" ou un chemin
    reranker: str = _env("RERANKER", "")

    # Découpage
    chunk_size: int = int(_env("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(_env("CHUNK_OVERLAP", "64"))

    describe_images: bool = _env("DESCRIBE_IMAGES", "1") == "1"

    # Étape 3 : service de maintenance simulé (MCP) et approbation humaine
    # Secret partagé entre l'application (émission des jetons) et le serveur MCP (vérification).
    approval_secret: str = _env("APPROVAL_SECRET", "dev-only-change-me")
    # Clé API de l'utilisateur courant, transmise au serveur MCP (identité côté serveur)
    mcp_api_key: str = _env("MAINTIDOC_API_KEY", "demo-key-dupont")
    # Étape 4 : clé de service du backend auprès du serveur MCP (délégation d'identité),
    # secret partagé bot -> backend, jeton du bot Telegram, limites d'appels
    mcp_service_key: str = _env("MCP_SERVICE_KEY", "")
    bot_shared_secret: str = _env("BOT_SHARED_SECRET", "")
    telegram_bot_token: str = _env("TELEGRAM_BOT_TOKEN", "")
    api_base_url: str = _env("MAINTIDOC_API_URL", "http://localhost:8000")
    rate_limit_per_minute: int = int(_env("RATE_LIMIT_PER_MINUTE", "20"))
    llm_rate_limit_per_minute: int = int(_env("LLM_RATE_LIMIT_PER_MINUTE", "4"))
    maintenance_seed: Path = field(default_factory=lambda: ROOT_DIR / "data" / "maintenance" / "seed.json")

    # Stockage local des images extraites (servies dans l'interface)
    storage_dir: Path = field(default_factory=lambda: ROOT_DIR / "storage")

    @property
    def maintenance_db(self) -> Path:
        path = Path(_env("MAINTENANCE_DB", str(self.storage_dir / "maintenance.db")))
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def images_dir(self) -> Path:
        path = self.storage_dir / "images"
        path.mkdir(parents=True, exist_ok=True)
        return path


settings = Settings()
