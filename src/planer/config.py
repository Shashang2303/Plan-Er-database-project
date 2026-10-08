"""Runtime settings, read from environment variables or a local .env file (NFR-6).

Nothing in this module is ever written to run_log, so credentials cannot leak into logs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # python-dotenv is optional
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    mongo_uri: str = field(default_factory=lambda: _env("MONGO_URI", "mongodb://localhost:27018/?directConnection=true"))
    mongo_db: str = field(default_factory=lambda: _env("MONGO_DB", "planer"))
    search_backend: str = field(default_factory=lambda: _env("SEARCH_BACKEND", "mongot").lower())
    atlas_uri: str = field(default_factory=lambda: _env("ATLAS_URI", ""))
    ollama_url: str = field(default_factory=lambda: _env("OLLAMA_URL", "http://localhost:11434").rstrip("/"))
    llm_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(m.strip() for m in _env("LLM_MODELS", "qwen2.5:1.5b").split(",") if m.strip())
    )
    llm_timeout_s: float = field(default_factory=lambda: float(_env("LLM_TIMEOUT_S", "120")))
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    usd_per_compute_second: float = field(default_factory=lambda: float(_env("USD_PER_COMPUTE_SECOND", "0.0001")))
    usd_per_label: float = field(default_factory=lambda: float(_env("USD_PER_LABEL", "0.02")))
    data_dir: Path = field(default_factory=lambda: Path(_env("PLANER_DATA_DIR", str(PROJECT_ROOT / "data" / "raw"))))

    def __post_init__(self) -> None:
        if self.search_backend not in {"mongot", "atlas", "local"}:
            raise ValueError(f"SEARCH_BACKEND must be mongot, atlas or local, not {self.search_backend!r}")
        if self.search_backend == "atlas" and not self.atlas_uri:
            raise ValueError("SEARCH_BACKEND=atlas requires ATLAS_URI")


settings = Settings()

EMBED_DIM = 384
