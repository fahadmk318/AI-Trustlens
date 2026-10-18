"""Configuration for the AI TrustLens RAG integration package.

All settings are read lazily from environment variables so changes take
effect without re-importing. Defaults match the RAG subsystem's own
configuration (API_HOST=127.0.0.1, API_PORT=8000, hybrid mode, top_k=5).

The RAG service runs as a separate process (its own virtualenv), so the
integration only talks to it over HTTP; no heavy retrieval dependencies
(torch, sentence-transformers, qdrant-client...) are required here.
"""
from __future__ import annotations

import os

BASE_URL_ENV = "RAG_API_BASE_URL"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"


def _getenv_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value.strip() != "" else default


def _getenv_int(name: str, default: int) -> int:
    try:
        return int(_getenv_str(name, str(default)))
    except (TypeError, ValueError):
        return default


def _getenv_float(name: str, default: float) -> float:
    try:
        return float(_getenv_str(name, str(default)))
    except (TypeError, ValueError):
        return default


class IntegrationConfig:
    """Environment-backed integration settings (attribute access reads env)."""

    @property
    def base_url(self) -> str:
        return _getenv_str(BASE_URL_ENV, DEFAULT_BASE_URL).rstrip("/")

    @property
    def connect_timeout(self) -> float:
        return _getenv_float("RAG_API_CONNECT_TIMEOUT", 5.0)

    @property
    def read_timeout(self) -> float:
        return _getenv_float("RAG_API_READ_TIMEOUT", 60.0)

    @property
    def top_k(self) -> int:
        return _getenv_int("RAG_API_TOP_K", 5)

    @property
    def retrieval_mode(self) -> str:
        return _getenv_str("RAG_API_RETRIEVAL_MODE", "hybrid")

    @property
    def domain(self) -> str | None:
        value = os.getenv("RAG_API_DOMAIN")
        return value if value is not None and value.strip() != "" else None

    @property
    def refine_claims(self) -> bool:
        return _getenv_str("TRUSTLENS_REFINE_CLAIMS", "0") in ("1", "true", "True", "yes")

    @property
    def batch_max_claims(self) -> int:
        return _getenv_int("RAG_API_BATCH_MAX_CLAIMS", 100)


def get_config() -> IntegrationConfig:
    """Return a fresh settings instance (re-evaluates environment changes)."""
    return IntegrationConfig()
