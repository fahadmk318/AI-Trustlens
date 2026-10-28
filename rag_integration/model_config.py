"""LLM model configuration for the AI TrustLens integration (new file).

Policy:
- Deployment / GitHub default: ``qwen3:8b`` (never replaced as the production
  default; the deployment machine must be able to run it).
- Local development / testing: any model the host can run, selected through
  the ``OLLAMA_MODEL`` environment variable, e.g.
      OLLAMA_MODEL=qwen3:4b-q4_K_M

The environment variable is read from the process environment only in this
module; teammates' pre-existing files (app.py, claimextractor.py) keep their
own on-disk defaults untouched:

- app.py: query_model(question, model_name=DEFAULT_MODEL) accepts an explicit
  model name; the integration adapter passes the configured model.
- claimextractor.py: extraction functions read the module constant MODEL at
  call time; rag_integration.claim_processor temporarily patches that constant
  at runtime (never in the file) so tests can drive a different local model.

Note: a configuration does NOT guarantee the target machine can run the model.
qwen3:8b has NOT been verified on the local development PC.
"""
from __future__ import annotations

import os

OLLAMA_MODEL_ENV = "OLLAMA_MODEL"
DEPLOYMENT_DEFAULT_MODEL = "qwen3:8b"


def get_ollama_model() -> str:
    """Resolve the active Ollama model.

    Returns the value of the OLLAMA_MODEL environment variable when it is set
    to a non-empty value; otherwise the deployment default ``qwen3:8b``.
    Never hard-codes a local-only model name.
    """
    value = os.getenv(OLLAMA_MODEL_ENV)
    if value is not None and value.strip():
        return value.strip()
    return DEPLOYMENT_DEFAULT_MODEL


def is_deployment_default(model_name: str) -> bool:
    return model_name == DEPLOYMENT_DEFAULT_MODEL
