"""AI TrustLens RAG integration: connects the Stage-1 QA/claim-extractor
application to the existing RAG evidence-retrieval service over HTTP.

Components:
- config:            environment-backed settings (RAG_API_BASE_URL etc.)
- rag_client:        typed HTTP client for POST /retrieve, /retrieve/batch, GET /health
- claim_processor:   adapter around the untouched claimextractor.py
- evidence_formatter: structured per-claim evidence output; verification stays
                     pending (never derived from retrieval scores)
- pipeline:          end-to-end orchestration

Retrieval is NOT verification: no Supported/Refuted/Conflicting Evidence/Not
Enough Evidence label is produced by this package.
"""
from .config import IntegrationConfig, get_config
from .claim_processor import ClaimProcessor
from .model_config import (
    DEPLOYMENT_DEFAULT_MODEL,
    OLLAMA_MODEL_ENV,
    get_ollama_model,
)
from .evidence_formatter import (
    STATUS_NO_EVIDENCE,
    STATUS_OK,
    STATUS_RETRIEVAL_ERROR,
    VERIFICATION_PENDING,
    format_claim_evidence,
)
from .pipeline import (
    BaselineAnswerError,
    TrustLensRAGPipeline,
    baseline_answer_provider,
)
from .rag_client import (
    RAGClient,
    RAGClientError,
    RAGConnectionError,
    RAGHTTPError,
    RAGInvalidResponseError,
    RAGTimeoutError,
)

__all__ = [
    "IntegrationConfig",
    "get_config",
    "ClaimProcessor",
    "DEPLOYMENT_DEFAULT_MODEL",
    "OLLAMA_MODEL_ENV",
    "get_ollama_model",
    "STATUS_NO_EVIDENCE",
    "STATUS_OK",
    "STATUS_RETRIEVAL_ERROR",
    "VERIFICATION_PENDING",
    "format_claim_evidence",
    "BaselineAnswerError",
    "TrustLensRAGPipeline",
    "baseline_answer_provider",
    "RAGClient",
    "RAGClientError",
    "RAGConnectionError",
    "RAGHTTPError",
    "RAGInvalidResponseError",
    "RAGTimeoutError",
]
