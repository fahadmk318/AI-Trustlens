"""HTTP client for the AI TrustLens RAG evidence-retrieval API.

Talks to the existing FastAPI service in the RAG subsystem:

    POST /retrieve        {claim, top_k, retrieval_mode, domain|filters, graph_max_depth}
    POST /retrieve/batch  {claims: [...], top_k, retrieval_mode, domain|filters}
    GET  /health

Contract notes (from the RAG subsystem's own schemas):
- The service returns EVIDENCE ONLY. It never returns verification labels,
  trust scores, or generated explanations.
- `retrieval_score` is a ranking signal (reciprocal-rank-fusion), NOT a truth
  probability. It is passed through unchanged; never invented.
- Missing metadata stays null on the server side and is preserved as null here.

Failures are surfaced as typed exceptions; a connection failure or HTTP error is
NEVER silently converted into a successful empty-evidence result.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import httpx

from .config import IntegrationConfig

# Keys the service is guaranteed to provide for each evidence item.
_REQUIRED_EVIDENCE_FIELDS = ("chunk_id", "document_id", "text")


class RAGClientError(Exception):
    """Base class for all RAG API client failures."""

    def __init__(self, message: str, detail: Any = None):
        super().__init__(message)
        self.detail = detail


class RAGConnectionError(RAGClientError):
    """Could not connect to / reach the RAG service at all."""


class RAGTimeoutError(RAGClientError):
    """The RAG service accepted the connection but did not answer in time."""


class RAGHTTPError(RAGClientError):
    """The RAG service returned a non-2xx HTTP response (e.g. 422/500)."""

    def __init__(self, status_code: int, url: str, detail: Any = None):
        message = f"RAG API returned HTTP {status_code} for {url}"
        super().__init__(message, detail)
        self.status_code = status_code


class RAGInvalidResponseError(RAGClientError):
    """The RAG service responded with a structure that violates the contract."""


def _validate_retrieve_response(payload: Any) -> Dict[str, Any]:
    """Validate a single claim-retrieval response against the service contract.

    Raises RAGInvalidResponseError for structural violations. Does NOT
    normalise, default-invent or drop anything: evidence fields that exist are
    preserved exactly, absent optional fields stay absent.
    """
    if not isinstance(payload, dict):
        raise RAGInvalidResponseError(
            f"expected a JSON object response, got {type(payload).__name__}", payload)
    if "evidence" not in payload or not isinstance(payload["evidence"], list):
        raise RAGInvalidResponseError(
            "response is missing the 'evidence' list", payload)
    if not isinstance(payload.get("claim"), str):
        raise RAGInvalidResponseError(
            "response is missing the string 'claim' field", payload)
    if "evidence_found" not in payload:
        raise RAGInvalidResponseError(
            "response is missing the 'evidence_found' flag", payload)
    for index, item in enumerate(payload["evidence"]):
        if not isinstance(item, dict):
            raise RAGInvalidResponseError(
                f"evidence[{index}] is not an object", item)
        missing = [k for k in _REQUIRED_EVIDENCE_FIELDS if not isinstance(item.get(k), str)]
        if missing:
            raise RAGInvalidResponseError(
                f"evidence[{index}] is missing/malformed required fields: {missing}", item)
    return payload


def _validate_batch_response(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
        raise RAGInvalidResponseError(
            "batch response is missing the 'results' list", payload)
    mode = payload.get("retrieval_mode")
    if not isinstance(mode, str):
        raise RAGInvalidResponseError(
            "batch response is missing the string 'retrieval_mode' field", payload)
    return [_validate_retrieve_response(item) for item in payload["results"]]


class RAGClient:
    """Small synchronous client for the RAG evidence-retrieval API."""

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        transport: Optional[httpx.BaseTransport] = None,
    ):
        self.config = config or IntegrationConfig()
        self._transport = transport  # test seam; production client uses None
        self._client: Optional[httpx.Client] = None

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "RAGClient":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    def open(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout,
                    read=self.config.read_timeout,
                    write=self.config.read_timeout,
                    pool=self.config.connect_timeout,
                ),
                transport=self._transport,
                headers={"Accept": "application/json"},
            )
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    # -- requests ------------------------------------------------------------

    def _post_json(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{self.config.base_url}{path}"
        try:
            response = self.open().post(path, json=body)
        except httpx.TimeoutException as exc:
            raise RAGTimeoutError(
                f"RAG API did not answer in time: {url}") from exc
        except httpx.HTTPError as exc:  # ConnectError, NetworkError, ...
            raise RAGConnectionError(
                f"could not reach the RAG API at {url}: "
                f"{exc.__class__.__name__}: {exc}") from exc

        if response.status_code >= 400:
            try:
                detail = response.json()
            except ValueError:
                detail = response.text
            raise RAGHTTPError(response.status_code, url, detail)
        try:
            return response.json()
        except ValueError as exc:
            raise RAGInvalidResponseError(
                f"RAG API returned a non-JSON body for {url}") from exc

    def health(self) -> Dict[str, Any]:
        url = f"{self.config.base_url}/health"
        try:
            response = self.open().get("/health")
        except httpx.TimeoutException as exc:
            raise RAGTimeoutError(f"RAG API health check timed out: {url}") from exc
        except httpx.HTTPError as exc:
            raise RAGConnectionError(
                f"could not reach the RAG API at {url}: "
                f"{exc.__class__.__name__}: {exc}") from exc
        if response.status_code >= 400:
            raise RAGHTTPError(response.status_code, url, response.text)
        try:
            payload = response.json()
        except ValueError as exc:
            raise RAGInvalidResponseError(
                f"RAG API returned a non-JSON /health body") from exc
        if not isinstance(payload, dict) or "status" not in payload:
            raise RAGInvalidResponseError(
                "health response is missing the 'status' field", payload)
        return payload

    def retrieve(
        self,
        claim: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
        graph_max_depth: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Retrieve evidence for ONE claim. Returns the validated response."""
        if not isinstance(claim, str) or not claim.strip():
            raise ValueError("claim must be a non-empty string")
        body: Dict[str, Any] = {
            "claim": claim,
            "top_k": top_k if top_k is not None else self.config.top_k,
            "retrieval_mode": retrieval_mode or self.config.retrieval_mode,
        }
        effective_domain = domain if domain is not None else self.config.domain
        if effective_domain is not None:
            body["domain"] = effective_domain
        if graph_max_depth is not None:
            body["graph_max_depth"] = graph_max_depth
        payload = self._post_json("/retrieve", body)
        return _validate_retrieve_response(payload)

    def retrieve_batch(
        self,
        claims: Iterable[str],
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve evidence for a LIST of claims (order-preserving).

        The service isolates per-claim failures itself: a failing claim comes
        back with `diagnostics.per_claim_failure` rather than dropping the rest.
        """
        claim_list = list(claims)
        if not claim_list:
            raise ValueError("claims list must not be empty")
        if not all(isinstance(c, str) and c.strip() for c in claim_list):
            raise ValueError("claims must all be non-empty strings")
        limit = self.config.batch_max_claims
        if len(claim_list) > limit:
            raise ValueError(
                f"batch size {len(claim_list)} exceeds the API limit of {limit}")
        body: Dict[str, Any] = {
            "claims": claim_list,
            "top_k": top_k if top_k is not None else self.config.top_k,
            "retrieval_mode": retrieval_mode or self.config.retrieval_mode,
        }
        effective_domain = domain if domain is not None else self.config.domain
        if effective_domain is not None:
            body["domain"] = effective_domain
        payload = self._post_json("/retrieve/batch", body)
        validated = _validate_batch_response(payload)
        if len(validated) != len(claim_list):
            raise RAGInvalidResponseError(
                f"batch returned {len(validated)} results for "
                f"{len(claim_list)} claims", payload)
        return validated

    def retrieve_for_claims(
        self,
        claims: List[str],
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
        graph_max_depth: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Convenience: batch endpoint for lists, single endpoint for one claim."""
        if len(claims) <= 1:
            result = self.retrieve(
                claims[0] if claims else "",
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                domain=domain,
                graph_max_depth=graph_max_depth,
            )
            return [result]
        return self.retrieve_batch(
            claims, top_k=top_k, retrieval_mode=retrieval_mode, domain=domain)
