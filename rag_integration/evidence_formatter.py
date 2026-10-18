"""Convert retrieval results into the structured claim-evidence output.

The RAG service returns evidence only (never verification labels), so this
formatter keeps retrieval and verification STRICTLY separated:

- Evidence text, sources, metadata and scores are passed through verbatim from
  the actual retrieval response. Nothing is invented or synthesised.
- `retrieval_status` distinguishes three cases:
      "ok"             successful retrieval that found evidence,
      "no_evidence"    successful retrieval that found nothing relevant,
      "retrieval_error" the retrieval system itself failed for this claim.
- `verification_status` is ALWAYS "Not Evaluated" unless a genuine verification
  component is invoked. A retrieved passage — however similar to the claim — is
  resemblance, not verification. No Supported/Refuted/Conflicting Evidence/
  Not Enough Evidence label is ever derived from retrieval ranks or scores.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

VERIFICATION_PENDING = "Not Evaluated"
STATUS_OK = "ok"
STATUS_NO_EVIDENCE = "no_evidence"
STATUS_RETRIEVAL_ERROR = "retrieval_error"

# Final verification labels for the future verification stage (not derived here).
FUTURE_VERIFICATION_STATUSES = (
    "Supported",
    "Refuted",
    "Conflicting Evidence",
    "Not Enough Evidence",
)

VERIFICATION_PENDING_NOTE = (
    "Evidence retrieval is decoupled from factual verification. The retrieved "
    "passages are candidates for a separate verification/classification stage, "
    "which will assign Supported / Refuted / Conflicting Evidence / "
    "Not Enough Evidence. No classifier was invoked here, so no verification "
    "label exists for this claim yet."
)


def _differentiate_retrieval_status(
    response: Optional[Dict[str, Any]],
    retrieval_error: Optional[str],
) -> str:
    """Decide ok / no_evidence / retrieval_error from the actual response.

    A backend failure is reported by the service either as a per-claim
    error (`diagnostics.error`, e.g. from /retrieve/batch) or via
    `diagnostics.per_method[].status == "error"`; it must never be mistaken
    for "no evidence exists".
    """
    if retrieval_error:
        return STATUS_RETRIEVAL_ERROR
    if not isinstance(response, dict):
        return STATUS_RETRIEVAL_ERROR
    diagnostics = response.get("diagnostics")
    if isinstance(diagnostics, dict) and diagnostics.get("error"):
        return STATUS_RETRIEVAL_ERROR
    per_method = diagnostics.get("per_method") if isinstance(diagnostics, dict) else None
    if isinstance(per_method, list) and per_method:
        statuses = [
            m.get("status") for m in per_method
            if isinstance(m, dict) and m.get("status") is not None
        ]
        if statuses and all(s == "error" for s in statuses):
            # Every backend the mode requested failed: a system failure, not
            # an empty result.
            return STATUS_RETRIEVAL_ERROR
    if response.get("evidence"):
        return STATUS_OK
    return STATUS_NO_EVIDENCE


def format_evidence_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """Pass one evidence record through, preserving text, source and metadata."""
    metadata_keys = (
        "chunk_id",
        "document_id",
        "title",
        "source_name",
        "source_url",
        "publication_date",
        "page_number",
        "domain",
        "graph_path",
    )
    methods = item.get("retrieval_methods")
    source = " > ".join(
        part for part in (
            item.get("source_name"),
            item.get("title"),
        ) if isinstance(part, str) and part.strip()
    ) or None
    return {
        "text": item.get("text", ""),
        "source": source,
        "metadata": {key: item.get(key) for key in metadata_keys},
        "retrieval_score": item.get("retrieval_score"),
        "retrieval_method": (
            list(methods) if isinstance(methods, list) and methods
            else None
        ),
    }


def _server_error_message(response: Dict[str, Any]) -> Optional[str]:
    """Extract the service's own error explanation for diagnostics purposes."""
    diagnostics = response.get("diagnostics")
    if isinstance(diagnostics, dict):
        top_error = diagnostics.get("error")
        if isinstance(top_error, str) and top_error.strip():
            return top_error
        per_method = diagnostics.get("per_method")
        if isinstance(per_method, list):
            method_errors = [
                f"{m.get('method', '?')}: {m.get('error')}"
                for m in per_method
                if isinstance(m, dict) and m.get("status") == "error"
                and isinstance(m.get("error"), str) and m["error"].strip()
            ]
            if method_errors:
                return "; ".join(method_errors)
    return None


def format_claim_evidence(
    claim: str,
    response: Optional[Dict[str, Any]],
    retrieval_error: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the structured per-claim output.

    Parameters
    ----------
    claim: the original extracted claim text.
    response: the validated RAG API response for this claim (or None when the
        retrieval call itself failed).
    retrieval_error: message describing why retrieval failed, if it failed.
    """
    if retrieval_error is not None:
        return {
            "claim": claim,
            "evidence": [],
            "evidence_found": False,
            "retrieval_status": STATUS_RETRIEVAL_ERROR,
            "retrieval_error": retrieval_error,
            "verification_status": VERIFICATION_PENDING,
            "verification_explanation": None,
        }

    retrieval_status = _differentiate_retrieval_status(response, None)
    if retrieval_status == STATUS_RETRIEVAL_ERROR and isinstance(response, dict):
        retrieval_error = (
            retrieval_error
            or _server_error_message(response)
            or "retrieval system reported an error for every requested method"
        )
    evidence_list: List[Dict[str, Any]] = []
    if retrieval_status == STATUS_OK:
        for item in response.get("evidence", []):
            if not isinstance(item, dict):
                # Malformed records are skipped and reported, never invented.
                continue
            if not isinstance(item.get("text"), str):
                continue
            evidence_list.append(format_evidence_item(item))

    if retrieval_status == STATUS_OK and not evidence_list:
        retrieval_status = STATUS_NO_EVIDENCE

    diagnostics = {}
    if isinstance(response, dict) and isinstance(response.get("diagnostics"), dict):
        diagnostics = response["diagnostics"]

    return {
        "claim": claim,
        "evidence": evidence_list,
        "evidence_found": bool(evidence_list),
        "retrieval_status": retrieval_status,
        "retrieval_mode": response.get("retrieval_mode") if isinstance(response, dict) else None,
        "retrieval_error": retrieval_error,
        "latency_ms": response.get("latency_ms") if isinstance(response, dict) else None,
        "diagnostics": diagnostics,
        "verification_status": VERIFICATION_PENDING,
        "verification_explanation": None,
    }
