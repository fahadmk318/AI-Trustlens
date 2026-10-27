"""Integration pipeline: question -> answer -> claims -> evidence -> structured output.

Flow (reuse-only, no re-implementation of any existing component):

    question
      -> (optional baseline answer via app.query_model adapter - Ollama)
      -> teammate's claimextractor.extract_medical_claims + split_compound_claims
      -> RAG subsystem POST /retrieve or /retrieve/batch (HTTP)
      -> evidence_formatter -> structured per-claim output

Retrieval and verification remain separate stages: this pipeline never assigns
Supported/Refuted/Conflicting Evidence/Not Enough Evidence. Per-claim failures
are reported in the output; one failing claim never drops the others.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

from .claim_processor import ClaimProcessor
from .config import IntegrationConfig
from .evidence_formatter import (
    VERIFICATION_PENDING,
    VERIFICATION_PENDING_NOTE,
    STATUS_RETRIEVAL_ERROR,
    format_claim_evidence,
)
from .rag_client import RAGClient, RAGClientError


# ---------------------------------------------------------------------------
# Baseline answer adapter (teammate's app.py is imported, never modified)
# ---------------------------------------------------------------------------

class BaselineAnswerError(RuntimeError):
    """Raised when the baseline QA model cannot produce an answer."""


def baseline_answer_provider(question: str) -> Tuple[Optional[str], Optional[str]]:
    """Obtain the baseline answer through the teammate's existing app.py.

    app.query_model is the real, untouched baseline function (Ollama). The
    model comes from rag_integration.model_config (OLLAMA_MODEL env; default
    qwen3:8b, the untouched on-disk deployment default), passed through the
    existing model_name parameter; app.py's file contents remain untouched.
    It returns None on any failure, so the failure is surfaced as an explicit
    error instead of being swallowed.
    """
    from .model_config import get_ollama_model

    try:
        import app  # teammate's live baseline module (kept untouched)
    except Exception as exc:  # ImportError or import-time failure
        return None, f"Could not import the baseline app module: {exc}"
    try:
        answer = app.query_model(question, model_name=get_ollama_model())
    except Exception as exc:  # defensive: query_model handles its own errors
        return None, f"Baseline answer generation failed: {exc}"
    if answer is None or not answer.strip():
        return None, (
            "The baseline model returned no answer. Check that Ollama is "
            "running and the configured model is available."
        )
    return answer, None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrustLensRAGPipeline:
    """Reusable claim-evidence retrieval pipeline."""

    def __init__(
        self,
        config: Optional[IntegrationConfig] = None,
        rag_client: Optional[RAGClient] = None,
        claim_processor: Optional[ClaimProcessor] = None,
        answer_provider: Optional[Callable[[str], Tuple[Optional[str], Optional[str]]]] = baseline_answer_provider,
        verifier: Optional[Any] = None,
    ):
        self.config = config or IntegrationConfig()
        self.rag_client = rag_client or RAGClient(self.config)
        self.claim_processor = claim_processor or ClaimProcessor(
            refine=self.config.refine_claims)
        self.answer_provider = (
            answer_provider if answer_provider is not None
            else baseline_answer_provider)
        self.verifier = verifier
        self._owns_client = rag_client is None

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "TrustLensRAGPipeline":
        self.rag_client.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._owns_client:
            self.rag_client.close()
        return False

    def close(self) -> None:
        if self._owns_client:
            self.rag_client.close()

    # -- stages --------------------------------------------------------------

    def process_answer(
        self,
        question: str,
        answer: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
        verify: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Full pipeline given the answer text (answer source irrelevant).

        The existing claim extractor is always applied to the answer; claims
        are never accepted from anywhere else, so the extractor is never
        bypassed.
        """
        errors: List[str] = []

        extraction = self.claim_processor.process(answer, source_text=answer)
        claims: List[Dict[str, Any]] = extraction["claims"]
        if extraction.get("error"):
            errors.append(f"claim extraction failed: {extraction['error']}")
            return self._empty_output(question, answer, errors, extraction)

        if not claims:
            errors.append("no usable claims were extracted from the answer")

        claim_texts = [
            claim.get("claim", "") if isinstance(claim, dict) else str(claim)
            for claim in claims
        ]
        retrievals = self._retrieve_all(
            claim_texts, top_k=top_k, retrieval_mode=retrieval_mode, domain=domain)

        results: List[Dict[str, Any]] = [
            self._format_claim(claim, retrieval, errors)
            for claim, retrieval in zip(claims, retrievals)
        ]

        output = {
            "question": question,
            "answer": answer,
            "claims": results,
            "extraction": {
                "error": extraction.get("error"),
                "duplicates": extraction.get("duplicates", {}),
                "notes": extraction.get("notes", []),
                "model_name": self.claim_processor.model_name,
            },
            "verification": {
                "status": VERIFICATION_PENDING,
                "stage": "pending: verification/classification is a separate later stage",
                "note": VERIFICATION_PENDING_NOTE,
            },
            "errors": errors,
        }

        should_verify = (
            verify is True
            or (verify is None and (self.verifier is not None or self.config.enable_verification))
        )

        if should_verify:
            from verification import MedicalClaimVerifier, verify_claims_pipeline

            # Attach structural validation to claims
            try:
                import claimextractor
                val_res = claimextractor.validate_extracted_claims(answer, {"claims": claims})
                struct_map = {
                    c.get("claim", ""): {
                        "status": c.get("validation", "PASSED"),
                        "issues": c.get("issues", []),
                        "is_structurally_valid": c.get("validation") == "PASSED",
                        "note": val_res.get("note", ""),
                    }
                    for c in val_res.get("claims", [])
                }
                for res_item in results:
                    c_text = res_item.get("claim", "")
                    if c_text in struct_map:
                        res_item["structural_validation"] = struct_map[c_text]
                        if isinstance(res_item.get("claim_meta"), dict):
                            res_item["claim_meta"]["structural_validation"] = struct_map[c_text]
            except Exception as exc:
                errors.append(f"structural validation error: {exc}")

            verifier = self.verifier or MedicalClaimVerifier(mode=self.config.verification_mode)
            output = verify_claims_pipeline(output, verifier=verifier)

        return output

    def process_question(
        self,
        question: str,
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
        verify: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Full pipeline starting from the question: generate the baseline
        answer through the untouched app.py adapter, then process it."""
        if not isinstance(question, str) or not question.strip():
            return {
                "question": question,
                "answer": None,
                "claims": [],
                "extraction": {"error": None, "duplicates": {}, "notes": []},
                "verification": {
                    "status": VERIFICATION_PENDING,
                    "stage": "pending",
                    "note": VERIFICATION_PENDING_NOTE,
                },
                "errors": ["question is empty"],
            }
        answer, error = self.answer_provider(question)
        if error or answer is None:
            errors = [f"baseline answer generation failed: {error}"]
            return self._empty_output(question, None, errors, {"error": None, "duplicates": {}, "notes": []})
        return self.process_answer(
            question, answer,
            top_k=top_k, retrieval_mode=retrieval_mode, domain=domain,
            verify=verify)

    # -- internals -----------------------------------------------------------

    def _retrieve_all(
        self,
        claim_texts: List[str],
        top_k: Optional[int] = None,
        retrieval_mode: Optional[str] = None,
        domain: Optional[str] = None,
    ) -> List[Any]:
        """Retrieve evidence for every claim; position i corresponds to claim i.

        Returns a list in which each element is either the validated RAG API
        response for that claim or an exception instance describing why that
        claim's retrieval failed. The batch endpoint is used for lists (order
        preserving, per-claim failure isolation server-side); requests larger
        than the API's batch limit are split; if a whole batch request fails,
        the claims in it are retried one by one so one HTTP failure never
        wipes out the whole group. An explicit graph_max_depth override is
        honoured by falling back to per-claim calls (the batch schema has no
        graph_max_depth field).
        """
        empty_claims = all(not claim.strip() for claim in claim_texts)
        if empty_claims:
            raise ValueError("claims must all be non-empty strings")
        retrievals: List[Any] = []
        limit = max(1, self.config.batch_max_claims)
        index = 0
        while index < len(claim_texts):
            chunk = claim_texts[index:index + limit]
            retrievals.extend(self._retrieve_chunk(
                chunk, top_k=top_k, retrieval_mode=retrieval_mode, domain=domain))
            index += limit
        return retrievals

    def _retrieve_chunk(
        self,
        claim_texts: List[str],
        top_k: Optional[int],
        retrieval_mode: Optional[str],
        domain: Optional[str],
    ) -> List[Any]:
        try:
            responses = self.rag_client.retrieve_for_claims(
                claim_texts,
                top_k=top_k,
                retrieval_mode=retrieval_mode,
                domain=domain,
            )
            if len(responses) == len(claim_texts):
                return responses
            raise RAGClientError(
                f"RAG API returned {len(responses)} results for "
                f"{len(claim_texts)} claims")
        except (RAGClientError, ValueError) as batch_exc:
            # Fall back to per-claim requests so a batch-level failure does
            # not discard unrelated claims (or crash the whole pipeline).
            per_claim: List[Any] = []
            for claim_text in claim_texts:
                try:
                    per_claim.append(self.rag_client.retrieve(
                        claim_text,
                        top_k=top_k,
                        retrieval_mode=retrieval_mode,
                        domain=domain,
                    ))
                except (RAGClientError, ValueError, httpx.HTTPError) as exc:
                    per_claim.append(exc)
            return per_claim

    def _format_claim(
        self,
        claim: Any,
        retrieval: Any,
        errors: List[str],
    ) -> Dict[str, Any]:
        claim_text = claim.get("claim", "") if isinstance(claim, dict) else str(claim)
        if isinstance(retrieval, Exception):
            message = str(retrieval)
            errors.append(f"retrieval failed for claim '{claim_text[:100]}': {message}")
            return {
                **format_claim_evidence(claim_text, None, retrieval_error=message),
                "claim_meta": claim if isinstance(claim, dict) else {},
            }
        try:
            formatted = format_claim_evidence(claim_text, retrieval, retrieval_error=None)
        except (KeyError, TypeError, AttributeError, ValueError) as exc:
            errors.append(
                f"evidence formatting failed for claim '{claim_text[:100]}': "
                f"{exc.__class__.__name__}: {exc}")
            return {
                **format_claim_evidence(claim_text, None,
                                        retrieval_error=f"malformed response: {exc}"),
                "claim_meta": claim if isinstance(claim, dict) else {},
            }
        return {
            **formatted,
            "claim_meta": claim if isinstance(claim, dict) else {},
        }

    def _empty_output(
        self,
        question: str,
        answer: Optional[str],
        errors: List[str],
        extraction: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "question": question,
            "answer": answer,
            "claims": [],
            "extraction": {
                "error": extraction.get("error"),
                "duplicates": extraction.get("duplicates", {}),
                "notes": extraction.get("notes", []),
            },
            "verification": {
                "status": VERIFICATION_PENDING,
                "stage": "pending: verification/classification is a separate later stage",
                "note": VERIFICATION_PENDING_NOTE,
            },
            "errors": errors,
        }
