"""Adapter around the teammate's existing claim extractor (claimextractor.py).

The extractor module is treated as READ-ONLY: it is imported and driven through
its real public interface, never re-implemented or modified.

Real interface used (verified in claimextractor.py):
- extract_medical_claims(answer: str) -> {"claims": [...], "error": None | str}
- split_compound_claims(result, source_text="") -> {"claims": [...], "error": None}
- refine_medical_claims(answer, result) -> {"claims": [...], "error": None|str}

Each claim item is a dict with keys:
    claim, category, source_sentence, heading_context,
    source_start, source_end, needs_review, reason

Extraction requires the local Ollama daemon + model; failures are returned by
the extractor as an 'error' string, not exceptions. This adapter propagates
them explicitly and NEVER manufactures placeholder claims when extraction
fails.

Model selection: the protected claimextractor.py hard-codes MODEL="qwen3:8b"
as its deployment default. To run a different local model (e.g.
qwen3:4b-q4_K_M on modest hardware) without touching the protected file, this
adapter temporarily patches the module-level constant at call time. The file
contents remain untouched; on disk the deployment default stays qwen3:8b.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .model_config import get_ollama_model

from claimextractor import (
    extract_medical_claims,
    refine_medical_claims,
    split_compound_claims,
)

DuplicateInfo = Dict[str, int]


def _normalise_for_dedup(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text.lower()).strip()


class ClaimProcessor:
    """Runs the existing extraction pipeline and post-processes its output."""

    def __init__(self, refine: bool = False, model_name: Optional[str] = None):
        self.refine = refine
        self.model_name = model_name or get_ollama_model()
        self.duplicates: DuplicateInfo = {}
        self.notes: List[str] = []

    def _apply_model_override(self) -> str:
        """Point the claim extractor at the configured model, runtime-only.

        claimextractor.MODEL is re-read on every ollama.chat call inside the
        protected module, so patching the attribute (NOT the file) redirects
        extraction. The default of get_ollama_model() is qwen3:8b, i.e. the
        deployment default, so an unset OLLAMA_MODEL behaves exactly like the
        untouched teammate code.
        """
        try:
            import claimextractor as claimextractor_module
        except Exception:  # pragma: no cover - import errors surface later
            return self.model_name
        if getattr(claimextractor_module, "MODEL", None) != self.model_name:
            claimextractor_module.MODEL = self.model_name
        return self.model_name


    def process(
        self,
        answer_text: str,
        source_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract, decompose (and optionally refine) claims from an answer.

        Returns {"claims": [...], "error": None|str, "duplicates": {...},
        "notes": [...]}. Individual claims keep the extractor's exact format.
        """
        self.duplicates = {}
        self.notes = []

        self._apply_model_override()

        if not isinstance(answer_text, str) or not answer_text.strip():
            return {
                "claims": [],
                "error": "The input answer is empty.",
                "duplicates": self.duplicates,
                "notes": self.notes,
            }

        result = extract_medical_claims(answer_text)

        if result.get("error"):
            # Extraction itself failed (e.g. Ollama down / model missing).
            return {
                "claims": [],
                "error": result["error"],
                "duplicates": self.duplicates,
                "notes": self.notes,
            }

        if self.refine:
            result = refine_medical_claims(answer_text, result)
            if result.get("error"):
                self.notes.append(f"Refinement failed; original claims retained: {result['error']}")
                result["claims"] = result.get("claims") or [
                    claim for claim in extract_medical_claims(answer_text).get("claims", [])
                ]

        result = split_compound_claims(result, source_text=source_text or answer_text)
        if result.get("error"):
            return {
                "claims": [],
                "error": result["error"],
                "duplicates": self.duplicates,
                "notes": self.notes,
            }

        claims = result.get("claims", [])
        claims, self.duplicates = self._drop_duplicates(claims)

        return {
            "claims": claims,
            "error": None,
            "duplicates": self.duplicates,
            "notes": self.notes,
        }

    @staticmethod
    def _drop_duplicates(claims: List[Dict[str, Any]]):
        """Remove exact-duplicate claim texts (normalised), keeping the first."""
        seen: Dict[str, int] = {}
        unique: List[Dict[str, Any]] = []
        for claim in claims:
            if not isinstance(claim, dict):
                continue
            key = _normalise_for_dedup(claim.get("claim", ""))
            if key in seen:
                seen[key] += 1
                continue
            seen[key] = 0
            unique.append(claim)
        counts = {text: n for text, n in seen.items() if n > 0}
        return unique, counts
