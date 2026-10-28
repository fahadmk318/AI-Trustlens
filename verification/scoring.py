"""AI TrustLens - Evidence-Grounded Confidence Scoring & Decision Policy.

Implements transparent, deterministic confidence scoring and decision policies:
- Explicit separation of:
  * Evidence Relevance (S_rel)
  * Evidence Support/Alignment (S_sup)
  * Source Authority Tier (S_auth)
  * Knowledge Graph Corroboration (S_kg)
- Standardized Decision Taxonomy:
  * SUPPORTED
  * CONTRADICTED
  * PARTIALLY_SUPPORTED
  * CONFLICTING_EVIDENCE
  * INSUFFICIENT_EVIDENCE
  * REVIEW_REQUIRED
- Fully configurable thresholds and documented score calculations.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Decision Status Taxonomy
STATUS_SUPPORTED = "SUPPORTED"
STATUS_CONTRADICTED = "CONTRADICTED"
STATUS_PARTIALLY_SUPPORTED = "PARTIALLY_SUPPORTED"
STATUS_CONFLICTING_EVIDENCE = "CONFLICTING_EVIDENCE"
STATUS_INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
STATUS_REVIEW_REQUIRED = "REVIEW_REQUIRED"

ALL_DECISION_STATUSES = (
    STATUS_SUPPORTED,
    STATUS_CONTRADICTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_CONFLICTING_EVIDENCE,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_REVIEW_REQUIRED,
)

# Source Authority Tiers
AUTHORITY_TIERS = {
    "who": 1.0,
    "cdc": 1.0,
    "nih": 1.0,
    "fda": 1.0,
    "mayo clinic": 0.90,
    "american heart association": 0.90,
    "aha": 0.90,
    "new england journal of medicine": 0.95,
    "lancet": 0.95,
    "high": 0.90,
    "medium": 0.75,
    "standard": 0.60,
}


class ScoringConfig:
    """Configurable scoring weights and decision thresholds."""

    def __init__(
        self,
        weight_relevance: float = 0.25,
        weight_support: float = 0.45,
        weight_authority: float = 0.15,
        weight_kg: float = 0.15,
        threshold_supported: float = 0.65,
        threshold_contradicted: float = 0.70,
        threshold_partial: float = 0.45,
        threshold_review: float = 0.35,
    ):
        total_w = weight_relevance + weight_support + weight_authority + weight_kg
        self.w_rel = weight_relevance / total_w
        self.w_sup = weight_support / total_w
        self.w_auth = weight_authority / total_w
        self.w_kg = weight_kg / total_w

        self.threshold_supported = threshold_supported
        self.threshold_contradicted = threshold_contradicted
        self.threshold_partial = threshold_partial
        self.threshold_review = threshold_review

    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": {
                "relevance": round(self.w_rel, 3),
                "support": round(self.w_sup, 3),
                "source_authority": round(self.w_auth, 3),
                "knowledge_graph": round(self.w_kg, 3),
            },
            "thresholds": {
                "supported": self.threshold_supported,
                "contradicted": self.threshold_contradicted,
                "partially_supported": self.threshold_partial,
                "review_required": self.threshold_review,
            },
        }


def compute_authority_score(source_name: Optional[str], authority_level: Optional[str] = None) -> float:
    """Determine authority multiplier based on published guidelines / organization."""
    if source_name:
        s_lower = source_name.lower()
        for key, weight in AUTHORITY_TIERS.items():
            if key in s_lower:
                return weight
    if authority_level:
        return AUTHORITY_TIERS.get(authority_level.lower(), 0.75)
    return 0.75


def compute_confidence_score(
    relevance_score: float,
    support_score: float,
    authority_score: float,
    kg_score: float,
    config: Optional[ScoringConfig] = None,
) -> Tuple[float, Dict[str, Any]]:
    """Compute deterministic overall confidence score and detailed breakdown."""
    cfg = config or ScoringConfig()

    rel_clamped = max(0.0, min(1.0, relevance_score))
    sup_clamped = max(0.0, min(1.0, support_score))
    auth_clamped = max(0.0, min(1.0, authority_score))
    kg_clamped = max(0.0, min(1.0, kg_score))

    overall = (
        cfg.w_rel * rel_clamped
        + cfg.w_sup * sup_clamped
        + cfg.w_auth * auth_clamped
        + cfg.w_kg * kg_clamped
    )
    overall = round(max(0.0, min(1.0, overall)), 3)

    breakdown = {
        "overall_confidence": overall,
        "relevance_score": round(rel_clamped, 3),
        "support_score": round(sup_clamped, 3),
        "authority_score": round(auth_clamped, 3),
        "kg_corroboration_score": round(kg_clamped, 3),
        "scoring_formula": f"formula: overall = ({cfg.w_rel:.2f} * rel) + ({cfg.w_sup:.2f} * sup) + ({cfg.w_auth:.2f} * auth) + ({cfg.w_kg:.2f} * kg)",
    }
    return overall, breakdown


def evaluate_decision_status(
    has_supporting_evidence: bool,
    has_contradicting_evidence: bool,
    is_partially_supported: bool,
    overall_confidence: float,
    is_truncated_or_fragment: bool = False,
    retrieval_error: Optional[str] = None,
    evidence_count: int = 0,
    config: Optional[ScoringConfig] = None,
) -> Tuple[str, str]:
    """Map verification outcomes and scores to standardized decision status."""
    cfg = config or ScoringConfig()

    # 1. Structural / Fragment Flag
    if is_truncated_or_fragment:
        return (
            STATUS_REVIEW_REQUIRED,
            "Claim is an incomplete source fragment or truncated sentence lacking predicate; requires human review.",
        )

    # 2. Retrieval Failure
    if retrieval_error:
        return (
            STATUS_INSUFFICIENT_EVIDENCE,
            f"Evidence retrieval failed: {retrieval_error}. Insufficient evidence to substantiate or refute.",
        )

    # 3. Empty Evidence
    if evidence_count == 0 or (not has_supporting_evidence and not has_contradicting_evidence and not is_partially_supported):
        return (
            STATUS_INSUFFICIENT_EVIDENCE,
            "No relevant medical evidence was retrieved to substantiate or refute this claim.",
        )

    # 4. Conflicting Evidence
    if has_supporting_evidence and has_contradicting_evidence:
        return (
            STATUS_CONFLICTING_EVIDENCE,
            "Credible retrieved sources disagree: both supporting and refuting evidence exist in the knowledge base.",
        )

    # 5. Contradicted
    if has_contradicting_evidence:
        return (
            STATUS_CONTRADICTED,
            f"Sufficient medical evidence contradicts the claim (confidence: {overall_confidence:.2f}).",
        )

    # 6. Supported
    if has_supporting_evidence and not is_partially_supported:
        if overall_confidence >= cfg.threshold_supported:
            return (
                STATUS_SUPPORTED,
                f"Sufficient authoritative medical evidence corroborates the complete claim (confidence: {overall_confidence:.2f}).",
            )
        else:
            return (
                STATUS_PARTIALLY_SUPPORTED,
                f"Evidence points toward support, but confidence ({overall_confidence:.2f}) falls below threshold ({cfg.threshold_supported}).",
            )

    # 7. Partially Supported
    if is_partially_supported or (has_supporting_evidence and overall_confidence >= cfg.threshold_partial):
        return (
            STATUS_PARTIALLY_SUPPORTED,
            f"Evidence supports part of the claim (e.g. underlying mechanism or entity), but not all asserted qualifiers or scope.",
        )

    # 8. Ambiguous / Review Required
    return (
        STATUS_REVIEW_REQUIRED,
        f"Evidence alignment is ambiguous or inconclusive (confidence: {overall_confidence:.2f}); manual review required.",
    )
