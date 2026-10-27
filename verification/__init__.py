"""AI TrustLens - Trusted Medical Claim Verification Package.

This package provides Method 2: Trusted Knowledge-Based Medical Claim Verification.

Subsystems:
- TrustedKnowledgeBase: Ingestion, provenance tracking, and BM25 retrieval of authoritative sources.
- MedicalKnowledgeGraph: Medical concepts, typed clinical relations, and graph assertion checks.
- ScoringConfig & compute_confidence_score: Transparent evidence-grounded scoring.
- MedicalClaimVerifier: End-to-end factual verifier uniting text retrieval and KG validation.
"""
from __future__ import annotations

from .knowledge_base import EvidenceChunk, TrustedDocument, TrustedKnowledgeBase
from .knowledge_graph import EntityCategory, MedicalEntity, MedicalKnowledgeGraph, MedicalRelation, RelationType
from .scoring import (
    ALL_DECISION_STATUSES,
    STATUS_CONFLICTING_EVIDENCE,
    STATUS_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_REVIEW_REQUIRED,
    STATUS_SUPPORTED as DECISION_SUPPORTED,
    ScoringConfig,
    compute_authority_score,
    compute_confidence_score,
    evaluate_decision_status,
)
from .verifier import (
    DECISION_CONFLICTING,
    DECISION_CONTRADICTED,
    DECISION_INSUFFICIENT,
    DECISION_PARTIALLY_SUPPORTED,
    DECISION_REVIEW_REQUIRED,
    DECISION_SUPPORTED,
    STATUS_CONFLICTING,
    STATUS_INSUFFICIENT,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_REFUTED,
    STATUS_REVIEW_REQUIRED,
    STATUS_SUPPORTED,
    VERIFICATION_STATUSES,
    EvidenceAlignment,
    EvidenceStance,
    MedicalClaimVerifier,
    VerificationResult,
    verify_claim_against_evidence,
    verify_claims_pipeline,
)

__all__ = [
    "STATUS_SUPPORTED",
    "STATUS_REFUTED",
    "STATUS_PARTIALLY_SUPPORTED",
    "STATUS_CONFLICTING",
    "STATUS_INSUFFICIENT",
    "STATUS_REVIEW_REQUIRED",
    "VERIFICATION_STATUSES",
    "DECISION_SUPPORTED",
    "DECISION_CONTRADICTED",
    "DECISION_PARTIALLY_SUPPORTED",
    "DECISION_CONFLICTING",
    "DECISION_INSUFFICIENT",
    "DECISION_REVIEW_REQUIRED",
    "ALL_DECISION_STATUSES",
    "EvidenceStance",
    "EvidenceAlignment",
    "VerificationResult",
    "MedicalClaimVerifier",
    "verify_claim_against_evidence",
    "verify_claims_pipeline",
    "TrustedDocument",
    "EvidenceChunk",
    "TrustedKnowledgeBase",
    "EntityCategory",
    "RelationType",
    "MedicalEntity",
    "MedicalRelation",
    "MedicalKnowledgeGraph",
    "ScoringConfig",
    "compute_confidence_score",
    "compute_authority_score",
    "evaluate_decision_status",
]
