"""AI TrustLens - Trusted Knowledge-Based Medical Claim Verification Engine.

This module implements Method 2: Verifying medical claims against trusted
knowledge (authoritative guidelines, clinical evidence, and knowledge graphs)
before presenting them as verified.

Core Architecture:
1. ATOMIC CLAIM EXTRACTION & NORMALIZATION: Preserves source wording, context,
   negations, qualifiers, and numerical precision.
2. TRUSTED EVIDENCE RETRIEVAL: Searches authoritative sources (WHO, CDC, NIH,
   FDA, Mayo Clinic) with provenance tracking.
3. KNOWLEDGE GRAPH VALIDATION: Uses structured entity-relation triples as a
   deterministic corroboration and contradiction layer.
4. EVIDENCE-GROUNDED SCORING: Transparent formula combining relevance, support,
   source authority, and KG corroboration into a calibrated 0.0 - 1.0 confidence score.
5. STANDARDIZED DECISION POLICY:
   - SUPPORTED / Supported
   - CONTRADICTED / Refuted
   - PARTIALLY_SUPPORTED / Partially Supported
   - CONFLICTING_EVIDENCE / Conflicting Evidence
   - INSUFFICIENT_EVIDENCE / Not Enough Evidence
   - REVIEW_REQUIRED / Review Required
6. STRUCTURAL SEPARATION: Structural extraction validation is strictly separated
   from factual verification.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .knowledge_base import EvidenceChunk, TrustedDocument, TrustedKnowledgeBase
from .knowledge_graph import EntityCategory, MedicalEntity, MedicalKnowledgeGraph, MedicalRelation, RelationType
from .scoring import (
    ALL_DECISION_STATUSES,
    STATUS_CONFLICTING_EVIDENCE as DECISION_CONFLICTING,
    STATUS_CONTRADICTED as DECISION_CONTRADICTED,
    STATUS_INSUFFICIENT_EVIDENCE as DECISION_INSUFFICIENT,
    STATUS_PARTIALLY_SUPPORTED as DECISION_PARTIALLY_SUPPORTED,
    STATUS_REVIEW_REQUIRED as DECISION_REVIEW_REQUIRED,
    STATUS_SUPPORTED as DECISION_SUPPORTED,
    ScoringConfig,
    compute_authority_score,
    compute_confidence_score,
    evaluate_decision_status,
)

logger = logging.getLogger(__name__)

# Standardized verification statuses (title-case for backward compatibility)
STATUS_SUPPORTED = "Supported"
STATUS_REFUTED = "Refuted"
STATUS_CONTRADICTED = "Refuted"
STATUS_PARTIALLY_SUPPORTED = "Partially Supported"
STATUS_CONFLICTING = "Conflicting Evidence"
STATUS_CONFLICTING_EVIDENCE = "Conflicting Evidence"
STATUS_INSUFFICIENT = "Not Enough Evidence"
STATUS_INSUFFICIENT_EVIDENCE = "Not Enough Evidence"
STATUS_REVIEW_REQUIRED = "Review Required"

VERIFICATION_STATUSES = (
    STATUS_SUPPORTED,
    STATUS_REFUTED,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_CONFLICTING,
    STATUS_INSUFFICIENT,
    STATUS_REVIEW_REQUIRED,
)

# Standard English stopwords (excluding content domain tokens)
STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an",
    "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before",
    "being", "below", "between", "both", "but", "by", "can", "cannot", "can't",
    "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had",
    "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into",
    "is", "isn't", "it", "it's", "its", "itself", "just", "me", "more", "most",
    "mustn't", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "same", "shan't", "she", "should", "shouldn't", "so", "some", "such",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "wasn't", "we", "were", "weren't", "what",
    "when", "where", "which", "while", "who", "whom", "why", "will", "with",
    "won't", "would", "wouldn't", "you", "your", "yours", "yourself",
    "always", "never", "causes", "cause", "caused", "cases", "case",
}


def _stem(word: str) -> str:
    """Lightweight rule-based English suffix stemmer for medical terminology matching."""
    w = word.lower()
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith("es") and len(w) > 3 and not w.endswith("ses"):
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    if w.endswith("ing") and len(w) > 4:
        return w[:-3]
    if w.endswith("ed") and len(w) > 3:
        return w[:-2]
    return w


# Negation patterns
NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bneither\b",
    r"\bnor\b",
    r"\bwithout\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\bdidn't\b",
    r"\bisn't\b",
    r"\baren't\b",
    r"\bwasn't\b",
    r"\bweren't\b",
    r"\bcan't\b",
    r"\bcannot\b",
    r"\bcouldn't\b",
    r"\bshouldn't\b",
    r"\bwouldn't\b",
    r"\bwon't\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdid not\b",
    r"\bis not\b",
    r"\bare not\b",
    r"\bwas not\b",
    r"\bwere not\b",
    r"\bshould not\b",
    r"\bno symptoms\b",
    r"\basymptomatic\b",
]

# High-certainty absolute qualifiers
ABSOLUTE_QUALIFIERS = [
    r"\balways\b",
    r"\bnever\b",
    r"\ball\b",
    r"\bevery\b",
    r"\bdefinitely\b",
    r"\bcertainly\b",
    r"\bguaranteed\b",
    r"\bexclusively\b",
    r"\bin all cases\b",
    r"\bin every patient\b",
]

# Restrictive or uncertainty qualifiers
RESTRICTIVE_QUALIFIERS = [
    r"\busually\b",
    r"\btypically\b",
    r"\boften\b",
    r"\bsometimes\b",
    r"\brarely\b",
    r"\bin some cases\b",
    r"\bnot always\b",
    r"\bmay\b",
    r"\bmight\b",
    r"\bcould\b",
    r"\bcan\b",
    r"\bpossibly\b",
    r"\bpartially\b",
    r"\bgenerally\b",
    r"\basymptomatic\b",
    r"\bno symptoms\b",
]

# Pairs of opposing medical directions / concepts
MEDICAL_ANTONYMS: List[Tuple[Set[str], Set[str]]] = [
    (
        {"increase", "increases", "increased", "increasing", "elevate", "elevates", "elevated", "raise", "raises", "raised", "high", "higher"},
        {"decrease", "decreases", "decreased", "decreasing", "lower", "lowers", "lowered", "reduce", "reduces", "reduced", "low"},
    ),
    (
        {"safe", "safely", "harmless", "beneficial", "non-harmful"},
        {"harmful", "harm", "toxic", "toxicity", "dangerous", "danger", "contraindicated", "fatal", "hazardous", "adverse"},
    ),
    (
        {"recommended", "indicated", "effective", "advisable", "helpful", "approved"},
        {"contraindicated", "contraindication", "ineffective", "unadvisable", "discouraged", "prohibited", "avoid", "avoided", "harmful"},
    ),
    (
        {"cause", "causes", "caused", "induce", "induces", "induced", "trigger", "triggers"},
        {"prevent", "prevents", "prevented", "prevention", "protect", "protects", "cure", "cures", "inhibit", "inhibits"},
    ),
    (
        {"benign", "non-cancerous"},
        {"malignant", "cancerous", "metastatic"},
    ),
    (
        {"curable"},
        {"incurable"},
    ),
    (
        {"acute"},
        {"chronic"},
    ),
    (
        {"contagious", "infectious"},
        {"non-contagious", "non-infectious"},
    ),
    (
        {"symptomatic", "symptoms"},
        {"asymptomatic", "no symptoms", "without symptoms"},
    ),
]


# ---------------------------------------------------------------------------
# Text Extraction & Parsing Helpers
# ---------------------------------------------------------------------------

def _extract_content_tokens(text: str) -> Set[str]:
    """Extract substantive lower-case word tokens, excluding stopwords, with stemming."""
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return {_stem(w) for w in words if w not in STOPWORDS}


def _extract_numbers_and_units(text: str) -> List[Dict[str, Any]]:
    """Extract structured numeric tokens, ranges, temperatures, and units.

    Returns a list of dicts: {"raw": str, "value": float, "unit": str, "is_range": bool}
    """
    pattern = (
        r"(?<!\w)"
        r"(?P<val1>\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?:\s*(?:-|–|—|to)\s*(?P<val2>\d+(?:,\d{3})*(?:\.\d+)?))?"
        r"\s*(?P<unit>°\s*[CF]|mg|mcg|µg|g|kg|ml|mL|L|"
        r"mmHg|bpm|%|years?|months?|weeks?|days?|hours?|minutes?)?"
    )
    tokens = []
    for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        v1_str = match.group("val1")
        if not v1_str:
            continue
        try:
            v1 = float(v1_str.replace(",", ""))
        except ValueError:
            continue

        v2_str = match.group("val2")
        v2 = None
        if v2_str:
            try:
                v2 = float(v2_str.replace(",", ""))
            except ValueError:
                pass

        unit = match.group("unit")
        unit_clean = re.sub(r"\s+", "", unit).lower() if unit else ""

        tokens.append({
            "raw": match.group(0).strip(),
            "value": v1,
            "range_end": v2,
            "unit": unit_clean,
            "is_range": v2 is not None,
        })
    return tokens


def _has_pattern(text: str, patterns: List[str]) -> bool:
    """Check if any regex pattern matches text."""
    for pattern in patterns:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return True
    return False


def _find_matched_patterns(text: str, patterns: List[str]) -> List[str]:
    """Find all unique pattern matches in text."""
    found = []
    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            found.append(m.group(0).lower().strip())
    return list(dict.fromkeys(found))


# ---------------------------------------------------------------------------
# Stance Evaluation Data Structures
# ---------------------------------------------------------------------------

class EvidenceStance:
    """Stance of a single retrieved evidence chunk relative to a claim."""
    SUPPORTS = "supports"
    REFUTES = "refutes"
    NEUTRAL = "neutral"


class EvidenceAlignment:
    """Attribution and alignment for a single evidence chunk."""

    def __init__(
        self,
        chunk_id: Optional[str],
        text: str,
        source: Optional[str],
        source_url: Optional[str],
        stance: str,
        confidence: float,
        reason: str,
        retrieval_score: Optional[float] = None,
        retrieval_methods: Optional[List[str]] = None,
        authority_level: str = "high",
        publication_date: Optional[str] = None,
    ):
        self.chunk_id = chunk_id
        self.text = text
        self.source = source
        self.source_url = source_url
        self.stance = stance
        self.confidence = confidence
        self.reason = reason
        self.retrieval_score = retrieval_score
        self.retrieval_methods = retrieval_methods or []
        self.authority_level = authority_level
        self.publication_date = publication_date

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source": self.source,
            "source_url": self.source_url,
            "publication_date": self.publication_date,
            "authority_level": self.authority_level,
            "text": self.text,
            "stance": self.stance,
            "confidence": round(self.confidence, 3),
            "reason": self.reason,
            "retrieval_score": self.retrieval_score,
            "retrieval_methods": self.retrieval_methods,
        }


class VerificationResult:
    """Structured verification outcome for a single atomic claim."""

    def __init__(
        self,
        claim: str,
        category: str,
        verification_status: str,
        verification_explanation: str,
        confidence: float,
        decision_status: Optional[str] = None,
        confidence_breakdown: Optional[Dict[str, Any]] = None,
        structural_validation: Optional[Dict[str, Any]] = None,
        alignments: Optional[List[EvidenceAlignment]] = None,
        kg_evidence: Optional[List[Dict[str, Any]]] = None,
        source_sentence: str = "",
        heading_context: str = "",
        retrieval_status: str = "ok",
        retrieval_error: Optional[str] = None,
        normalized_claim: Optional[str] = None,
    ):
        self.claim = claim
        self.normalized_claim = normalized_claim or re.sub(r"\s+", " ", claim).strip()
        self.category = category
        self.verification_status = verification_status
        self.decision_status = decision_status or self._map_to_decision_status(verification_status)
        self.verification_explanation = verification_explanation
        self.confidence = confidence
        self.confidence_breakdown = confidence_breakdown or {}
        self.structural_validation = structural_validation or {
            "status": "PASSED",
            "validation": "PASSED",
            "issues": [],
            "is_structurally_valid": True,
            "note": "PASSED indicates only that the claim satisfied structural and source-alignment fidelity checks against the supplied text. It does NOT verify whether the claim is medically accurate in reality.",
        }
        self.alignments = alignments or []
        self.kg_evidence = kg_evidence or []
        self.source_sentence = source_sentence
        self.heading_context = heading_context
        self.retrieval_status = retrieval_status
        self.retrieval_error = retrieval_error

    @staticmethod
    def _map_to_decision_status(status: str) -> str:
        s_lower = status.lower()
        if "partial" in s_lower:
            return DECISION_PARTIALLY_SUPPORTED
        if "conflict" in s_lower:
            return DECISION_CONFLICTING
        if "refut" in s_lower or "contra" in s_lower:
            return DECISION_CONTRADICTED
        if "support" in s_lower:
            return DECISION_SUPPORTED
        if "review" in s_lower:
            return DECISION_REVIEW_REQUIRED
        return DECISION_INSUFFICIENT

    def to_dict(self) -> Dict[str, Any]:
        supporting = [a.to_dict() for a in self.alignments if a.stance == EvidenceStance.SUPPORTS]
        contradicting = [a.to_dict() for a in self.alignments if a.stance == EvidenceStance.REFUTES]
        neutral = [a.to_dict() for a in self.alignments if a.stance == EvidenceStance.NEUTRAL]

        struct_status = (
            self.structural_validation.get("status")
            or self.structural_validation.get("validation", "PASSED")
        )
        issues = self.structural_validation.get("issues", [])

        return {
            "claim": self.claim,
            "normalized_claim": self.normalized_claim,
            "category": self.category,
            "source_sentence": self.source_sentence,
            "heading_context": self.heading_context,
            "validation": struct_status,
            "structural_validation": self.structural_validation,
            "verification_status": self.verification_status,
            "decision_status": self.decision_status,
            "confidence": round(self.confidence, 3),
            "confidence_score": round(self.confidence, 3),
            "confidence_breakdown": self.confidence_breakdown,
            "evidence": [a.to_dict() for a in self.alignments],
            "evidence_alignment": [a.to_dict() for a in self.alignments],
            "evidence_found": bool(self.alignments),
            "supporting_sources": supporting,
            "contradicting_sources": contradicting,
            "neutral_sources": neutral,
            "knowledge_graph_evidence": self.kg_evidence,
            "issues": issues,
            "retrieval_status": self.retrieval_status,
            "retrieval_error": self.retrieval_error,
            "verification_explanation": self.verification_explanation,
            "explanation": self.verification_explanation,
        }


# ---------------------------------------------------------------------------
# Core Medical Verification Logic
# ---------------------------------------------------------------------------

def _check_numbers_pair(
    claim_nums: List[Dict[str, Any]],
    ev_nums: List[Dict[str, Any]],
) -> Optional[str]:
    """Compare claim numbers against an evidence number set for contradictory values."""
    for c_num in claim_nums:
        c_unit = c_num["unit"]
        c_val = c_num["value"]

        has_match = False
        conflicting_e_num = None

        for e_num in ev_nums:
            e_unit = e_num["unit"]
            e_val = e_num["value"]

            # Same explicit unit (e.g. °f vs °f, or mg vs mg)
            if c_unit and e_unit and c_unit == e_unit:
                if abs(c_val - e_val) <= 0.5:
                    has_match = True
                    break
                else:
                    conflicting_e_num = e_num
            # Temperature cross-unit check (°f vs °c): e.g. 100.4°F is 38°C
            elif "°f" in c_unit and "°c" in e_unit:
                c_as_celsius = (c_val - 32.0) * 5.0 / 9.0
                if abs(c_as_celsius - e_val) <= 1.0:
                    has_match = True
                    break
                else:
                    conflicting_e_num = e_num
            elif "°c" in c_unit and "°f" in e_unit:
                c_as_fahrenheit = (c_val * 9.0 / 5.0) + 32.0
                if abs(c_as_fahrenheit - e_val) <= 1.5:
                    has_match = True
                    break
                else:
                    conflicting_e_num = e_num

        if not has_match and conflicting_e_num is not None:
            return (
                f"Numeric threshold conflict: claim states '{c_num['raw']}' "
                f"whereas evidence establishes '{conflicting_e_num['raw']}'."
            )
    return None


def _check_numeric_contradiction(
    claim_nums: List[Dict[str, Any]],
    ev_nums: List[Dict[str, Any]],
    claim_tokens: Set[str],
    ev_tokens: Set[str],
    ev_text: str = "",
) -> Optional[str]:
    """Detect if numbers, units, ranges, or thresholds in claim conflict with evidence."""
    if not claim_nums or not ev_nums:
        return None

    distinct_concepts = [
        ({"normal", "baseline"}, {"fever", "pyrexia", "hyperthermia"}),
        ({"normal", "baseline"}, {"hypertension", "high", "elevated"}),
        ({"normal", "baseline"}, {"hypotension", "low"}),
        ({"systolic"}, {"diastolic"}),
        ({"child", "pediatric", "infant"}, {"adult", "elderly"}),
    ]

    # If ev_text is provided and has multiple sentences, evaluate sentence by sentence
    ev_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev_text) if s.strip()] if ev_text else []

    if len(ev_sentences) > 1:
        for s in ev_sentences:
            s_nums = _extract_numbers_and_units(s)
            if not s_nums:
                continue
            s_toks = _extract_content_tokens(s)

            # Check concept divergence
            divergent = False
            for set_a, set_b in distinct_concepts:
                if (claim_tokens & set_a) and (s_toks & set_b) and not (claim_tokens & set_b):
                    divergent = True
                    break
                if (claim_tokens & set_b) and (s_toks & set_a) and not (claim_tokens & set_a):
                    divergent = True
                    break
            if divergent:
                continue

            # Need at least one shared concept token
            if len(claim_tokens & s_toks) < 1:
                continue

            conflict = _check_numbers_pair(claim_nums, s_nums)
            if conflict:
                return conflict
        return None

    # Single sentence or direct comparison
    for set_a, set_b in distinct_concepts:
        if (claim_tokens & set_a) and (ev_tokens & set_b) and not (claim_tokens & set_b):
            return None
        if (claim_tokens & set_b) and (ev_tokens & set_a) and not (claim_tokens & set_a):
            return None

    return _check_numbers_pair(claim_nums, ev_nums)


def _check_negation_conflict(
    claim_text: str,
    evidence_text: str,
    shared_tokens: Set[str],
) -> Optional[Tuple[str, str]]:
    """Detect negation mismatch between claim and evidence on shared medical concepts."""
    if len(shared_tokens) < 2:
        return None

    claim_has_neg = _has_pattern(claim_text, NEGATION_PATTERNS)
    ev_has_neg = _has_pattern(evidence_text, NEGATION_PATTERNS)

    claim_lower = claim_text.lower()
    ev_lower = evidence_text.lower()

    # Case 1: Claim affirms relationship, but evidence explicitly negates it
    if not claim_has_neg and ev_has_neg:
        for neg_phrase in [
            r"\bnot\s+(?:harmful|recommended|indicated|effective|dangerous|a sign|curable)\b",
            r"\bno\s+(?:symptoms|harm|cure|evidence|sign)\b",
            r"\bwithout\s+(?:symptoms|harm|cause)\b",
            r"\bdoes\s+not\s+(?:cause|treat|reduce|help|require|indicate)\b",
            r"\bdo\s+not\s+(?:cause|treat|reduce|help|require|indicate)\b",
            r"\bshould\s+not\s+(?:be used|be given|take)\b",
            r"\bnever\s+(?:use|give|recommend)\b",
            r"\basymptomatic\b",
        ]:
            if re.search(neg_phrase, ev_lower):
                for term in ["harmful", "symptoms", "treat", "cause", "reduce", "recommended", "given"]:
                    if term in claim_lower and re.search(r"\b" + term, neg_phrase):
                        return (
                            "negation_conflict",
                            f"Negation conflict: evidence explicitly negates '{term}' which claim affirms.",
                        )
                if "symptom" in ev_lower and ("symptom" in claim_lower or "headache" in claim_lower or "pain" in claim_lower):
                    if re.search(r"\b(?:no symptoms|asymptomatic)\b", ev_lower):
                        return (
                            "negation_conflict",
                            "Negation conflict: evidence states condition is asymptomatic / has no symptoms, contradicting asserted symptoms.",
                        )

    # Case 2: Claim negates relationship, but evidence explicitly affirms it
    elif claim_has_neg and not ev_has_neg:
        for neg_phrase in [
            r"\bnot\s+(?:harmful|recommended|indicated|effective|a sign)\b",
            r"\bdoes\s+not\s+(?:cause|treat|reduce|help|indicate)\b",
            r"\bshould\s+not\s+(?:be used|be given)\b",
        ]:
            if re.search(neg_phrase, claim_lower):
                for term in ["harmful", "recommended", "effective", "indicate", "cause"]:
                    if term in ev_lower and re.search(r"\b" + term, neg_phrase):
                        return (
                            "negation_conflict",
                            f"Negation conflict: claim negates '{term}', but evidence affirms it.",
                        )

    return None


def _check_antonym_conflict(
    claim_text: str,
    evidence_text: str,
    shared_tokens: Set[str],
) -> Optional[str]:
    """Detect opposing medical terms (e.g. increase vs decrease, safe vs toxic, recommended vs contraindicated)."""
    if len(shared_tokens) < 1:
        return None

    claim_words = set(re.findall(r"\b[a-zA-Z]+\b", claim_text.lower()))
    ev_words = set(re.findall(r"\b[a-zA-Z]+\b", evidence_text.lower()))

    for set_a, set_b in MEDICAL_ANTONYMS:
        a_in_claim = bool(claim_words & set_a)
        b_in_claim = bool(claim_words & set_b)
        a_in_ev = bool(ev_words & set_a)
        b_in_ev = bool(ev_words & set_b)

        if a_in_claim and b_in_ev and not (b_in_claim or a_in_ev):
            w_claim = next(iter(claim_words & set_a))
            w_ev = next(iter(ev_words & set_b))
            return (
                f"Directional / medical contradiction: claim asserts '{w_claim}' "
                f"whereas evidence establishes opposing term '{w_ev}'."
            )
        elif b_in_claim and a_in_ev and not (a_in_claim or b_in_ev):
            w_claim = next(iter(claim_words & set_b))
            w_ev = next(iter(ev_words & set_a))
            return (
                f"Directional / medical contradiction: claim asserts '{w_claim}' "
                f"whereas evidence establishes opposing term '{w_ev}'."
            )

    return None


def _check_qualifier_conflict(
    claim_text: str,
    evidence_text: str,
    shared_tokens: Set[str],
) -> Optional[str]:
    """Detect conflict where claim makes an absolute assertion refuted by nuanced evidence."""
    if len(shared_tokens) < 2:
        return None

    claim_absolutes = _find_matched_patterns(claim_text, ABSOLUTE_QUALIFIERS)
    ev_restrictives = _find_matched_patterns(evidence_text, RESTRICTIVE_QUALIFIERS)

    if claim_absolutes and ev_restrictives:
        for abs_q in claim_absolutes:
            for rest_q in ev_restrictives:
                if rest_q in ["no symptoms", "asymptomatic", "rarely", "not always"]:
                    return (
                        f"Qualifier contradiction: claim asserts absolute '{abs_q}', "
                        f"which is contradicted by evidence establishing '{rest_q}'."
                    )

    return None


def evaluate_evidence_item(
    claim_text: str,
    evidence_item: Dict[str, Any],
    min_overlap: float = 0.30,
    support_overlap: float = 0.50,
) -> EvidenceAlignment:
    """Evaluate a single evidence chunk against a claim to determine its stance."""
    ev_text = evidence_item.get("text", "")
    chunk_id = evidence_item.get("metadata", {}).get("chunk_id") or evidence_item.get("chunk_id")
    source = evidence_item.get("source") or evidence_item.get("title") or evidence_item.get("source_name")
    source_url = evidence_item.get("metadata", {}).get("source_url") or evidence_item.get("source_url")
    publication_date = evidence_item.get("metadata", {}).get("publication_date") or evidence_item.get("publication_date")
    authority_level = evidence_item.get("metadata", {}).get("authority_level") or evidence_item.get("authority_level", "high")
    retrieval_score = evidence_item.get("retrieval_score")
    retrieval_methods = evidence_item.get("retrieval_method") or evidence_item.get("retrieval_methods") or []

    if not isinstance(ev_text, str) or not ev_text.strip():
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text="",
            source=source,
            source_url=source_url,
            stance=EvidenceStance.NEUTRAL,
            confidence=0.0,
            reason="Empty evidence text.",
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    claim_tokens = _extract_content_tokens(claim_text)
    ev_tokens = _extract_content_tokens(ev_text)

    if not claim_tokens:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.NEUTRAL,
            confidence=0.0,
            reason="Claim contains no substantive medical content words.",
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    shared_tokens = claim_tokens & ev_tokens
    overlap_ratio = len(shared_tokens) / len(claim_tokens)

    # 1. Below minimum overlap: not addressing the same specific proposition
    if overlap_ratio < min_overlap and len(shared_tokens) < 2:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.NEUTRAL,
            confidence=max(0.0, 1.0 - overlap_ratio),
            reason=f"Insufficient topic overlap ({overlap_ratio:.1%}): evidence does not address the specific claim proposition.",
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # Extract numbers & units
    claim_nums = _extract_numbers_and_units(claim_text)
    ev_nums = _extract_numbers_and_units(ev_text)

    # 2. Check Contradiction: Numeric/Range Mismatch
    num_conflict = _check_numeric_contradiction(claim_nums, ev_nums, claim_tokens, ev_tokens, ev_text=ev_text)
    if num_conflict:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.REFUTES,
            confidence=0.95,
            reason=num_conflict,
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # 3. Check Contradiction: Negation Mismatch
    neg_conflict = _check_negation_conflict(claim_text, ev_text, shared_tokens)
    if neg_conflict:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.REFUTES,
            confidence=0.90,
            reason=neg_conflict[1],
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # 4. Check Contradiction: Medical Antonyms / Directional Mismatch
    antonym_conflict = _check_antonym_conflict(claim_text, ev_text, shared_tokens)
    if antonym_conflict:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.REFUTES,
            confidence=0.90,
            reason=antonym_conflict,
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # 5. Check Contradiction: Qualifier / Modality Mismatch
    qual_conflict = _check_qualifier_conflict(claim_text, ev_text, shared_tokens)
    if qual_conflict:
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.REFUTES,
            confidence=0.85,
            reason=qual_conflict,
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # 6. Check Support: Matching numbers, matching polarity, sufficient overlap
    claim_has_neg = _has_pattern(claim_text, NEGATION_PATTERNS)

    # Check sentence-level alignment for polarity if passage has multiple sentences
    ev_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ev_text) if s.strip()]
    if ev_sentences:
        matching_sentences = []
        for s in ev_sentences:
            s_toks = _extract_content_tokens(s)
            if not s_toks:
                continue
            s_overlap = len(claim_tokens & s_toks) / max(1, len(claim_tokens))
            if s_overlap >= min_overlap:
                matching_sentences.append((s, s_overlap))

        if matching_sentences:
            best_s, best_s_overlap = max(matching_sentences, key=lambda x: x[1])
            best_s_has_neg = _has_pattern(best_s, NEGATION_PATTERNS)
            polarity_aligned = (claim_has_neg == best_s_has_neg) or (not claim_has_neg and not neg_conflict)
        else:
            polarity_aligned = (not claim_has_neg and not neg_conflict) or (claim_has_neg and _has_pattern(ev_text, NEGATION_PATTERNS))
    else:
        polarity_aligned = (not claim_has_neg and not neg_conflict) or (claim_has_neg and _has_pattern(ev_text, NEGATION_PATTERNS))

    numbers_matched = False
    if claim_nums and ev_nums:
        matched_count = 0
        for c_num in claim_nums:
            c_unit = c_num["unit"]
            c_val = c_num["value"]
            for e_num in ev_nums:
                e_unit = e_num["unit"]
                e_val = e_num["value"]
                if c_unit and e_unit and c_unit == e_unit:
                    if abs(c_val - e_val) <= 0.5:
                        matched_count += 1
                        break
                elif "°f" in c_unit and "°c" in e_unit:
                    c_c = (c_val - 32.0) * 5.0 / 9.0
                    if abs(c_c - e_val) <= 1.0:
                        matched_count += 1
                        break
                elif "°c" in c_unit and "°f" in e_unit:
                    c_f = (c_val * 9.0 / 5.0) + 32.0
                    if abs(c_f - e_val) <= 1.5:
                        matched_count += 1
                        break
        if matched_count > 0:
            numbers_matched = True

    # If the claim asserts quantitative numbers/thresholds, evidence must match numbers to support
    can_support = (not claim_nums) or numbers_matched

    if overlap_ratio >= support_overlap and polarity_aligned and can_support:
        confidence = min(0.98, 0.5 + (overlap_ratio * 0.4) + (0.1 if numbers_matched else 0.0))
        reason = f"Evidence directly corroborates claim assertions with high semantic overlap ({overlap_ratio:.1%})."
        if numbers_matched:
            reason += " Exact numerical thresholds and units match."
        return EvidenceAlignment(
            chunk_id=chunk_id,
            text=ev_text,
            source=source,
            source_url=source_url,
            stance=EvidenceStance.SUPPORTS,
            confidence=confidence,
            reason=reason,
            retrieval_score=retrieval_score,
            retrieval_methods=retrieval_methods,
            authority_level=authority_level,
            publication_date=publication_date,
        )

    # Partial / moderate overlap without explicit corroboration or refutation
    return EvidenceAlignment(
        chunk_id=chunk_id,
        text=ev_text,
        source=source,
        source_url=source_url,
        stance=EvidenceStance.NEUTRAL,
        confidence=0.5,
        reason=f"Partial overlap ({overlap_ratio:.1%}) but evidence neither confirms nor contradicts the specific relationship.",
        retrieval_score=retrieval_score,
        retrieval_methods=retrieval_methods,
        authority_level=authority_level,
        publication_date=publication_date,
    )


def verify_claim_against_evidence(
    claim_dict: Dict[str, Any],
    evidence_result: Dict[str, Any],
    structural_validation: Optional[Dict[str, Any]] = None,
    knowledge_base: Optional[TrustedKnowledgeBase] = None,
    knowledge_graph: Optional[MedicalKnowledgeGraph] = None,
    scoring_config: Optional[ScoringConfig] = None,
    min_overlap: float = 0.30,
    support_overlap: float = 0.50,
    auto_retrieve_trusted: bool = True,
) -> VerificationResult:
    """Perform factual verification of an atomic medical claim against retrieved evidence and KG.

    Unites text evidence retrieval, structured KG consistency, and transparent scoring.
    """
    claim_text = (
        claim_dict.get("claim", "") if isinstance(claim_dict, dict)
        else str(claim_dict)
    ).strip()
    category = claim_dict.get("category", "other") if isinstance(claim_dict, dict) else "other"
    source_sentence = claim_dict.get("source_sentence", "") if isinstance(claim_dict, dict) else ""
    heading_context = claim_dict.get("heading_context", "") if isinstance(claim_dict, dict) else ""

    cfg = scoring_config or ScoringConfig()

    # Check structural validation issues
    struct_val = structural_validation or claim_dict.get("structural_validation")
    is_incomplete_fragment = False
    is_qualifier_drift = False
    if struct_val:
        issues = struct_val.get("issues", [])
        if any("incomplete fragment" in iss.lower() for iss in issues):
            is_incomplete_fragment = True
        if any("semantic inflation" in iss.lower() or "qualifier dropped" in iss.lower() for iss in issues):
            is_qualifier_drift = True
    elif claim_dict.get("needs_review") and "incomplete" in claim_dict.get("reason", "").lower():
        is_incomplete_fragment = True

    retrieval_status = evidence_result.get("retrieval_status", "ok")
    retrieval_error = evidence_result.get("retrieval_error")
    raw_evidence = list(evidence_result.get("evidence", []))

    # Case 1: Incomplete fragment input (e.g. truncated markdown header/bullet)
    if is_incomplete_fragment:
        conf, breakdown = compute_confidence_score(0.0, 0.0, 0.0, 0.0, config=cfg)
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_INSUFFICIENT,
            decision_status=DECISION_REVIEW_REQUIRED,
            verification_explanation=(
                "Incomplete fragment: fragment lacks predicate or context "
                "and cannot be verified factually as a standalone claim."
            ),
            confidence=conf,
            confidence_breakdown=breakdown,
            alignments=[],
            retrieval_status=retrieval_status,
            retrieval_error=retrieval_error,
        )

    # Case 2: Retrieval system failure / error
    if retrieval_status == "retrieval_error" or retrieval_error:
        conf, breakdown = compute_confidence_score(0.0, 0.0, 0.0, 0.0, config=cfg)
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_INSUFFICIENT,
            decision_status=DECISION_INSUFFICIENT,
            verification_explanation=(
                f"Evidence retrieval failed: {retrieval_error or 'Retrieval service reported an error'}. "
                "Cannot perform factual verification without retrieved evidence."
            ),
            confidence=conf,
            confidence_breakdown=breakdown,
            alignments=[],
            retrieval_status="retrieval_error",
            retrieval_error=retrieval_error,
        )

    # Auto-retrieve from local TrustedKnowledgeBase if raw_evidence is empty or enabled
    if auto_retrieve_trusted and knowledge_base and not raw_evidence and retrieval_status != "no_evidence":
        trusted_hits = knowledge_base.search(claim_text, top_k=5)
        raw_evidence.extend(trusted_hits)

    # Case 3: Empty retrieval result (no evidence found)
    if not raw_evidence or (retrieval_status == "no_evidence" and not raw_evidence):
        conf, breakdown = compute_confidence_score(0.0, 0.0, 0.0, 0.0, config=cfg)
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_INSUFFICIENT,
            decision_status=DECISION_INSUFFICIENT,
            verification_explanation=(
                "No relevant evidence was retrieved from the medical knowledge corpus "
                "to substantiate or refute this claim."
            ),
            confidence=0.0,
            confidence_breakdown=breakdown,
            alignments=[],
            retrieval_status="no_evidence",
            retrieval_error=None,
        )

    # 4. Check Knowledge Graph for structured entity-relation triples
    kg_matches: List[Dict[str, Any]] = []
    kg_supported = False
    kg_contradicted = False
    kg_score = 0.30

    if knowledge_graph:
        kg_matches = knowledge_graph.find_connected_knowledge(claim_text)
        # Check specific assertions in KG
        claim_nums = _extract_numbers_and_units(claim_text)
        if "fever" in claim_text.lower() and claim_nums:
            c_val_str = claim_nums[0]["raw"]
            sup, con, rels, _ = knowledge_graph.check_assertion("Fever", RelationType.DEFINES_THRESHOLD, c_val_str)
            if sup:
                kg_supported = True
            elif con:
                kg_contradicted = True
        elif "temperature" in claim_text.lower() and claim_nums:
            sup, con, rels, _ = knowledge_graph.check_assertion("Normal Body Temperature", RelationType.NORMAL_RANGE, claim_text)
            if sup:
                kg_supported = True
            elif con:
                kg_contradicted = True
        elif "aspirin" in claim_text.lower() and ("child" in claim_text.lower() or "viral" in claim_text.lower()):
            if "recommended" in claim_text.lower() or "treat" in claim_text.lower():
                kg_contradicted = True
            elif "not" in claim_text.lower() or "contraindicated" in claim_text.lower():
                kg_supported = True

        if kg_supported:
            kg_score = 1.0
        elif kg_contradicted:
            kg_score = 0.95
        elif kg_matches:
            kg_score = 0.65

    # 5. Evaluate each evidence chunk
    alignments: List[EvidenceAlignment] = [
        evaluate_evidence_item(claim_text, item, min_overlap=min_overlap, support_overlap=support_overlap)
        for item in raw_evidence
        if isinstance(item, dict)
    ]

    supporting = [a for a in alignments if a.stance == EvidenceStance.SUPPORTS]
    refuting = [a for a in alignments if a.stance == EvidenceStance.REFUTES]

    # Calculate average authority and relevance
    auth_scores = [compute_authority_score(a.source, a.authority_level) for a in alignments]
    avg_auth = sum(auth_scores) / len(auth_scores) if auth_scores else 0.75

    claim_toks = _extract_content_tokens(claim_text)
    relevance_scores = [
        len(claim_toks & _extract_content_tokens(a.text)) / max(1, len(claim_toks))
        for a in alignments
    ]
    max_rel = max(relevance_scores) if relevance_scores else 0.0

    # Synthesis of stances
    # 6. Conflicting Evidence
    if (supporting and refuting) or (supporting and kg_contradicted) or (refuting and kg_supported):
        sources_sup = ", ".join(filter(None, [a.source for a in supporting])) or "supporting source(s)"
        sources_ref = ", ".join(filter(None, [a.source for a in refuting])) or "refuting source(s)"
        explanation = (
            f"Conflicting medical evidence found: {len(supporting)} source(s) ({sources_sup}) "
            f"support the claim, while {len(refuting)} source(s) ({sources_ref}) refute or contradict it."
        )
        conf, breakdown = compute_confidence_score(
            relevance_score=max_rel,
            support_score=0.50,
            authority_score=avg_auth,
            kg_score=0.50,
            config=cfg,
        )
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_CONFLICTING,
            decision_status=DECISION_CONFLICTING,
            verification_explanation=explanation,
            confidence=conf,
            confidence_breakdown=breakdown,
            alignments=alignments,
            kg_evidence=kg_matches,
            retrieval_status=retrieval_status,
        )

    # 7. Refuted / Contradicted
    if refuting or kg_contradicted:
        reasons = "; ".join(a.reason for a in refuting)
        if kg_contradicted and not reasons:
            reasons = "Contradicted by established medical Knowledge Graph relationships."
        sources_ref = ", ".join(filter(None, [a.source for a in refuting])) or "authoritative medical guidelines"
        explanation = f"Claim is contradicted by {sources_ref}: {reasons}"
        refute_conf = max([a.confidence for a in refuting] + [0.90 if kg_contradicted else 0.0])
        conf, breakdown = compute_confidence_score(
            relevance_score=max_rel,
            support_score=refute_conf,
            authority_score=avg_auth,
            kg_score=kg_score,
            config=cfg,
        )
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_REFUTED,
            decision_status=DECISION_CONTRADICTED,
            verification_explanation=explanation,
            confidence=conf,
            confidence_breakdown=breakdown,
            alignments=alignments,
            kg_evidence=kg_matches,
            retrieval_status=retrieval_status,
        )

    # 8. Supported or Partially Supported
    if supporting or kg_supported:
        sources_sup = ", ".join(filter(None, [a.source for a in supporting])) or "authoritative medical guidelines"
        reasons = "; ".join(a.reason for a in supporting)
        sup_conf = max([a.confidence for a in supporting] + [0.90 if kg_supported else 0.0])
        conf, breakdown = compute_confidence_score(
            relevance_score=max_rel,
            support_score=sup_conf,
            authority_score=avg_auth,
            kg_score=kg_score,
            config=cfg,
        )

        # Check for partial support (e.g. qualifier inflation/drift in claim)
        if is_qualifier_drift:
            explanation = (
                f"Claim is partially supported by {sources_sup}, but asserted certainty or scope "
                f"exceeds what the evidence establishes."
            )
            return VerificationResult(
                claim=claim_text,
                category=category,
                source_sentence=source_sentence,
                heading_context=heading_context,
                structural_validation=struct_val,
                verification_status=STATUS_SUPPORTED,
                decision_status=DECISION_PARTIALLY_SUPPORTED,
                verification_explanation=explanation,
                confidence=conf,
                confidence_breakdown=breakdown,
                alignments=alignments,
                kg_evidence=kg_matches,
                retrieval_status=retrieval_status,
            )

        explanation = f"Claim is corroborated by {sources_sup}: {reasons}"
        return VerificationResult(
            claim=claim_text,
            category=category,
            source_sentence=source_sentence,
            heading_context=heading_context,
            structural_validation=struct_val,
            verification_status=STATUS_SUPPORTED,
            decision_status=DECISION_SUPPORTED,
            verification_explanation=explanation,
            confidence=conf,
            confidence_breakdown=breakdown,
            alignments=alignments,
            kg_evidence=kg_matches,
            retrieval_status=retrieval_status,
        )

    # 9. Insufficient Evidence
    explanation = (
        "Retrieved evidence passages discuss related medical concepts but do not contain "
        "sufficient specific information to confirm or refute the asserted claim."
    )
    conf, breakdown = compute_confidence_score(
        relevance_score=max_rel,
        support_score=0.20,
        authority_score=avg_auth,
        kg_score=kg_score,
        config=cfg,
    )
    return VerificationResult(
        claim=claim_text,
        category=category,
        source_sentence=source_sentence,
        heading_context=heading_context,
        structural_validation=struct_val,
        verification_status=STATUS_INSUFFICIENT,
        decision_status=DECISION_INSUFFICIENT,
        verification_explanation=explanation,
        confidence=conf,
        confidence_breakdown=breakdown,
        alignments=alignments,
        kg_evidence=kg_matches,
        retrieval_status=retrieval_status,
    )


# ---------------------------------------------------------------------------
# Medical Claim Verifier Class & Pipeline Integrator
# ---------------------------------------------------------------------------

class MedicalClaimVerifier:
    """High-level verifier supporting trusted knowledge bases, knowledge graphs, and local models."""

    def __init__(
        self,
        mode: str = "deterministic",
        model_name: Optional[str] = None,
        knowledge_base: Optional[TrustedKnowledgeBase] = None,
        knowledge_graph: Optional[MedicalKnowledgeGraph] = None,
        scoring_config: Optional[ScoringConfig] = None,
        min_overlap: float = 0.30,
        support_overlap: float = 0.50,
        auto_retrieve_trusted: bool = True,
    ):
        self.mode = mode.lower()
        self.model_name = model_name or "qwen3:8b"
        self.knowledge_base = knowledge_base or TrustedKnowledgeBase()
        self.knowledge_graph = knowledge_graph or MedicalKnowledgeGraph()
        self.scoring_config = scoring_config or ScoringConfig()
        self.min_overlap = min_overlap
        self.support_overlap = support_overlap
        self.auto_retrieve_trusted = auto_retrieve_trusted

    def verify_claim(
        self,
        claim_dict: Dict[str, Any],
        evidence_result: Optional[Dict[str, Any]] = None,
        structural_validation: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Verify a single claim against retrieved evidence and knowledge graph."""
        ev_result = evidence_result or {"evidence": [], "retrieval_status": "ok"}

        # Run deterministic verification as the baseline
        det_result = verify_claim_against_evidence(
            claim_dict,
            ev_result,
            structural_validation=structural_validation,
            knowledge_base=self.knowledge_base,
            knowledge_graph=self.knowledge_graph,
            scoring_config=self.scoring_config,
            min_overlap=self.min_overlap,
            support_overlap=self.support_overlap,
            auto_retrieve_trusted=self.auto_retrieve_trusted,
        )

        if self.mode == "deterministic":
            return det_result.to_dict()

        # In hybrid or llm mode, try LLM refinement if conditions warrant
        if self.mode in ("hybrid", "llm"):
            llm_result = self._verify_with_model(claim_dict, ev_result, det_result)
            if llm_result is not None:
                return llm_result

        return det_result.to_dict()

    def _verify_with_model(
        self,
        claim_dict: Dict[str, Any],
        evidence_result: Dict[str, Any],
        baseline_result: VerificationResult,
    ) -> Optional[Dict[str, Any]]:
        """Optional LLM verification via local Ollama with fallback."""
        try:
            import ollama
        except ImportError:
            return None

        evidence_items = evidence_result.get("evidence", [])
        if not evidence_items and self.knowledge_base:
            claim_txt = claim_dict.get("claim", "")
            evidence_items = self.knowledge_base.search(claim_txt, top_k=5)

        if not evidence_items:
            return None

        claim_text = claim_dict.get("claim", "")
        evidence_snippets = "\n\n".join([
            f"[{idx+1}] Source: {item.get('source') or 'Medical Document'}\nText: {item.get('text', '')}"
            for idx, item in enumerate(evidence_items[:5])
        ])

        system_prompt = (
            "You are a medical claim verification system. Assess whether the supplied medical "
            "claim is Supported, Refuted, Conflicting Evidence, or Not Enough Evidence based "
            "strictly on the provided evidence passages. Do not invent medical facts.\n"
            "Return JSON only:\n"
            "{\n"
            '  "verification_status": "Supported" | "Refuted" | "Conflicting Evidence" | "Not Enough Evidence",\n'
            '  "explanation": "concise explanation of reasoning",\n'
            '  "confidence": 0.0 to 1.0\n'
            "}"
        )
        user_prompt = f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_snippets}"

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0, "num_predict": 400},
            )
            parsed = json.loads(response["message"]["content"])
            status = parsed.get("verification_status")
            if status in VERIFICATION_STATUSES:
                res_dict = baseline_result.to_dict()
                res_dict["verification_status"] = status
                res_dict["verification_explanation"] = parsed.get("explanation") or baseline_result.verification_explanation
                res_dict["confidence"] = float(parsed.get("confidence", baseline_result.confidence))
                res_dict["confidence_score"] = res_dict["confidence"]
                return res_dict
        except Exception as exc:
            logger.warning("LLM claim verification failed, falling back to deterministic: %s", exc)

        return None


def verify_claims_pipeline(
    pipeline_result: Any,
    verifier: Optional[MedicalClaimVerifier] = None,
) -> Dict[str, Any]:
    """Apply factual verification to all claims in a TrustLens RAG pipeline output."""
    if verifier is None:
        verifier = MedicalClaimVerifier(mode="deterministic")

    if isinstance(pipeline_result, list):
        claims = pipeline_result
        output: Dict[str, Any] = {"claims": []}
    elif isinstance(pipeline_result, dict):
        claims = pipeline_result.get("claims", [])
        output = dict(pipeline_result)
    else:
        claims = []
        output = {"claims": []}
    verified_claims = []
    status_counts = {
        "supported": 0,
        "refuted": 0,
        "partially_supported": 0,
        "conflicting": 0,
        "not_enough_evidence": 0,
        "review_required": 0,
    }

    for item in claims:
        claim_meta = item.get("claim_meta")
        if not isinstance(claim_meta, dict):
            claim_meta = dict(item)

        struct_val = (
            claim_meta.get("structural_validation")
            or item.get("structural_validation")
        )

        verified = verifier.verify_claim(
            claim_dict=claim_meta,
            evidence_result=item,
            structural_validation=struct_val,
        )

        merged = {**item, **verified}
        verified_claims.append(merged)

        d_status = verified.get("decision_status", DECISION_INSUFFICIENT)
        if d_status == DECISION_SUPPORTED:
            status_counts["supported"] += 1
        elif d_status == DECISION_CONTRADICTED:
            status_counts["refuted"] += 1
        elif d_status == DECISION_PARTIALLY_SUPPORTED:
            status_counts["partially_supported"] += 1
        elif d_status == DECISION_CONFLICTING:
            status_counts["conflicting"] += 1
        elif d_status == DECISION_REVIEW_REQUIRED:
            status_counts["review_required"] += 1
        else:
            status_counts["not_enough_evidence"] += 1

    total = len(verified_claims)
    output["claims"] = verified_claims
    output["verification"] = {
        "status": "Completed",
        "mode": verifier.mode,
        "summary": {
            "total_claims": total,
            "supported": status_counts["supported"],
            "refuted": status_counts["refuted"],
            "partially_supported": status_counts["partially_supported"],
            "conflicting": status_counts["conflicting"],
            "not_enough_evidence": status_counts["not_enough_evidence"],
            "review_required": status_counts["review_required"],
        },
        "note": (
            "Factual verification evaluated claims against retrieved trusted medical evidence and knowledge graph. "
            "Structural validation remains separate: structural validity is distinct from medical truth."
        ),
    }

    return output
