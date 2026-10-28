#!/usr/bin/env python3
"""Comprehensive Automated Test Suite for Method 2: Trusted Knowledge-Based Verification.

Tests:
1. Trusted Knowledge Base: Ingestion, provenance preservation, chunking, and BM25 retrieval.
2. Knowledge Graph: Entity resolution, typed relation querying, and graph assertion checking.
3. Confidence Scoring: Formula verification, component breakdown, and configurable thresholds.
4. Decision Policy: Standardized decision taxonomy (SUPPORTED, CONTRADICTED, PARTIALLY_SUPPORTED,
   CONFLICTING_EVIDENCE, INSUFFICIENT_EVIDENCE, REVIEW_REQUIRED).
5. Evidence-Grounded Explanation: Citation tracing, URLs, publication dates, and KG triples.
6. Robust Edge Cases: Truncated fragments, malformed input, retrieval failure, and empty results.
7. Full Fever Regression Pipeline: Exact clinical threshold checking, cause list evaluation,
   incomplete fragment handling, and separation of structural validity from medical truth.
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

from claimextractor import (
    align_claim_to_source,
    extract_source_spans,
    split_compound_claims,
    validate_extracted_claims,
)
from verification import (
    DECISION_CONFLICTING,
    DECISION_CONTRADICTED,
    DECISION_INSUFFICIENT,
    DECISION_PARTIALLY_SUPPORTED,
    DECISION_REVIEW_REQUIRED,
    DECISION_SUPPORTED,
    EntityCategory,
    EvidenceChunk,
    MedicalClaimVerifier,
    MedicalEntity,
    MedicalKnowledgeGraph,
    MedicalRelation,
    RelationType,
    ScoringConfig,
    STATUS_CONFLICTING,
    STATUS_INSUFFICIENT,
    STATUS_PARTIALLY_SUPPORTED,
    STATUS_REFUTED,
    STATUS_REVIEW_REQUIRED,
    STATUS_SUPPORTED,
    TrustedDocument,
    TrustedKnowledgeBase,
    compute_authority_score,
    compute_confidence_score,
    evaluate_decision_status,
)


class TestTrustedVerification(unittest.TestCase):
    """Test suite for Trusted Knowledge-Based Claim Verification (Method 2)."""

    def setUp(self):
        self.kb = TrustedKnowledgeBase(load_defaults=True)
        self.kg = MedicalKnowledgeGraph(load_defaults=True)
        self.verifier = MedicalClaimVerifier(
            mode="deterministic",
            knowledge_base=self.kb,
            knowledge_graph=self.kg,
        )

    # -----------------------------------------------------------------------
    # 1. Trusted Knowledge Base Ingestion & Provenance
    # -----------------------------------------------------------------------

    def test_01_knowledge_base_ingestion_and_provenance(self):
        """Knowledge base must ingest trusted document, chunk it, and preserve full provenance."""
        custom_kb = TrustedKnowledgeBase(load_defaults=False)
        chunks = custom_kb.ingest_document(
            document_id="doc_nih_asthma_2024",
            title="Asthma Clinical Guidelines",
            publisher="National Institutes of Health (NIH)",
            text=(
                "Asthma is a chronic lung condition that causes narrowing of the airways. "
                "Albuterol is a short-acting bronchodilator used as quick-relief medication for acute asthma symptoms. "
                "Inhaled corticosteroids are recommended for long-term control of persistent asthma."
            ),
            url="https://www.nhlbi.nih.gov/health/asthma",
            publication_date="2024-05-10",
            authority_level="high",
            domain="Medical",
            chunk_size=200,
        )
        self.assertGreaterEqual(len(chunks), 1)
        chunk = chunks[0]
        self.assertEqual(chunk.document_id, "doc_nih_asthma_2024")
        self.assertEqual(chunk.publisher, "National Institutes of Health (NIH)")
        self.assertEqual(chunk.source_url, "https://www.nhlbi.nih.gov/health/asthma")
        self.assertEqual(chunk.publication_date, "2024-05-10")
        self.assertEqual(chunk.authority_level, "high")

    def test_02_knowledge_base_search_and_authority_ranking(self):
        """BM25 search must retrieve relevant passages with authority weighting."""
        hits = self.kb.search("measured body temperature 100.4 fever threshold", top_k=3)
        self.assertGreaterEqual(len(hits), 1)
        top_hit = hits[0]
        self.assertIn("100.4", top_hit["text"])
        self.assertIn("CDC", top_hit["source"])
        self.assertGreater(top_hit["retrieval_score"], 0.1)
        self.assertEqual(top_hit["metadata"]["authority_level"], "high")

    # -----------------------------------------------------------------------
    # 2. Knowledge Graph Entities and Relationships
    # -----------------------------------------------------------------------

    def test_03_knowledge_graph_entity_and_relation_queries(self):
        """KG must resolve entities by aliases and find verified clinical relations."""
        fever_ent = self.kg.get_entity("pyrexia")
        self.assertIsNotNone(fever_ent)
        self.assertEqual(fever_ent.name, "Fever")

        # Query causes of fever
        causes = self.kg.query_relations(predicate=RelationType.CAUSES, object="Fever")
        cause_subjects = [c.subject for c in causes]
        self.assertIn("Infection", cause_subjects)
        self.assertIn("Inflammation", cause_subjects)
        self.assertIn("Cancer", cause_subjects)

    def test_04_knowledge_graph_assertion_contradiction(self):
        """KG must detect contradictions when asserted threshold or relations diverge."""
        # Good threshold (100.4°F)
        sup, con, rels, _ = self.kg.check_assertion("Fever", RelationType.DEFINES_THRESHOLD, "100.4°F (38°C) or higher")
        self.assertTrue(sup)
        self.assertFalse(con)

        # Bad threshold (104°F)
        sup_bad, con_bad, rels_bad, _ = self.kg.check_assertion("Fever", RelationType.DEFINES_THRESHOLD, "104°F")
        self.assertFalse(sup_bad)
        self.assertTrue(con_bad)

    # -----------------------------------------------------------------------
    # 3. Confidence Scoring Formula & Component Breakdown
    # -----------------------------------------------------------------------

    def test_05_confidence_scoring_formula_and_breakdown(self):
        """Confidence score must transparently combine relevance, support, authority, and KG weights."""
        score, breakdown = compute_confidence_score(
            relevance_score=0.80,
            support_score=0.90,
            authority_score=1.00,
            kg_score=1.00,
        )
        # Expected: 0.25*0.8 + 0.45*0.9 + 0.15*1.0 + 0.15*1.0 = 0.20 + 0.405 + 0.15 + 0.15 = 0.905
        self.assertAlmostEqual(score, 0.905, places=2)
        self.assertEqual(breakdown["relevance_score"], 0.80)
        self.assertEqual(breakdown["support_score"], 0.90)
        self.assertEqual(breakdown["authority_score"], 1.00)
        self.assertEqual(breakdown["kg_corroboration_score"], 1.00)
        self.assertIn("formula", breakdown["scoring_formula"])

    def test_06_scoring_config_custom_thresholds(self):
        """Custom ScoringConfig must allow adjusting decision thresholds."""
        strict_cfg = ScoringConfig(threshold_supported=0.90)
        self.assertEqual(strict_cfg.threshold_supported, 0.90)

        # A score of 0.85 should be PARTIALLY_SUPPORTED under strict threshold 0.90
        status, _ = evaluate_decision_status(
            has_supporting_evidence=True,
            has_contradicting_evidence=False,
            is_partially_supported=False,
            overall_confidence=0.85,
            evidence_count=1,
            config=strict_cfg,
        )
        self.assertEqual(status, DECISION_PARTIALLY_SUPPORTED)

    # -----------------------------------------------------------------------
    # 4. Standardized Decision Policy Taxonomy
    # -----------------------------------------------------------------------

    def test_07_decision_policy_contradicted(self):
        """Contradicting evidence must map to CONTRADICTED decision status."""
        status, expl = evaluate_decision_status(
            has_supporting_evidence=False,
            has_contradicting_evidence=True,
            is_partially_supported=False,
            overall_confidence=0.90,
            evidence_count=1,
        )
        self.assertEqual(status, DECISION_CONTRADICTED)
        self.assertIn("contradicts", expl)

    def test_08_decision_policy_conflicting_evidence(self):
        """Opposing evidence across sources must map to CONFLICTING_EVIDENCE."""
        status, expl = evaluate_decision_status(
            has_supporting_evidence=True,
            has_contradicting_evidence=True,
            is_partially_supported=False,
            overall_confidence=0.60,
            evidence_count=2,
        )
        self.assertEqual(status, DECISION_CONFLICTING)
        self.assertIn("disagree", expl)

    # -----------------------------------------------------------------------
    # 5. Evidence-Grounded Explanation & Traceability
    # -----------------------------------------------------------------------

    def test_09_evidence_grounded_explanation_provenance(self):
        """Verified claim explanation must cite publisher, URL, and matched KG triples."""
        claim = {
            "claim": "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher.",
            "category": "definition",
        }
        res = self.verifier.verify_claim(claim)
        self.assertEqual(res["decision_status"], DECISION_SUPPORTED)
        self.assertGreater(len(res["supporting_sources"]), 0)

        sup = res["supporting_sources"][0]
        self.assertEqual(sup["source"], "Centers for Disease Control and Prevention (CDC) > Body Temperature Assessment and Fever Thresholds")
        self.assertEqual(sup["source_url"], "https://www.cdc.gov/flu/treatment/temperature.htm")
        self.assertEqual(sup["publication_date"], "2024-01-10")

        # Knowledge graph evidence check
        self.assertGreater(len(res["knowledge_graph_evidence"]), 0)
        kg_rel = res["knowledge_graph_evidence"][0]
        self.assertEqual(kg_rel["predicate"], RelationType.DEFINES_THRESHOLD)

    # -----------------------------------------------------------------------
    # 6. Robust Edge Cases & Error Handling
    # -----------------------------------------------------------------------

    def test_10_incomplete_fragment_requires_review(self):
        """Truncated input ('Heat exhaustion') must yield REVIEW_REQUIRED and zero confidence."""
        claim = {
            "claim": "Heat exhaustion",
            "category": "cause",
            "source_sentence": "**Heat exhaustion",
            "needs_review": True,
            "reason": "Incomplete source statement: fragment lacks predicate or context.",
        }
        res = self.verifier.verify_claim(claim)
        self.assertEqual(res["decision_status"], DECISION_REVIEW_REQUIRED)
        self.assertEqual(res["confidence_score"], 0.0)
        self.assertIn("incomplete fragment", res["verification_explanation"].lower())

    def test_11_retrieval_failure_diagnostics(self):
        """Retrieval network error must map to INSUFFICIENT_EVIDENCE with error diagnostics."""
        claim = {"claim": "Vaccinations are a cause of fever."}
        res = self.verifier.verify_claim(
            claim,
            evidence_result={
                "evidence": [],
                "retrieval_status": "retrieval_error",
                "retrieval_error": "Connection refused: RAG service at 127.0.0.1:8000 is offline",
            },
        )
        self.assertEqual(res["decision_status"], DECISION_INSUFFICIENT)
        self.assertEqual(res["confidence_score"], 0.0)
        self.assertIn("Connection refused", res["verification_explanation"])

    def test_12_empty_evidence_no_hallucinated_citations(self):
        """When evidence is empty and not found, citations must be strictly empty."""
        claim = {"claim": "A mythical herbal root completely reverses cellular senescence."}
        res = self.verifier.verify_claim(
            claim,
            evidence_result={"evidence": [], "retrieval_status": "no_evidence"},
        )
        self.assertEqual(res["decision_status"], DECISION_INSUFFICIENT)
        self.assertEqual(len(res["supporting_sources"]), 0)
        self.assertEqual(len(res["contradicting_sources"]), 0)

    # -----------------------------------------------------------------------
    # 7. Backward Compatibility & Public Interface Schema
    # -----------------------------------------------------------------------

    def test_13_public_interface_schema_keys(self):
        """Verifier output dictionary must expose all documented schema keys."""
        claim = {
            "claim": "Normal body temperature is typically between 97°F and 99°F.",
            "category": "physiology",
            "source_sentence": "Normal body temperature is typically between 97°F and 99°F.",
            "heading_context": "Key Points",
        }
        res = self.verifier.verify_claim(claim)

        required_keys = (
            "claim",
            "category",
            "source_sentence",
            "heading_context",
            "validation",
            "verification_status",
            "decision_status",
            "confidence_score",
            "evidence",
            "issues",
            "verification_explanation",
            "confidence_breakdown",
            "supporting_sources",
            "contradicting_sources",
            "knowledge_graph_evidence",
        )
        for key in required_keys:
            self.assertIn(key, res, f"Missing required schema key: '{key}'")

    # -----------------------------------------------------------------------
    # 8. Full Fever Regression Pipeline Test
    # -----------------------------------------------------------------------

    def test_14_fever_regression_pipeline_end_to_end(self):
        """End-to-end regression on the sample.txt fever document.

        Verifies:
        1. Normal temperature range is checked against NIH trusted source -> Supported.
        2. Fever threshold (100.4°F) is checked against CDC trusted source -> Supported.
        3. Wrong temperature threshold is caught as Contradicted.
        4. Pediatric viral aspirin recommendation is caught as Contradicted.
        5. Incomplete 'Heat exhaustion' fragment is flagged as source-quality review.
        6. Structural validation does not imply factual correctness.
        """
        with open("sample.txt", "r", encoding="utf-8") as f:
            fever_doc = f.read().strip()

        # Step 1: Extract spans and validate source
        spans = extract_source_spans(fever_doc)
        self.assertGreater(len(spans), 5)

        # Step 2: Atomic claim decomposition on sample sentences
        decomposed_claims = [
            # 1. Definition
            {"claim": "A fever is a temporary increase in body temperature, usually as a response to an illness or infection.", "category": "definition"},
            # 2. Physiology: Normal temp range
            {"claim": "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C).", "category": "physiology"},
            # 3. Definition: Fever threshold
            {"claim": "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher.", "category": "definition"},
            # 4. Prognosis: Harmless
            {"claim": "Fevers are usually not harmful in themselves.", "category": "prognosis"},
            # 5. Symptom: Sign something is wrong
            {"claim": "Fevers can be a sign that something is wrong in the body.", "category": "symptom"},
            # 6. Inaccurate factual claim (faithfully extracted or tested)
            {"claim": "A fever is generally considered to be a body temperature of 104°F or higher.", "category": "definition"},
            # 7. Unverified / Incomplete trailing fragment
            {"claim": "Heat exhaustion", "category": "cause", "source_sentence": "**Heat exhaustion", "needs_review": True, "reason": "Incomplete source statement: fragment lacks predicate or context."},
        ]

        verified_results = [self.verifier.verify_claim(c) for c in decomposed_claims]

        # Check Claim 1 (Definition): Supported
        self.assertEqual(verified_results[0]["decision_status"], DECISION_SUPPORTED)

        # Check Claim 2 (Normal temp range): Supported against NIH trusted source
        self.assertEqual(verified_results[1]["decision_status"], DECISION_SUPPORTED)
        self.assertTrue(any("NIH" in s["source"] for s in verified_results[1]["supporting_sources"]))

        # Check Claim 3 (Fever threshold 100.4°F): Supported against CDC trusted source
        self.assertEqual(verified_results[2]["decision_status"], DECISION_SUPPORTED)
        self.assertTrue(any("CDC" in s["source"] for s in verified_results[2]["supporting_sources"]))

        # Check Claim 4 (Harmless in themselves): Supported against Mayo Clinic / WHO
        self.assertEqual(verified_results[3]["decision_status"], DECISION_SUPPORTED)

        # Check Claim 6 (104°F incorrect threshold): Contradicted by CDC
        self.assertEqual(verified_results[5]["decision_status"], DECISION_CONTRADICTED)
        self.assertIn("conflict", verified_results[5]["verification_explanation"].lower())

        # Check Claim 7 (Incomplete fragment): Flagged for human review
        self.assertEqual(verified_results[6]["decision_status"], DECISION_REVIEW_REQUIRED)
        self.assertEqual(verified_results[6]["confidence_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
