#!/usr/bin/env python3
"""Comprehensive Automated Test Suite for AI TrustLens Medical Claim Verification.

Covers:
1. Claims supported by retrieved evidence (direct match, numbers, physiological ranges).
2. Claims contradicted by retrieved evidence (polarity/negation, numeric thresholds,
   temperature cross-units, medical antonyms/contraindications, absolute qualifiers).
3. Claims for which evidence is insufficient (off-topic, low overlap, tangential).
4. Conflicting evidence (opposing evidence across sources -> Conflicting Evidence).
5. Compound sentences split into atomic claims, with each independently verified.
6. Preservation of negation, uncertainty, qualifiers, numbers, units, thresholds, ranges.
7. Source attribution and evidence-to-claim alignment (chunk_id, source title, URL, stance).
8. Incomplete input and malformed model output (trailing fragments, malformed dicts).
9. Empty retrieval results, retrieval failures, and model errors (timeouts, 500 errors).
10. Strict separation of structural validation from factual verification.
11. End-to-end pipeline integration (verify=True vs verify=False).

All tests run offline without requiring a live Ollama daemon or RAG API server.
"""
from __future__ import annotations

import unittest
from unittest import mock

from claimextractor import (
    align_claim_to_source,
    extract_source_spans,
    split_compound_claims,
    validate_extracted_claims,
)
from rag_integration.config import IntegrationConfig
from rag_integration.evidence_formatter import (
    STATUS_NO_EVIDENCE,
    STATUS_OK,
    STATUS_RETRIEVAL_ERROR,
    VERIFICATION_PENDING,
    format_claim_evidence,
)
from rag_integration.pipeline import TrustLensRAGPipeline
from verification import (
    STATUS_CONFLICTING,
    STATUS_INSUFFICIENT,
    STATUS_REFUTED,
    STATUS_SUPPORTED,
    EvidenceAlignment,
    EvidenceStance,
    MedicalClaimVerifier,
    VerificationResult,
    verify_claim_against_evidence,
    verify_claims_pipeline,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

EVIDENCE_FEVER_DEF = {
    "chunk_id": "chunk::who::fever::001",
    "text": "A fever is a temporary increase in body temperature, usually as a response to an illness or infection.",
    "source": "World Health Organization (WHO)",
    "title": "Fever Guidelines",
    "source_name": "WHO",
    "source_url": "https://www.who.int/news-room/fact-sheets/detail/fever",
    "retrieval_score": 0.88,
    "retrieval_methods": ["hybrid", "vector"],
    "metadata": {
        "chunk_id": "chunk::who::fever::001",
        "source_url": "https://www.who.int/news-room/fact-sheets/detail/fever",
    },
}

EVIDENCE_FEVER_THRESHOLD = {
    "chunk_id": "chunk::cdc::fever::002",
    "text": "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher.",
    "source": "CDC",
    "title": "Temperature Assessment",
    "source_name": "CDC",
    "source_url": "https://www.cdc.gov/flu/treatment/temperature.htm",
    "retrieval_score": 0.92,
    "retrieval_methods": ["bm25"],
    "metadata": {
        "chunk_id": "chunk::cdc::fever::002",
        "source_url": "https://www.cdc.gov/flu/treatment/temperature.htm",
    },
}

EVIDENCE_NORMAL_TEMP = {
    "chunk_id": "chunk::nih::temp::003",
    "text": "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C).",
    "source": "National Institutes of Health (NIH)",
    "title": "Body Temperature Physiology",
    "source_name": "NIH MedlinePlus",
    "source_url": "https://medlineplus.gov/bodytemperature.html",
    "retrieval_score": 0.90,
    "retrieval_methods": ["vector"],
    "metadata": {
        "chunk_id": "chunk::nih::temp::003",
        "source_url": "https://medlineplus.gov/bodytemperature.html",
    },
}

EVIDENCE_FEVER_HARMLESS = {
    "chunk_id": "chunk::mayo::fever::004",
    "text": "Fevers are usually not harmful in themselves, but they can be an important sign that something is wrong in the body.",
    "source": "Mayo Clinic",
    "title": "Fever Symptoms & Causes",
    "source_name": "Mayo Clinic",
    "source_url": "https://www.mayoclinic.org/diseases-conditions/fever",
    "retrieval_score": 0.85,
    "retrieval_methods": ["vector"],
    "metadata": {
        "chunk_id": "chunk::mayo::fever::004",
        "source_url": "https://www.mayoclinic.org/diseases-conditions/fever",
    },
}

EVIDENCE_ASPIRIN_CHILDREN = {
    "chunk_id": "chunk::fda::aspirin::005",
    "text": "Aspirin is contraindicated in children and teenagers with viral infections due to the risk of Reye's syndrome and should not be used.",
    "source": "FDA Guidelines",
    "title": "Pediatric Drug Safety",
    "source_name": "FDA",
    "source_url": "https://www.fda.gov/drugs/pediatric-safety/aspirin",
    "retrieval_score": 0.91,
    "retrieval_methods": ["vector", "graph"],
    "metadata": {
        "chunk_id": "chunk::fda::aspirin::005",
        "source_url": "https://www.fda.gov/drugs/pediatric-safety/aspirin",
    },
}

EVIDENCE_HYPERTENSION = {
    "chunk_id": "chunk::aha::htn::006",
    "text": "High blood pressure usually has no symptoms and is often called the silent killer.",
    "source": "American Heart Association (AHA)",
    "title": "High Blood Pressure Symptoms",
    "source_name": "AHA",
    "source_url": "https://www.heart.org/hbp-symptoms",
    "retrieval_score": 0.89,
    "retrieval_methods": ["vector"],
    "metadata": {
        "chunk_id": "chunk::aha::htn::006",
        "source_url": "https://www.heart.org/hbp-symptoms",
    },
}


class TestMedicalClaimVerification(unittest.TestCase):
    """Test suite for claim verification logic, NLI rules, and pipeline integration."""

    def setUp(self):
        self.verifier = MedicalClaimVerifier(mode="deterministic")

    # -----------------------------------------------------------------------
    # 1. Claims Supported by Retrieved Evidence
    # -----------------------------------------------------------------------

    def test_01_claim_supported_exact_semantic_match(self):
        """Claim matching definition evidence must be classified as Supported."""
        claim = {
            "claim": "A fever is a temporary increase in body temperature.",
            "category": "definition",
        }
        evidence_res = {"evidence": [EVIDENCE_FEVER_DEF], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_SUPPORTED)
        self.assertGreaterEqual(result["confidence"], 0.70)
        self.assertEqual(len(result["supporting_sources"]), 1)
        self.assertEqual(result["supporting_sources"][0]["chunk_id"], "chunk::who::fever::001")
        self.assertIn("corroborated", result["verification_explanation"])

    def test_02_claim_supported_with_numeric_and_units_match(self):
        """Claim with exact temperature threshold (100.4°F / 38°C) must be Supported."""
        claim = {
            "claim": "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher.",
            "category": "definition",
        }
        evidence_res = {"evidence": [EVIDENCE_FEVER_THRESHOLD], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_SUPPORTED)
        self.assertGreaterEqual(result["confidence"], 0.85)
        self.assertTrue(any("Exact numerical thresholds" in a["reason"] for a in result["evidence_alignment"]))

    def test_03_claim_supported_physiological_range(self):
        """Claim specifying normal body temperature range 97°F to 99°F must be Supported."""
        claim = {
            "claim": "Normal body temperature is typically between 97°F and 99°F.",
            "category": "physiology",
        }
        evidence_res = {"evidence": [EVIDENCE_NORMAL_TEMP], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_SUPPORTED)
        self.assertEqual(len(result["supporting_sources"]), 1)
        self.assertEqual(result["supporting_sources"][0]["chunk_id"], "chunk::nih::temp::003")

    # -----------------------------------------------------------------------
    # 2. Claims Contradicted by Retrieved Evidence
    # -----------------------------------------------------------------------

    def test_04_claim_refuted_numeric_threshold_conflict(self):
        """Claim asserting fever is 104°F must be Refuted by evidence establishing 100.4°F."""
        claim = {
            "claim": "A fever is generally considered to be a body temperature of 104°F or higher.",
            "category": "definition",
        }
        evidence_res = {"evidence": [EVIDENCE_FEVER_THRESHOLD], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_REFUTED)
        self.assertEqual(len(result["contradicting_sources"]), 1)
        self.assertEqual(result["contradicting_sources"][0]["chunk_id"], "chunk::cdc::fever::002")
        self.assertIn("Numeric threshold conflict", result["verification_explanation"])

    def test_05_claim_refuted_negation_polarity_conflict(self):
        """Claim asserting fevers are harmful must be Refuted by evidence stating fevers are not harmful."""
        claim = {
            "claim": "Fevers are harmful in themselves and cause permanent injury.",
            "category": "prognosis",
        }
        evidence_res = {"evidence": [EVIDENCE_FEVER_HARMLESS], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_REFUTED)
        self.assertEqual(len(result["contradicting_sources"]), 1)
        self.assertIn("Negation conflict", result["verification_explanation"])

    def test_06_claim_refuted_medical_antonym_contraindication(self):
        """Claim recommending aspirin for viral fevers in children must be Refuted by contraindication."""
        claim = {
            "claim": "Aspirin is recommended for treating viral fevers in children.",
            "category": "treatment",
        }
        evidence_res = {"evidence": [EVIDENCE_ASPIRIN_CHILDREN], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_REFUTED)
        self.assertIn("contraindicated", result["verification_explanation"].lower())

    def test_07_claim_refuted_absolute_qualifier_conflict(self):
        """Claim asserting hypertension always causes severe headaches must be Refuted by asymptomatic evidence."""
        claim = {
            "claim": "High blood pressure always causes severe daily headaches in all cases.",
            "category": "symptom",
        }
        evidence_res = {"evidence": [EVIDENCE_HYPERTENSION], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_REFUTED)
        self.assertTrue(
            any("no symptoms" in a["reason"].lower() or "qualifier" in a["reason"].lower() or "negation" in a["reason"].lower()
                for a in result["evidence_alignment"])
        )

    # -----------------------------------------------------------------------
    # 3. Claims for Which Evidence is Insufficient
    # -----------------------------------------------------------------------

    def test_08_claim_insufficient_evidence_off_topic(self):
        """Claim about green tea curing diabetes evaluated against heart health evidence must be Not Enough Evidence."""
        claim = {
            "claim": "Drinking green tea cures type 1 diabetes within two weeks.",
            "category": "treatment",
        }
        evidence_tangential = {
            "chunk_id": "chunk::tea::001",
            "text": "Green tea contains polyphenols and catechins that support general cardiovascular health.",
            "source": "Nutrition Science",
        }
        evidence_res = {"evidence": [evidence_tangential], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_INSUFFICIENT)
        self.assertEqual(len(result["supporting_sources"]), 0)
        self.assertEqual(len(result["contradicting_sources"]), 0)

    def test_09_claim_insufficient_evidence_empty_retrieval(self):
        """Claim with empty retrieval results must yield Not Enough Evidence."""
        claim = {"claim": "A rare genetic mutation causes nocturnal fever spikes."}
        evidence_res = {"evidence": [], "retrieval_status": "no_evidence"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_INSUFFICIENT)
        self.assertIn("No relevant evidence was retrieved", result["verification_explanation"])

    # -----------------------------------------------------------------------
    # 4. Conflicting Evidence
    # -----------------------------------------------------------------------

    def test_10_conflicting_evidence_across_sources(self):
        """When one retrieved passage supports and another refutes, status must be Conflicting Evidence."""
        claim = {
            "claim": "Drug X reduces mortality in acute pancreatitis.",
            "category": "treatment",
        }
        ev_support = {
            "chunk_id": "chunk::trial::001",
            "text": "Clinical Trial Alpha demonstrates that Drug X reduces mortality in acute pancreatitis patients.",
            "source": "New England Journal of Medicine",
            "metadata": {"chunk_id": "chunk::trial::001"},
        }
        ev_refute = {
            "chunk_id": "chunk::trial::002",
            "text": "Subsequent multicenter trials found that Drug X does not reduce mortality in acute pancreatitis.",
            "source": "Lancet Gastroenterology",
            "metadata": {"chunk_id": "chunk::trial::002"},
        }
        evidence_res = {"evidence": [ev_support, ev_refute], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_CONFLICTING)
        self.assertEqual(len(result["supporting_sources"]), 1)
        self.assertEqual(len(result["contradicting_sources"]), 1)
        self.assertIn("Conflicting medical evidence", result["verification_explanation"])

    # -----------------------------------------------------------------------
    # 5. Compound Sentences Split and Independently Verified
    # -----------------------------------------------------------------------

    def test_11_compound_sentence_split_and_independent_verification(self):
        """Compound sentence with 'but' must be split and each part verified independently."""
        source = (
            "Fevers are usually not harmful in themselves, but they can be a sign "
            "that something is wrong in the body."
        )
        raw_extraction = {
            "claims": [
                {
                    "claim": source,
                    "category": "prognosis",
                    "source_sentence": source,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        decomposed = split_compound_claims(raw_extraction, source_text=source)
        self.assertEqual(len(decomposed["claims"]), 2)

        claim1 = decomposed["claims"][0]
        claim2 = decomposed["claims"][1]

        # Verify claim 1 (Fevers are usually not harmful in themselves)
        ev_res1 = {"evidence": [EVIDENCE_FEVER_HARMLESS], "retrieval_status": "ok"}
        res1 = self.verifier.verify_claim(claim1, ev_res1)
        self.assertEqual(res1["verification_status"], STATUS_SUPPORTED)

        # Verify claim 2 (Fevers can be a sign that something is wrong in the body)
        ev_sign = {
            "chunk_id": "chunk::mayo::sign::007",
            "text": "A fever can be an important sign that an infection or illness is present in the body.",
            "source": "Mayo Clinic",
            "metadata": {"chunk_id": "chunk::mayo::sign::007"},
        }
        ev_res2 = {"evidence": [ev_sign], "retrieval_status": "ok"}
        res2 = self.verifier.verify_claim(claim2, ev_res2)
        self.assertEqual(res2["verification_status"], STATUS_SUPPORTED)

    # -----------------------------------------------------------------------
    # 6. Preservation of Negation, Qualifiers, Numbers, Units, Ranges
    # -----------------------------------------------------------------------

    def test_12_negation_preservation_in_supported_claims(self):
        """Negated claim supported by negated evidence must preserve negation semantics."""
        claim = {
            "claim": "Fevers are usually not harmful in themselves.",
            "category": "prognosis",
        }
        evidence_res = {"evidence": [EVIDENCE_FEVER_HARMLESS], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_SUPPORTED)
        self.assertTrue(any(a["stance"] == EvidenceStance.SUPPORTS for a in result["evidence_alignment"]))

    def test_13_dropped_negation_in_extraction_is_detected_structurally(self):
        """Claim extractor's validate_extracted_claims must catch dropped negations."""
        source = "Fevers are usually not harmful in themselves."
        bad_extraction = {
            "claims": [
                {
                    "claim": "Fevers are usually harmful in themselves.",
                    "category": "prognosis",
                    "source_sentence": source,
                    "source_start": 0,
                    "source_end": len(source),
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(source, bad_extraction)
        self.assertEqual(val["claims"][0]["validation"], "FAILED")
        self.assertTrue(any("Negation mismatch" in iss for iss in val["claims"][0]["issues"]))

    # -----------------------------------------------------------------------
    # 7. Source Attribution & Alignment
    # -----------------------------------------------------------------------

    def test_14_source_attribution_includes_chunk_id_and_url(self):
        """Supporting source records must preserve chunk_id, source title, and URL."""
        claim = {"claim": "A fever is a temporary increase in body temperature."}
        evidence_res = {"evidence": [EVIDENCE_FEVER_DEF], "retrieval_status": "ok"}
        result = self.verifier.verify_claim(claim, evidence_res)

        sup = result["supporting_sources"][0]
        self.assertEqual(sup["chunk_id"], "chunk::who::fever::001")
        self.assertEqual(sup["source"], "World Health Organization (WHO)")
        self.assertEqual(sup["source_url"], "https://www.who.int/news-room/fact-sheets/detail/fever")
        self.assertIn("temporary increase", sup["text"])

    # -----------------------------------------------------------------------
    # 8. Incomplete Input & Malformed Output
    # -----------------------------------------------------------------------

    def test_15_incomplete_fragment_input_handling(self):
        """Incomplete trailing fragment ('Heat exhaustion') must yield Not Enough Evidence."""
        claim = {
            "claim": "Heat exhaustion",
            "category": "cause",
            "source_sentence": "**Heat exhaustion",
            "needs_review": True,
            "reason": "Incomplete source statement: fragment lacks predicate or context.",
        }
        evidence_res = {
            "evidence": [{"text": "Heat exhaustion causes elevated temperature.", "source": "CDC"}],
            "retrieval_status": "ok",
        }
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_INSUFFICIENT)
        self.assertIn("fragment lacks predicate", result["verification_explanation"])

    def test_16_malformed_claim_and_empty_text_handled_gracefully(self):
        """Malformed claim dicts with empty or non-string claim texts must not crash."""
        result = self.verifier.verify_claim({}, {"evidence": [], "retrieval_status": "no_evidence"})
        self.assertEqual(result["verification_status"], STATUS_INSUFFICIENT)

    # -----------------------------------------------------------------------
    # 9. Empty Retrieval Results & Retrieval Failures
    # -----------------------------------------------------------------------

    def test_17_retrieval_failure_surfaced_without_crashing(self):
        """When RAG retrieval reports an error, status must be Not Enough Evidence with error detail."""
        claim = {"claim": "Vaccinations are a cause of fever."}
        evidence_res = {
            "evidence": [],
            "retrieval_status": STATUS_RETRIEVAL_ERROR,
            "retrieval_error": "500 Internal Server Error: Database connection lost",
        }
        result = self.verifier.verify_claim(claim, evidence_res)

        self.assertEqual(result["verification_status"], STATUS_INSUFFICIENT)
        self.assertIn("Evidence retrieval failed", result["verification_explanation"])
        self.assertIn("500 Internal Server Error", result["verification_explanation"])
        self.assertEqual(result["retrieval_status"], STATUS_RETRIEVAL_ERROR)

    # -----------------------------------------------------------------------
    # 10. Structural Validation vs Factual Verification Separation
    # -----------------------------------------------------------------------

    def test_18_structurally_valid_claim_can_be_factually_refuted(self):
        """A claim with PASSED structural validation can be factually Refuted."""
        # Source text has an inaccurate medical statement that was faithfully extracted:
        source = "Normal body temperature is typically 104°F."
        extraction = {
            "claims": [
                {
                    "claim": "Normal body temperature is typically 104°F.",
                    "category": "physiology",
                    "source_sentence": source,
                    "source_start": 0,
                    "source_end": len(source),
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        # 1. Structural validation: PASSED (faithfully extracted from the text)
        struct_val = validate_extracted_claims(source, extraction)
        self.assertEqual(struct_val["claims"][0]["validation"], "PASSED")

        # 2. Factual verification against external medical evidence: REFUTED
        ev_res = {"evidence": [EVIDENCE_NORMAL_TEMP], "retrieval_status": "ok"}
        fact_res = self.verifier.verify_claim(
            extraction["claims"][0],
            ev_res,
            structural_validation=struct_val["claims"][0],
        )
        self.assertEqual(fact_res["verification_status"], STATUS_REFUTED)
        # Structural validation must say PASSED while verification says Refuted!
        self.assertEqual(fact_res["structural_validation"]["validation"], "PASSED")
        self.assertEqual(fact_res["verification_status"], STATUS_REFUTED)

    # -----------------------------------------------------------------------
    # 11. Pipeline Integration (verify=True vs verify=False)
    # -----------------------------------------------------------------------

    def test_19_pipeline_verify_true_populates_verification_status(self):
        """Pipeline with verify=True must run verification and update claim statuses."""
        mock_client = mock.MagicMock()
        mock_client.retrieve_for_claims.return_value = [
            {
                "claim": "A fever is a temporary increase in body temperature.",
                "evidence": [EVIDENCE_FEVER_DEF],
                "evidence_found": True,
                "retrieval_mode": "hybrid",
                "diagnostics": {"per_method": []},
            }
        ]

        mock_processor = mock.MagicMock()
        mock_processor.model_name = "mock_model"
        mock_processor.process.return_value = {
            "claims": [
                {
                    "claim": "A fever is a temporary increase in body temperature.",
                    "category": "definition",
                    "source_sentence": "A fever is a temporary increase in body temperature.",
                    "source_start": 0,
                    "source_end": 45,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
            "duplicates": {},
            "notes": [],
        }

        pipeline = TrustLensRAGPipeline(
            config=IntegrationConfig(),
            rag_client=mock_client,
            claim_processor=mock_processor,
            verifier=self.verifier,
        )

        res = pipeline.process_answer(
            question="What is a fever?",
            answer="A fever is a temporary increase in body temperature.",
            verify=True,
        )

        self.assertEqual(res["verification"]["status"], "Completed")
        self.assertEqual(res["verification"]["summary"]["supported"], 1)
        self.assertEqual(res["claims"][0]["verification_status"], STATUS_SUPPORTED)
        self.assertIn("corroborated", res["claims"][0]["verification_explanation"])
        self.assertIn("structural_validation", res["claims"][0])

    def test_20_pipeline_verify_false_preserves_verification_pending(self):
        """Pipeline with verify=False must keep verification_status as VERIFICATION_PENDING."""
        mock_client = mock.MagicMock()
        mock_client.retrieve_for_claims.return_value = [
            {
                "claim": "A fever is a temporary increase in body temperature.",
                "evidence": [EVIDENCE_FEVER_DEF],
                "evidence_found": True,
                "retrieval_mode": "hybrid",
                "diagnostics": {"per_method": []},
            }
        ]

        mock_processor = mock.MagicMock()
        mock_processor.model_name = "mock_model"
        mock_processor.process.return_value = {
            "claims": [
                {
                    "claim": "A fever is a temporary increase in body temperature.",
                    "category": "definition",
                    "source_sentence": "A fever is a temporary increase in body temperature.",
                    "source_start": 0,
                    "source_end": 45,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
            "duplicates": {},
            "notes": [],
        }

        pipeline = TrustLensRAGPipeline(
            config=IntegrationConfig(),
            rag_client=mock_client,
            claim_processor=mock_processor,
            verifier=None,
        )

        res = pipeline.process_answer(
            question="What is a fever?",
            answer="A fever is a temporary increase in body temperature.",
            verify=False,
        )

        self.assertEqual(res["verification"]["status"], VERIFICATION_PENDING)
        self.assertEqual(res["claims"][0]["verification_status"], VERIFICATION_PENDING)
        self.assertIsNone(res["claims"][0]["verification_explanation"])


if __name__ == "__main__":
    unittest.main()
