#!/usr/bin/env python3
"""
Comprehensive Unit & Regression Test Suite for AI TrustLens Medical Claim Extractor.
Tests all requirements specified for semantic fidelity, decomposition, and validation.
"""

import unittest
import json
from claimextractor import (
    extract_source_spans,
    align_claim_to_source,
    split_compound_claims,
    validate_extracted_claims,
    MEDICAL_CATEGORIES,
)


class TestClaimExtractor(unittest.TestCase):

    def setUp(self):
        with open("sample.txt", "r", encoding="utf-8") as f:
            self.fever_source = f.read().strip()
        self.spans = extract_source_spans(self.fever_source)

    def test_01_two_independent_assertions_with_but(self):
        """Sentence with 'but' must be split into two complete assertions."""
        raw_extraction = {
            "claims": [
                {
                    "claim": "Fevers are usually not harmful in themselves, but they can be a sign that something is wrong in the body.",
                    "category": "prognosis",
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        res = split_compound_claims(raw_extraction, source_text=self.fever_source)
        self.assertEqual(len(res["claims"]), 2)
        claim1 = res["claims"][0]["claim"]
        claim2 = res["claims"][1]["claim"]
        self.assertIn("not harmful in themselves", claim1)
        self.assertIn("sign that something is wrong", claim2)
        self.assertTrue(claim2.startswith("Fevers can be a sign"))

    def test_02_examples_joined_by_and_must_not_split(self):
        """Lists of examples joined by 'and' or 'or' must remain intact."""
        raw_claim = "It is a common sign that the body is fighting off something, such as a virus, bacteria, or other foreign substances."
        raw_extraction = {
            "claims": [{"claim": raw_claim, "category": "symptom", "needs_review": False, "reason": ""}],
            "error": None,
        }
        res = split_compound_claims(raw_extraction, source_text=self.fever_source)
        self.assertEqual(len(res["claims"]), 1)
        self.assertEqual(res["claims"][0]["claim"], raw_claim)

    def test_03_adverbial_clause_must_not_create_fragment(self):
        """Adverbial modifier ('usually as a response...') must not be split into an incomplete fragment."""
        raw_claim = "A fever is a temporary increase in body temperature, usually as a response to an illness or infection."
        raw_extraction = {
            "claims": [{"claim": raw_claim, "category": "definition", "needs_review": False, "reason": ""}],
            "error": None,
        }
        res = split_compound_claims(raw_extraction, source_text=self.fever_source)
        self.assertEqual(len(res["claims"]), 1)
        self.assertEqual(res["claims"][0]["claim"], raw_claim)

    def test_04_negation_preservation_and_mismatch(self):
        """Dropping a negation present in the source sentence must be caught as FAILED."""
        # Correctly preserved:
        good_claim = {
            "claims": [
                {
                    "claim": "Fevers are usually not harmful in themselves.",
                    "category": "prognosis",
                    "source_sentence": "Fevers are usually not harmful in themselves, but they can be a sign that something is wrong in the body.",
                    "source_start": 438,
                    "source_end": 549,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        good_val = validate_extracted_claims(self.fever_source, good_claim)
        self.assertEqual(good_val["claims"][0]["validation"], "PASSED")

        # Negation flipped/dropped:
        bad_claim = {
            "claims": [
                {
                    "claim": "Fevers are usually harmful in themselves.",
                    "category": "prognosis",
                    "source_sentence": "Fevers are usually not harmful in themselves, but they can be a sign that something is wrong in the body.",
                    "source_start": 438,
                    "source_end": 549,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        bad_val = validate_extracted_claims(self.fever_source, bad_claim)
        self.assertEqual(bad_val["claims"][0]["validation"], "FAILED")
        self.assertTrue(any("Negation mismatch" in iss for iss in bad_val["claims"][0]["issues"]))

    def test_05_qualifiers_preservation_and_drift(self):
        """Dropping 'in some cases' from Cancer must be caught as a review issue."""
        drift_claim = {
            "claims": [
                {
                    "claim": "Cancer can cause fever.",
                    "category": "cause",
                    "source_sentence": "**Cancer** (in some cases)",
                    "source_start": 686,
                    "source_end": 714,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, drift_claim)
        self.assertEqual(val["claims"][0]["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("Scope qualifier dropped" in iss for iss in val["claims"][0]["issues"]))

    def test_06_medical_numbers_units_ranges(self):
        """Numbers and units must be validated strictly against source sentence."""
        good_claim = {
            "claims": [
                {
                    "claim": "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C).",
                    "category": "physiology",
                    "source_sentence": "**Normal body temperature** is typically between **97°F (36.1°C)** and **99°F (37.2°C)**.",
                    "source_start": 252,
                    "source_end": 343,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, good_claim)
        self.assertEqual(val["claims"][0]["validation"], "PASSED")

        # Altered number (e.g. 102°F instead of 100.4°F):
        bad_num_claim = {
            "claims": [
                {
                    "claim": "A fever is generally considered to be a body temperature of 102°F or higher.",
                    "category": "definition",
                    "source_sentence": "A **fever** is generally considered to be a body temperature of **100.4°F (38°C)** or higher.",
                    "source_start": 343,
                    "source_end": 438,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val_bad = validate_extracted_claims(self.fever_source, bad_num_claim)
        self.assertEqual(val_bad["claims"][0]["validation"], "FAILED")
        self.assertTrue(any("Altered or ungrounded numeric token" in iss for iss in val_bad["claims"][0]["issues"]))

    def test_07_incomplete_trailing_input(self):
        """Incomplete trailing fragment ('Heat exhaustion') must be detected and flagged."""
        trailing_claim = {
            "claims": [
                {
                    "claim": "Heat exhaustion can cause fever.",
                    "category": "cause",
                    "source_sentence": "**Heat exhaustion",
                    "source_start": 732,
                    "source_end": 751,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, trailing_claim)
        self.assertEqual(val["claims"][0]["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("incomplete fragment" in iss for iss in val["claims"][0]["issues"]))

    def test_08_duplicate_claims_detected(self):
        """Duplicate claims must be flagged."""
        dupe_input = {
            "claims": [
                {"claim": "A fever is a temporary increase in body temperature.", "category": "definition", "needs_review": False, "reason": ""},
                {"claim": "A fever is a temporary increase in body temperature.", "category": "definition", "needs_review": False, "reason": ""},
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, dupe_input)
        self.assertTrue(any("Duplicate claim" in iss for iss in val["claims"][1]["issues"]))

    def test_09_semantic_inflation_common_cause(self):
        """Asserting 'common' where source sentence has no such claim must be caught."""
        inflated_claim = {
            "claims": [
                {
                    "claim": "Cancer is a common cause of fever.",
                    "category": "cause",
                    "source_sentence": "**Cancer** (in some cases)",
                    "source_start": 686,
                    "source_end": 714,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, inflated_claim)
        self.assertEqual(val["claims"][0]["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("Semantic inflation" in iss for iss in val["claims"][0]["issues"]))

    def test_10_unaligned_claim_detected(self):
        """Claims with no reliable source alignment must be flagged."""
        unaligned_claim = {
            "claims": [
                {
                    "claim": "Aspirin is recommended for all fevers in infants.",
                    "category": "treatment",
                    "source_sentence": "",
                    "source_start": None,
                    "source_end": None,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, unaligned_claim)
        self.assertEqual(val["claims"][0]["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("cannot establish reliable alignment" in iss for iss in val["claims"][0]["issues"]))

    def test_11_full_fever_regression_pipeline(self):
        """End-to-end regression test on the fever passage."""
        decomposed_fever_claims = {
            "claims": [
                {"claim": "A fever is a temporary increase in body temperature, usually as a response to an illness or infection.", "category": "definition", "needs_review": False, "reason": ""},
                {"claim": "A fever is a common sign that the body is fighting off something, such as a virus, bacteria, or other foreign substances.", "category": "symptom", "needs_review": False, "reason": ""},
                {"claim": "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C).", "category": "physiology", "needs_review": False, "reason": ""},
                {"claim": "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher.", "category": "definition", "needs_review": False, "reason": ""},
                {"claim": "Fevers are usually not harmful in themselves.", "category": "prognosis", "needs_review": False, "reason": ""},
                {"claim": "Fevers can be a sign that something is wrong in the body.", "category": "symptom", "needs_review": False, "reason": ""},
                {"claim": "Infections (viral, bacterial, fungal, or parasitic) are a cause of fever.", "category": "cause", "needs_review": False, "reason": ""},
                {"claim": "Inflammation (such as from autoimmune diseases) is a cause of fever.", "category": "cause", "needs_review": False, "reason": ""},
                {"claim": "Cancer (in some cases) is a cause of fever.", "category": "cause", "needs_review": False, "reason": ""},
                {"claim": "Vaccinations are a cause of fever.", "category": "cause", "needs_review": False, "reason": ""},
                {"claim": "Heat exhaustion is a cause of fever.", "category": "cause", "needs_review": False, "reason": ""},
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, decomposed_fever_claims)
        # 10 complete claims should PASS
        passed_claims = [c for c in val["claims"] if c["validation"] == "PASSED"]
        self.assertEqual(len(passed_claims), 10)
        # The truncated trailing fragment ("Heat exhaustion") must be flagged REVIEW_REQUIRED
        heat_claim = next(c for c in val["claims"] if "heat exhaustion" in c["claim"].lower())
        self.assertEqual(heat_claim["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("incomplete fragment" in iss for iss in heat_claim["issues"]))

    def test_12_heading_context_inheritance_for_list_items(self):
        """List items inherit parent heading context so asserting 'common' under 'Common Causes of Fever:' passes."""
        inherited_claim = {
            "claims": [
                {
                    "claim": "Infections (viral, bacterial, fungal, or parasitic) are a common cause of fever.",
                    "category": "cause",
                    "source_sentence": "**Infections** (viral, bacterial, fungal, or parasitic)",
                    "source_start": 576,
                    "source_end": 633,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, inherited_claim)
        c0 = val["claims"][0]
        self.assertEqual(c0["validation"], "PASSED")
        self.assertEqual(c0["heading_context"], "Common Causes of Fever")
        self.assertEqual(c0["source_sentence"], "**Infections** (viral, bacterial, fungal, or parasitic)")
        self.assertEqual(c0["source_start"], 576)
        self.assertEqual(c0["source_end"], 633)

    def test_13_unsupported_qualifier_semantic_inflation_without_heading_support(self):
        """Asserting 'common' when neither the item nor the heading contains 'common' must be flagged."""
        source_doc = "Causes of Fever:\n* Infections (viral, bacterial, fungal, or parasitic)\n* Vaccinations\n"
        claim_input = {
            "claims": [
                {
                    "claim": "Infections are a common cause of fever.",
                    "category": "cause",
                    "source_sentence": "Infections (viral, bacterial, fungal, or parasitic)",
                    "heading_context": "Causes of Fever",
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(source_doc, claim_input)
        c0 = val["claims"][0]
        self.assertEqual(c0["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("Semantic inflation" in iss for iss in c0["issues"]))

    def test_14_unsupported_qualifiers_always_typically(self):
        """Adding unsupported high-certainty qualifiers like 'always' must be flagged as REVIEW_REQUIRED."""
        claim_input = {
            "claims": [
                {
                    "claim": "Vaccinations always cause fever.",
                    "category": "cause",
                    "source_sentence": "**Vaccinations**",
                    "heading_context": "Common Causes of Fever",
                    "source_start": 714,
                    "source_end": 732,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, claim_input)
        c0 = val["claims"][0]
        self.assertEqual(c0["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("Unsupported qualifier" in iss for iss in c0["issues"]))

    def test_15_restrictive_qualifier_conflicts_with_heading_common(self):
        """Claim asserting 'common' for an item with restrictive '(in some cases)' must be flagged."""
        claim_input = {
            "claims": [
                {
                    "claim": "Cancer is a common cause of fever.",
                    "category": "cause",
                    "source_sentence": "**Cancer** (in some cases)",
                    "source_start": 686,
                    "source_end": 714,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, claim_input)
        c0 = val["claims"][0]
        self.assertEqual(c0["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("Semantic inflation" in iss for iss in c0["issues"]))
        self.assertTrue(any("Scope qualifier dropped" in iss for iss in c0["issues"]))

    def test_16_incomplete_fragment_preserved_and_flagged(self):
        """Incomplete trailing fragment must preserve fragment text and be flagged REVIEW_REQUIRED."""
        fragment_claim = {
            "claims": [
                {
                    "claim": "Heat exhaustion",
                    "category": "other",
                    "source_sentence": "**Heat exhaustion",
                    "source_start": 732,
                    "source_end": 751,
                    "needs_review": True,
                    "reason": "Incomplete source statement: fragment lacks predicate or context.",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(self.fever_source, fragment_claim)
        c0 = val["claims"][0]
        self.assertEqual(c0["validation"], "REVIEW_REQUIRED")
        self.assertTrue(any("incomplete fragment" in iss for iss in c0["issues"]))

    def test_17_complete_statement_without_final_punctuation_not_rejected(self):
        """Complete statements must not be rejected or marked truncated solely because they lack final punctuation."""
        doc_no_period = "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C)"
        spans = extract_source_spans(doc_no_period)
        self.assertEqual(len(spans), 1)
        self.assertFalse(spans[0]["is_truncated"])

        claim_input = {
            "claims": [
                {
                    "claim": "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C)",
                    "category": "physiology",
                    "source_sentence": doc_no_period,
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        val = validate_extracted_claims(doc_no_period, claim_input)
        self.assertEqual(val["claims"][0]["validation"], "PASSED")

    def test_18_compound_claim_elliptical_subject_resolution(self):
        """Compound claims with elliptical verb in second clause must resolve subject."""
        raw_extraction = {
            "claims": [
                {
                    "claim": "Fevers are usually not harmful in themselves, but can be a sign that something is wrong in the body.",
                    "category": "prognosis",
                    "needs_review": False,
                    "reason": "",
                }
            ],
            "error": None,
        }
        res = split_compound_claims(raw_extraction, source_text=self.fever_source)
        self.assertEqual(len(res["claims"]), 2)
        claim1 = res["claims"][0]["claim"]
        claim2 = res["claims"][1]["claim"]
        self.assertEqual(claim1, "Fevers are usually not harmful in themselves.")
        self.assertEqual(claim2, "Fevers can be a sign that something is wrong in the body.")

    def test_19_multiline_asterisk_markdown_headings_and_lists(self):
        """Markdown headings with asterisk list items must correctly associate headings and detect fragments."""
        doc = (
            "Common Causes of Fever:\n\n"
            "* Infections (viral, bacterial, fungal, or parasitic)\n"
            "* Inflammation (such as from autoimmune diseases)\n"
            "* Cancer (in some cases)\n"
            "* Vaccinations\n"
            "* Heat exhaustion"
        )
        spans = extract_source_spans(doc)
        # Header + 5 items = 6 spans
        self.assertEqual(len(spans), 6)
        self.assertTrue(spans[0]["is_header"])
        self.assertEqual(spans[0]["text"], "Common Causes of Fever:")

        for i in range(1, 6):
            self.assertEqual(spans[i]["heading_context"], "Common Causes of Fever")
            self.assertFalse(spans[i]["is_header"])

        # Final item 'Heat exhaustion' is a fragment lacking a predicate
        self.assertTrue(spans[5]["is_truncated"])
        self.assertEqual(spans[5]["text"], "Heat exhaustion")


if __name__ == "__main__":
    unittest.main()
