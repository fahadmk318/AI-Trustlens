#!/usr/bin/env python3
"""Integration test suite for the AI TrustLens RAG integration package.

All tests are mocked: no live Ollama model, RAG API, Qdrant, Neo4j or internet
connection is required. HTTP behaviour is exercised through httpx.MockTransport
(a real httpx.Client over a fake transport), so request/response semantics are
genuine while remaining offline.

Run:  python -m unittest tests.test_rag_integration  (from the repo root)
or    python -m pytest tests/test_rag_integration.py
"""
from __future__ import annotations

import json
import unittest
from unittest import mock

import httpx

from rag_integration.claim_processor import ClaimProcessor
from rag_integration.config import IntegrationConfig
from rag_integration.evidence_formatter import (
    STATUS_NO_EVIDENCE,
    STATUS_OK,
    STATUS_RETRIEVAL_ERROR,
    VERIFICATION_PENDING,
    format_claim_evidence,
)
from rag_integration.model_config import (
    DEPLOYMENT_DEFAULT_MODEL,
    OLLAMA_MODEL_ENV,
    get_ollama_model,
)
from rag_integration.pipeline import TrustLensRAGPipeline
from rag_integration.rag_client import (
    RAGClient,
    RAGConnectionError,
    RAGHTTPError,
    RAGInvalidResponseError,
    RAGTimeoutError,
    _validate_retrieve_response,
)

# ---------------------------------------------------------------------------
# Fixtures: canonical RAG API payloads matching the RAG service's
# app/api/schemas.py contract (same shape as its README sample response)
# ---------------------------------------------------------------------------

EVIDENCE_A = {
    "chunk_id": "doc_diabetes_e9d0a8e1::c0000::1a2b",
    "document_id": "doc_diabetes_e9d0a8e1",
    "text": "Diabetes is a chronic disease that affects how your body turns food into energy.",
    "title": "Diabetes",
    "source_name": "World Health Organization (WHO)",
    "source_url": "https://www.who.int/news-room/fact-sheets/detail/diabetes",
    "publication_date": None,
    "page_number": None,
    "domain": "Medical",
    "retrieval_methods": ["bm25", "vector"],
    "retrieval_score": 0.049,
    "graph_path": ["ent::MedicalConcept::chronic disease"],
}

EVIDENCE_B = {
    "chunk_id": "doc_hypertension_4c5d::c0002::9e8f",
    "document_id": "doc_hypertension_4c5d",
    "text": "High blood pressure usually has no symptoms.",
    "title": "Hypertension",
    "source_name": None,
    "source_url": None,
    "publication_date": None,
    "page_number": 4,
    "domain": "Medical",
    "retrieval_methods": ["graph"],
    "retrieval_score": 0.033,
    "graph_path": ["ent::MedicalConcept::blood pressure"],
}


def retrieve_response(claim: str, evidence, mode: str = "hybrid",
                      per_method=None) -> dict:
    response = {
        "claim": claim,
        "retrieval_mode": mode,
        "evidence": evidence,
        "evidence_found": bool(evidence),
        "latency_ms": 21,
        "diagnostics": {
            "per_method": per_method if per_method is not None else [
                {"method": "vector", "status": "ok" if evidence else "empty",
                 "count": len(evidence), "error": None, "latency_ms": 5},
                {"method": "bm25", "status": "ok" if evidence else "empty",
                 "count": len(evidence), "error": None, "latency_ms": 3},
                {"method": "graph", "status": "empty", "count": 0,
                 "error": None, "latency_ms": 2},
            ],
        },
    }
    return response


CANNED_ANSWER = (
    "A fever is a temporary increase in body temperature, usually as a response "
    "to an illness or infection. Normal body temperature is typically between "
    "97\u00b0F (36.1\u00b0C) and 99\u00b0F (37.2\u00b0C)."
)

FAKE_CLAIMS = [
    {
        "claim": "A fever is a temporary increase in body temperature.",
        "category": "definition",
        "source_sentence": "A fever is a temporary increase in body temperature.",
        "heading_context": "",
        "source_start": 0,
        "source_end": 57,
        "needs_review": False,
        "reason": "",
    },
    {
        "claim": ("Normal body temperature is typically between "
                  "97\u00b0F (36.1\u00b0C) and 99\u00b0F (37.2\u00b0C)."),
        "category": "physiology",
        "source_sentence": ("Normal body temperature is typically between "
                            "97\u00b0F (36.1\u00b0C) and 99\u00b0F (37.2\u00b0C)."),
        "heading_context": "",
        "source_start": 122,
        "source_end": 214,
        "needs_review": False,
        "reason": "",
    },
]

FAKE_MODEL_JSON = json.dumps({"claims": [
    {"claim": FAKE_CLAIMS[0]["claim"], "category": "definition",
     "source_sentence": FAKE_CLAIMS[0]["source_sentence"],
     "heading_context": "", "needs_review": False, "reason": ""},
    {"claim": FAKE_CLAIMS[1]["claim"], "category": "physiology",
     "source_sentence": FAKE_CLAIMS[1]["source_sentence"],
     "heading_context": "", "needs_review": False, "reason": ""},
]})


def mock_transport_handler(request: httpx.Request) -> httpx.Response:
    """Default fake handler: serves /health, /retrieve, /retrieve/batch."""
    if request.method == "GET" and request.url.path == "/health":
        return httpx.Response(200, json={
            "status": "ok", "qdrant": {"ok": True}, "neo4j": {},
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "embedding_backend": "sentence-transformers",
            "bm25_index_size": 104, "warnings": []})
    body = json.loads(request.read() if request.content else b"null")
    if request.method == "POST" and request.url.path == "/retrieve":
        return httpx.Response(200, json=retrieve_response(
            body["claim"], [dict(EVIDENCE_A)]))
    if request.method == "POST" and request.url.path == "/retrieve/batch":
        results = [
            retrieve_response(claim, [dict(EVIDENCE_A if i == 0 else EVIDENCE_B)])
            for i, claim in enumerate(body["claims"])
        ]
        return httpx.Response(200, json={
            "results": results, "retrieval_mode": body.get("retrieval_mode", "hybrid")})
    return httpx.Response(404, json={"detail": "still busy"})


def make_client(handler=mock_transport_handler, **kwargs) -> RAGClient:
    config = IntegrationConfig()
    return RAGClient(config, transport=httpx.MockTransport(handler), **kwargs)


# ---------------------------------------------------------------------------
# Model configuration: deployment default qwen3:8b, local override via env
# ---------------------------------------------------------------------------

class TestModelConfig(unittest.TestCase):

    def test_default_model_is_qwen3_8b_deployment_default(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop(OLLAMA_MODEL_ENV, None)
            self.assertEqual(get_ollama_model(), DEPLOYMENT_DEFAULT_MODEL)
            self.assertEqual(get_ollama_model(), "qwen3:8b")

    def test_local_model_override_via_environment(self):
        with mock.patch.dict(
                "os.environ", {OLLAMA_MODEL_ENV: "qwen3:4b-q4_K_M"}, clear=False):
            self.assertEqual(get_ollama_model(), "qwen3:4b-q4_K_M")

    def test_blank_env_falls_back_to_deployment_default(self):
        with mock.patch.dict("os.environ", {OLLAMA_MODEL_ENV: "  "}, clear=False):
            self.assertEqual(get_ollama_model(), "qwen3:8b")

    def test_claim_processor_points_extractor_at_configured_model(self):
        # The protected claimextractor keeps qwen3:8b on disk; the processor
        # must redirect it at runtime to the config-resolved model, and the
        # ollama.chat call inside extraction must use exactly that model.
        import claimextractor
        original_model = claimextractor.MODEL
        self.addCleanup(setattr, claimextractor, "MODEL", original_model)
        with mock.patch.dict(
                "os.environ", {OLLAMA_MODEL_ENV: "qwen3:4b-q4_K_M"}, clear=False):
            with mock.patch("ollama.chat") as fake_chat:
                fake_chat.return_value = {
                    "message": {"content": json.dumps({"claims": [
                        {"claim": "A fever is a temporary increase in body temperature.",
                         "category": "definition", "source_sentence": "",
                         "heading_context": "", "needs_review": False, "reason": ""}]})}}
                processor = ClaimProcessor()
                self.assertEqual(processor.model_name, "qwen3:4b-q4_K_M")
                result = processor.process(CANNED_ANSWER)
        import claimextractor
        self.assertEqual(claimextractor.MODEL, "qwen3:4b-q4_K_M")
        self.assertIsNone(result["error"])
        _, kwargs = fake_chat.call_args
        self.assertEqual(kwargs.get("model"), "qwen3:4b-q4_K_M")

    def test_claim_processor_defaults_to_deployment_model(self):
        with mock.patch.dict("os.environ", {}, clear=False):
            import os
            os.environ.pop(OLLAMA_MODEL_ENV, None)
            with mock.patch("ollama.chat") as fake_chat:
                fake_chat.return_value = {"message": {"content": json.dumps(
                    {"claims": [{"claim": "Test claim.", "category": "other",
                                 "source_sentence": "", "heading_context": "",
                                 "needs_review": False, "reason": ""}]})}}
                processor = ClaimProcessor()
                self.assertEqual(processor.model_name, "qwen3:8b")
                processor.process(CANNED_ANSWER)
        import claimextractor
        self.assertEqual(claimextractor.MODEL, "qwen3:8b")
        _, kwargs = fake_chat.call_args
        self.assertEqual(kwargs.get("model"), "qwen3:8b")


# ---------------------------------------------------------------------------
# 1-4: RAG API client behaviour
# ---------------------------------------------------------------------------

class TestRAGClient(unittest.TestCase):

    def setUp(self):
        self.client = make_client()
        self.client.open()

    def tearDown(self):
        self.client.close()

    def test_01_successful_retrieve_validates_and_preserves_response(self):
        response = self.client.retrieve("Diabetes is a chronic disease.")
        self.assertEqual(response["claim"], "Diabetes is a chronic disease.")
        self.assertTrue(response["evidence_found"])
        self.assertEqual(len(response["evidence"]), 1)
        ev = response["evidence"][0]
        self.assertEqual(ev["text"], EVIDENCE_A["text"])
        self.assertEqual(ev["retrieval_score"], 0.049)
        self.assertEqual(ev["retrieval_methods"], ["bm25", "vector"])
        self.assertEqual(_validate_retrieve_response(response), response)

    def test_01b_retrieve_sends_expected_request_schema(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["path"] = request.url.path
            captured["body"] = json.loads(request.read())
            return httpx.Response(200, json=retrieve_response(
                captured["body"]["claim"], [dict(EVIDENCE_A)]))

        client = make_client(handler)
        try:
            client.retrieve("some claim", top_k=3, domain="Medical")
        finally:
            client.close()
        self.assertEqual(captured["path"], "/retrieve")
        self.assertEqual(captured["body"]["claim"], "some claim")
        self.assertEqual(captured["body"]["top_k"], 3)
        self.assertEqual(captured["body"]["retrieval_mode"], "hybrid")
        self.assertEqual(captured["body"]["domain"], "Medical")

    def test_02_connection_failure_raises_connection_error(self):
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        client = make_client(broken)
        try:
            with self.assertRaises(RAGConnectionError):
                client.retrieve("any claim")
        finally:
            client.close()

    def test_03_http_error_raises_http_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, json={"detail": "qdrant exploded"})

        client = make_client(handler)
        try:
            with self.assertRaises(RAGHTTPError) as ctx:
                client.retrieve("some claim")
            self.assertEqual(ctx.exception.status_code, 500)
            self.assertEqual(ctx.exception.detail["detail"], "qdrant exploded")
        finally:
            client.close()

    def test_03b_health_returns_status_ok(self):
        health = self.client.health()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["bm25_index_size"], 104)

    def test_03c_timeout_raises_timeout_error(self):
        def slow(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("read timed out", request=request)

        client = make_client(slow)
        try:
            with self.assertRaises(RAGTimeoutError):
                client.retrieve("claim")
        finally:
            client.close()

    def test_04_invalid_response_structure_rejected(self):
        cases = [
            {"nope": True},                                        # missing evidence
            {"claim": "c", "evidence_found": False},               # missing evidence list
            {"claim": 5, "evidence": []},                          # wrong claim type
            {"claim": "c", "evidence_found": False,                # missing 'text'
             "evidence": [{"chunk_id": "x", "document_id": "y"}]},
            {"claim": "c", "evidence_found": "yes",                # wrong lower item type
             "evidence": ["not an object"]},
        ]
        for payload in cases:
            with self.assertRaises(RAGInvalidResponseError):
                _validate_retrieve_response(payload)

    def test_04b_batch_count_mismatch_rejected(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            return httpx.Response(200, json={
                "results": [],
                "retrieval_mode": body.get("retrieval_mode", "hybrid"),
            })

        client = make_client(handler)
        try:
            with self.assertRaises(RAGInvalidResponseError):
                client.retrieve_batch(["claim a", "claim b"])
        finally:
            client.close()

    def test_04c_empty_claim_rejected_locally(self):
        with self.assertRaises(ValueError):
            self.client.retrieve("")

    def test_04d_health_connection_failure_visible(self):
        def broken(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        client = make_client(broken)
        try:
            with self.assertRaises(RAGConnectionError):
                client.health()
        finally:
            client.close()


# ---------------------------------------------------------------------------
# 5-6: claim extraction through the real existing interface (Ollama mocked)
# ---------------------------------------------------------------------------

class TestClaimProcessor(unittest.TestCase):

    def setUp(self):
        # The extractor calls ollama.chat against a local daemon; mocking
        # ollama.chat keeps the extraction logic genuine while staying offline.
        self.patcher = mock.patch("ollama.chat")
        fake_chat = self.patcher.start()
        fake_chat.return_value = {"message": {"content": FAKE_MODEL_JSON}}
        self.addCleanup(self.patcher.stop)

    def test_05_extraction_with_actual_interface(self):
        processor = ClaimProcessor()
        result = processor.process(CANNED_ANSWER)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["claims"]), 2)
        for claim in result["claims"]:
            self.assertIn("claim", claim)
            self.assertIn("category", claim)
            self.assertIn("source_sentence", claim)
            self.assertIn("needs_review", claim)
        self.assertEqual(result["claims"][0]["claim"], FAKE_CLAIMS[0]["claim"])
        self.assertEqual(result["claims"][1]["category"], "physiology")

    def test_06_extraction_failure_makes_error_explicit(self):
        with mock.patch("ollama.chat", side_effect=RuntimeError("boom")):
            processor = ClaimProcessor()
            result = processor.process(CANNED_ANSWER)
        self.assertEqual(result["claims"], [])
        self.assertIn("Claim extraction failed", result["error"])
        self.assertIn("boom", result["error"])

    def test_06b_empty_answer_fails_fast(self):
        processor = ClaimProcessor()
        result = processor.process("   ")
        self.assertEqual(result["claims"], [])
        self.assertIn("empty", result["error"].lower())

    def test_06c_duplicate_claims_removed_first_kept(self):
        with mock.patch(
            "rag_integration.claim_processor.extract_medical_claims",
            return_value={"claims": [dict(FAKE_CLAIMS[0]), dict(FAKE_CLAIMS[1]),
                                     dict(FAKE_CLAIMS[1])], "error": None},
        ):
            processor = ClaimProcessor()
            result = processor.process(CANNED_ANSWER)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["claims"]), 2)
        dup_values = list(result["duplicates"].values())
        self.assertEqual(dup_values, [1])
        self.assertEqual(result["claims"][1]["claim"], FAKE_CLAIMS[1]["claim"])

    def test_06d_non_string_claims_skipped_not_invented(self):
        with mock.patch(
            "rag_integration.claim_processor.extract_medical_claims",
            return_value={"claims": ["oops", None, dict(FAKE_CLAIMS[0])],
                          "error": None},
        ):
            processor = ClaimProcessor()
            result = processor.process(CANNED_ANSWER)
        self.assertIsNone(result["error"])
        self.assertEqual(len(result["claims"]), 1)


# ---------------------------------------------------------------------------
# 7-10: evidence formatting per claim
# ---------------------------------------------------------------------------

class TestEvidenceFormatter(unittest.TestCase):

    def test_07_claim_with_retrieved_evidence(self):
        response = retrieve_response("c", [dict(EVIDENCE_A)])
        out = format_claim_evidence("c", response)
        self.assertEqual(out["retrieval_status"], STATUS_OK)
        self.assertTrue(out["evidence_found"])
        self.assertEqual(len(out["evidence"]), 1)
        ev = out["evidence"][0]
        self.assertEqual(ev["text"], EVIDENCE_A["text"])
        self.assertEqual(ev["retrieval_score"], 0.049)
        self.assertEqual(ev["retrieval_method"], ["bm25", "vector"])
        self.assertEqual(out["verification_status"], VERIFICATION_PENDING)
        self.assertIsNone(out["verification_explanation"])

    def test_08_claim_with_no_relevant_evidence(self):
        response = retrieve_response("c", [])
        out = format_claim_evidence("c", response)
        self.assertEqual(out["retrieval_status"], STATUS_NO_EVIDENCE)
        self.assertFalse(out["evidence_found"])
        self.assertEqual(out["evidence"], [])
        self.assertIsNone(out.get("retrieval_error"))
        self.assertEqual(out["verification_status"], VERIFICATION_PENDING)

    def test_09_evidence_source_and_metadata_preserved(self):
        out = format_claim_evidence("c", retrieve_response("c", [dict(EVIDENCE_A)]))
        meta = out["evidence"][0]["metadata"]
        self.assertEqual(meta["chunk_id"], EVIDENCE_A["chunk_id"])
        self.assertEqual(meta["document_id"], EVIDENCE_A["document_id"])
        self.assertEqual(meta["source_name"], EVIDENCE_A["source_name"])
        self.assertEqual(meta["source_url"], EVIDENCE_A["source_url"])
        self.assertEqual(meta["graph_path"], EVIDENCE_A["graph_path"])
        self.assertEqual(out["evidence"][0]["source"],
                         "World Health Organization (WHO) > Diabetes")

        out_b = format_claim_evidence("c", retrieve_response("c", [dict(EVIDENCE_B)]))
        meta_b = out_b["evidence"][0]["metadata"]
        self.assertIsNone(meta_b["source_name"])
        self.assertIsNone(meta_b["source_url"])
        self.assertEqual(meta_b["page_number"], 4)
        # no source_name: the actual document title is used as the source label
        self.assertEqual(out_b["evidence"][0]["source"], "Hypertension")

    def test_09b_retrieval_error_distinct_from_no_evidence(self):
        # per-claim backend failure reported by the batch endpoint
        response = retrieve_response(
            "c", [], per_method=[
                {"method": "vector", "status": "error", "count": 0,
                 "error": "qdrant lock", "latency_ms": 0},
            ])
        response["diagnostics"]["error"] = "qdrant lock"
        out = format_claim_evidence("c", response)
        self.assertEqual(out["retrieval_status"], STATUS_RETRIEVAL_ERROR)
        self.assertFalse(out["evidence_found"])

        # all requested methods failed, even without a top-level error key
        response2 = retrieve_response(
            "c", [], per_method=[
                {"method": "vector", "status": "error", "count": 0,
                 "error": "x", "latency_ms": 0},
                {"method": "bm25", "status": "error", "count": 0,
                 "error": "y", "latency_ms": 0},
            ])
        out2 = format_claim_evidence("c", response2)
        self.assertEqual(out2["retrieval_status"], STATUS_RETRIEVAL_ERROR)

        # one method failing while others succeed is NOT a total failure
        response3 = retrieve_response(
            "c", [dict(EVIDENCE_A)], per_method=[
                {"method": "vector", "status": "ok", "count": 1,
                 "error": None, "latency_ms": 4},
                {"method": "bm25", "status": "error", "count": 0,
                 "error": "bm25 index corrupt", "latency_ms": 0},
            ])
        out3 = format_claim_evidence("c", response3)
        self.assertEqual(out3["retrieval_status"], STATUS_OK)

    def test_09c_client_level_failure_explicit_not_empty(self):
        out = format_claim_evidence("c", None, retrieval_error="conn refused")
        self.assertEqual(out["retrieval_status"], STATUS_RETRIEVAL_ERROR)
        self.assertEqual(out["retrieval_error"], "conn refused")
        self.assertEqual(out["evidence"], [])
        self.assertEqual(out["verification_status"], VERIFICATION_PENDING)
        # and the retrieval_error is not confused with a quiet empty result
        self.assertNotEqual(out, format_claim_evidence("c", retrieve_response("c", [])))

    def test_10_verification_never_derived_from_retrieval(self):
        # With strong evidence: still Not Evaluated.
        with_ev = format_claim_evidence("c", retrieve_response("c", [dict(EVIDENCE_A)]))
        self.assertEqual(with_ev["verification_status"], VERIFICATION_PENDING)
        self.assertIsNone(with_ev["verification_explanation"])
        # Without any evidence: still Not Evaluated (never 'Not Enough Evidence').
        without_ev = format_claim_evidence("c", retrieve_response("c", []))
        self.assertEqual(without_ev["verification_status"], VERIFICATION_PENDING)
        # On retrieval failure: still Not Evaluated.
        failed = format_claim_evidence("c", None, retrieval_error="boom")
        self.assertEqual(failed["verification_status"], VERIFICATION_PENDING)

    def test_13_malformed_evidence_records_skipped_and_reported(self):
        response = retrieve_response("c", [
            "not-an-object",
            {"chunk_id": "x", "document_id": "y"},  # missing text
            dict(EVIDENCE_A),
        ])
        # client-side validation: exact structural contract
        with self.assertRaises(RAGInvalidResponseError):
            _validate_retrieve_response(response)
        # formatter side: tolerate, skip malformed, keep the valid one
        out = format_claim_evidence("c", response)
        self.assertEqual(len(out["evidence"]), 1)
        self.assertEqual(out["evidence"][0]["text"], EVIDENCE_A["text"])


# ---------------------------------------------------------------------------
# 11 + 14: pipeline behaviour (end to end with mocked HTTP + Ollama)
# ---------------------------------------------------------------------------

class TestPipeline(unittest.TestCase):

    def _pipeline(self, handler=mock_transport_handler, answer_provider=None):
        client = make_client(handler)
        processor = ClaimProcessor()
        pipeline = TrustLensRAGPipeline(
            rag_client=client, claim_processor=processor,
            answer_provider=answer_provider)
        self.addCleanup(client.close)
        return pipeline

    def test_14_end_to_end_with_mocked_rag_responses(self):
        with mock.patch("ollama.chat") as fake_chat:
            fake_chat.return_value = {"message": {"content": FAKE_MODEL_JSON}}
            pipeline = self._pipeline()
            result = pipeline.process_answer("What causes fever?", CANNED_ANSWER)
        self.assertIsNone(result["extraction"]["error"])
        self.assertEqual(len(result["claims"]), 2)
        first = result["claims"][0]
        self.assertEqual(first["claim"], FAKE_CLAIMS[0]["claim"])
        self.assertEqual(first["retrieval_status"], STATUS_OK)
        self.assertEqual(first["evidence"][0]["text"], EVIDENCE_A["text"])
        self.assertEqual(first["claim_meta"]["category"], "definition")
        self.assertEqual(result["verification"]["status"], VERIFICATION_PENDING)
        self.assertEqual(result["answer"], CANNED_ANSWER)
        self.assertEqual(result["question"], "What causes fever?")

    def test_11_and_14_multiple_claims_with_separate_evidence(self):
        with mock.patch("ollama.chat") as fake_chat:
            fake_chat.return_value = {"message": {"content": FAKE_MODEL_JSON}}
            pipeline = self._pipeline()
            result = pipeline.process_answer("Two claims?", CANNED_ANSWER)
        c0, c1 = result["claims"][0], result["claims"][1]
        self.assertEqual(c0["evidence"][0]["metadata"]["document_id"], EVIDENCE_A["document_id"])
        self.assertEqual(c1["evidence"][0]["metadata"]["document_id"], EVIDENCE_B["document_id"])
        self.assertNotEqual(c0["evidence"][0]["metadata"]["chunk_id"],
                            c1["evidence"][0]["metadata"]["chunk_id"])
        self.assertEqual(c1["claim_meta"]["category"], "physiology")

    def test_11_single_claim_retrieval_failure_does_not_crash_pipeline(self):
        # One claim only: the pipeline goes through the single-claim endpoint.
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection reset by peer", request=request)

        pipeline = self._pipeline(handler)
        with mock.patch(
            "rag_integration.claim_processor.extract_medical_claims",
            return_value={"claims": [dict(FAKE_CLAIMS[0])], "error": None},
        ):
            result = pipeline.process_answer("q", CANNED_ANSWER)
        self.assertEqual(len(result["claims"]), 1)
        claim = result["claims"][0]
        self.assertEqual(claim["retrieval_status"], STATUS_RETRIEVAL_ERROR)
        self.assertIn("connection reset by peer", claim["retrieval_error"])
        self.assertEqual(claim["evidence"], [])
        self.assertFalse(claim["evidence_found"])
        self.assertTrue(any("retrieval failed for claim" in e
                            for e in result["errors"]))
        self.assertEqual(claim["verification_status"], VERIFICATION_PENDING)

    def test_11b_per_claim_error_in_batch_isolated(self):
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            if request.url.path == "/retrieve/batch":
                first = retrieve_response(body["claims"][0], [dict(EVIDENCE_A)])
                second = retrieve_response(body["claims"][1], [])
                second["diagnostics"] = {
                    "error": "qdrant lock: already accessed by another instance",
                    "per_claim_failure": True,
                }
                return httpx.Response(200, json={
                    "results": [first, second], "retrieval_mode": "hybrid"})
            return httpx.Response(404)

        with mock.patch(
            "rag_integration.claim_processor.extract_medical_claims",
            return_value={"claims": [dict(c) for c in FAKE_CLAIMS], "error": None},
        ):
            pipeline = self._pipeline(handler)
            result = pipeline.process_answer("q", CANNED_ANSWER)
        c0, c1 = result["claims"]
        self.assertEqual(c0["retrieval_status"], STATUS_OK)
        self.assertEqual(c0["evidence"][0]["metadata"]["document_id"], EVIDENCE_A["document_id"])
        self.assertEqual(c1["retrieval_status"], STATUS_RETRIEVAL_ERROR)
        self.assertIn("qdrant lock", c1["retrieval_error"])
        # no fabricated evidence on the failed claim
        self.assertEqual(c1["evidence"], [])
        self.assertFalse(c1["evidence_found"])

    def test_12_verification_block_reports_pending_stage(self):
        with mock.patch("ollama.chat") as fake_chat:
            fake_chat.return_value = {"message": {"content": FAKE_MODEL_JSON}}
            pipeline = self._pipeline()
            result = pipeline.process_answer("q", CANNED_ANSWER)
        self.assertEqual(result["verification"]["status"], VERIFICATION_PENDING)
        for claim in result["claims"]:
            self.assertEqual(claim["verification_status"], VERIFICATION_PENDING)
            self.assertIsNone(claim["verification_explanation"])
        self.assertIn("separate", result["verification"]["note"])

    def test_14b_question_mode_uses_answer_provider(self):
        pipeline = self._pipeline(answer_provider=lambda q: (CANNED_ANSWER, None))
        with mock.patch("ollama.chat") as fake_chat:
            fake_chat.return_value = {"message": {"content": FAKE_MODEL_JSON}}
            result = pipeline.process_question("What causes fever?")
        self.assertEqual(result["question"], "What causes fever?")
        self.assertEqual(result["answer"], CANNED_ANSWER)
        self.assertEqual(len(result["claims"]), 2)

    def test_14c_answer_provider_failure_reported_no_claims(self):
        pipeline = self._pipeline(
            answer_provider=lambda q: (None, "ollama is down"))
        result = pipeline.process_question("What causes fever?")
        self.assertEqual(result["claims"], [])
        self.assertIsNone(result["answer"])
        self.assertTrue(any("baseline answer generation failed" in e
                            for e in result["errors"]))
        self.assertEqual(result["verification"]["status"], VERIFICATION_PENDING)

    def test_14d_extraction_failure_breaks_nothing_and_reports(self):
        with mock.patch(
            "rag_integration.claim_processor.extract_medical_claims",
            return_value={"claims": [], "error": "Claim extraction failed: boom"},
        ):
            pipeline = self._pipeline()
            result = pipeline.process_answer("q", CANNED_ANSWER)
        self.assertEqual(result["claims"], [])
        self.assertTrue(any("claim extraction failed" in e for e in result["errors"]))
        self.assertEqual(result["extraction"]["error"], "Claim extraction failed: boom")

    def test_11c_client_failure_reports_retrieval_error_for_affected_claim(self):
        # One claim retrieval raises; the pipeline records it explicitly and
        # processes the rest.
        class FlakyClient(RAGClient):
            def retrieve_batch(self, claims, **kwargs):
                raise RAGConnectionError("socket exploded")  # force the
                # batch -> per-claim fallback inside the pipeline

            def retrieve(self, claim, **kwargs):
                if "temporary increase" in claim:  # only the first canned claim
                    raise RAGConnectionError("socket exploded")
                return retrieve_response(claim, [dict(EVIDENCE_A)])

        flaky = FlakyClient(IntegrationConfig(),
                            transport=httpx.MockTransport(mock_transport_handler))
        pipeline = TrustLensRAGPipeline(
            rag_client=flaky, claim_processor=None,
            answer_provider=lambda q: (CANNED_ANSWER, None))
        pipeline.claim_processor = _FakeProcessor()
        try:
            result = pipeline.process_answer("q", CANNED_ANSWER)
        finally:
            flaky.close()
        statuses = [c["retrieval_status"] for c in result["claims"]]
        self.assertIn(STATUS_OK, statuses)
        self.assertIn(STATUS_RETRIEVAL_ERROR, statuses)
        error_claims = [c for c in result["claims"]
                        if c["retrieval_status"] == STATUS_RETRIEVAL_ERROR]
        self.assertTrue(error_claims[0]["retrieval_error"])
        self.assertTrue(any("claim '" in e for e in result["errors"]))


class _FakeProcessor(ClaimProcessor):
    """Extracts canned claims without needing Ollama (pipeline unit test)."""

    def process(self, answer_text: str, source_text: str = None):  # type: ignore[override]
        return {
            "claims": [dict(c) for c in FAKE_CLAIMS],
            "error": None,
            "duplicates": {},
            "notes": [],
        }


if __name__ == "__main__":
    unittest.main()
