#!/usr/bin/env python3
"""AI TrustLens - Interactive Medical QA & Real-Time Verification CLI.

Provides an interactive clinical terminal session that:
1. Accepts user medical questions or quick-select benchmark prompts.
2. Generates baseline answers using local Ollama (qwen3:8b).
3. Extracts atomic medical claims using the clinical NER extractor.
4. Retrieves hybrid evidence (Vector + BM25 + Knowledge Graph) from the RAG service.
5. Performs factual verification, confidence scoring, and structural validation.
6. Renders color-coded verification scorecards, confidence meters, and citations.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import threading
import time
from typing import Any, Dict, List, Optional

from rag_integration import (
    RAGClient,
    RAGClientError,
    RAGConnectionError,
    RAGHTTPError,
    RAGInvalidResponseError,
    RAGTimeoutError,
    TrustLensRAGPipeline,
    get_config,
)

# -----------------------------------------------------------------------------
# ANSI Colors & Formatting
# -----------------------------------------------------------------------------
USE_COLOR = sys.stdout.isatty()

RESET = "\033[0m" if USE_COLOR else ""
BOLD = "\033[1m" if USE_COLOR else ""
DIM = "\033[2m" if USE_COLOR else ""
UNDERLINE = "\033[4m" if USE_COLOR else ""

# Foreground Colors
CYAN = "\033[96m" if USE_COLOR else ""
BLUE = "\033[94m" if USE_COLOR else ""
GREEN = "\033[92m" if USE_COLOR else ""
YELLOW = "\033[93m" if USE_COLOR else ""
RED = "\033[91m" if USE_COLOR else ""
MAGENTA = "\033[95m" if USE_COLOR else ""
WHITE = "\033[97m" if USE_COLOR else ""

# Background Badges
BG_GREEN = "\033[42m\033[30m" if USE_COLOR else ""
BG_RED = "\033[41m\033[97m" if USE_COLOR else ""
BG_YELLOW = "\033[43m\033[30m" if USE_COLOR else ""
BG_BLUE = "\033[44m\033[97m" if USE_COLOR else ""
BG_MAGENTA = "\033[45m\033[97m" if USE_COLOR else ""
BG_GRAY = "\033[100m\033[97m" if USE_COLOR else ""

# Preset Clinical Questions for Instant Testing
SAMPLE_QUESTIONS = [
    "What are the recommended first-line medications for hypertension?",
    "Is Metformin used as a first-line treatment for type 2 diabetes?",
    "Can antibiotics be used to treat viral infections like the common cold or flu?",
    "What are the common symptoms and screening guidelines for cervical cancer?",
    "What is high blood pressure and why is it called the silent killer?",
]


class Spinner:
    """Threaded progress spinner with elapsed timer."""

    def __init__(self, message: str = "Processing..."):
        self.message = message
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_time = 0.0

    def start(self) -> "Spinner":
        self._start_time = time.time()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()
        return self

    def _spin(self) -> None:
        glyphs = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])
        while not self._stop_event.is_set():
            elapsed = time.time() - self._start_time
            glyph = next(glyphs)
            sys.stdout.write(f"\r  {CYAN}{glyph}{RESET} {self.message} {DIM}({elapsed:.1f}s){RESET}   ")
            sys.stdout.flush()
            time.sleep(0.08)

    def update_message(self, new_message: str) -> None:
        self.message = new_message

    def stop(self, success_message: Optional[str] = None) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        elapsed = time.time() - self._start_time
        if success_message:
            sys.stdout.write(f"\r  {GREEN}✓{RESET} {success_message} {DIM}({elapsed:.1f}s){RESET}\n")
        else:
            sys.stdout.write("\r" + " " * 80 + "\r")
        sys.stdout.flush()


def format_confidence_bar(conf: float, width: int = 10) -> str:
    """Format confidence as a visual progress bar [████████░░] 80%."""
    pct = max(0.0, min(1.0, conf))
    filled = int(round(pct * width))
    empty = width - filled
    color = GREEN if pct >= 0.75 else (YELLOW if pct >= 0.45 else RED)
    bar = "█" * filled + "░" * empty
    return f"{color}[{bar}] {pct*100:4.1f}%{RESET}"


def format_badge(status: str) -> str:
    """Return color badge corresponding to decision status."""
    st = status.upper().replace("_", " ").strip()
    if st == "SUPPORTED":
        return f"{BG_GREEN} SUPPORTED {RESET}"
    elif st == "REFUTED":
        return f"{BG_RED} REFUTED {RESET}"
    elif st in ("PARTIALLY SUPPORTED",):
        return f"{BG_YELLOW} PARTIALLY SUPPORTED {RESET}"
    elif st == "CONFLICTING":
        return f"{BG_YELLOW} CONFLICTING {RESET}"
    elif st in ("REVIEW REQUIRED",):
        return f"{BG_MAGENTA} REVIEW REQUIRED {RESET}"
    elif st in ("NOT ENOUGH EVIDENCE", "INSUFFICIENT EVIDENCE"):
        return f"{BG_GRAY} NOT ENOUGH EVIDENCE {RESET}"
    else:
        return f"{BG_GRAY} {st} {RESET}"


def print_banner(rag_status: Dict[str, Any]) -> None:
    """Print the interactive header with live RAG and model information."""
    print()
    print(f"{CYAN}╔═══════════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
    print(f"{CYAN}║{BOLD}{WHITE}             AI TrustLens — Interactive Medical QA & Claim Verification            {RESET}{CYAN}║{RESET}")
    print(f"{CYAN}╚═══════════════════════════════════════════════════════════════════════════════════════════╝{RESET}")

    qdrant_points = rag_status.get("qdrant", {}).get("points", 0)
    entities_count = rag_status.get("neo4j", {}).get("counts", {}).get("entities", 0)
    docs_count = rag_status.get("neo4j", {}).get("counts", {}).get("documents", 0)
    bm25_size = rag_status.get("bm25_index_size", 0)

    print(f"  {DIM}Local LLM  :{RESET} {BOLD}qwen3:8b (Ollama){RESET}")
    print(f"  {DIM}RAG Engine :{RESET} {GREEN}Online{RESET} at http://127.0.0.1:8000")
    print(f"  {DIM}Corpus KB  :{RESET} {BOLD}{docs_count}{RESET} documents · {BOLD}{qdrant_points}{RESET} Qdrant vectors · {BOLD}{bm25_size}{RESET} BM25 chunks")
    print(f"  {DIM}Graph Store:{RESET} {BOLD}{entities_count}{RESET} medical entities & relations (JSON memory fallback)")
    print(f"  {DIM}Verifier   :{RESET} MedicalClaimVerifier (Deterministic + BM25/Vector alignment + KG)")
    print(f"{DIM}{'─' * 91}{RESET}")


def print_verification_report(result: Dict[str, Any]) -> None:
    """Render a clean, formatted clinical verification report."""
    question = result.get("question", "")
    answer = result.get("answer", "")
    claims = result.get("claims", [])
    verification = result.get("verification", {})
    summary = verification.get("summary", {})

    print()
    print(f"{BOLD}{BLUE}┌── Medical Question ────────────────────────────────────────────────────────────────────────┐{RESET}")
    print(f"{WHITE}{BOLD}  {question}{RESET}")
    print(f"{BOLD}{BLUE}└── Baseline Model Answer (Ollama / qwen3:8b) ───────────────────────────────────────────────┘{RESET}")
    for line in answer.strip().split("\n"):
        print(f"  {DIM}{line}{RESET}")
    print()

    # Verification Summary Scorecard
    total = summary.get("total_claims", len(claims))
    supported = summary.get("supported", 0)
    refuted = summary.get("refuted", 0)
    review = summary.get("review_required", 0)
    insufficient = summary.get("not_enough_evidence", 0)
    conflicting = summary.get("conflicting", 0)

    print(f"{BOLD}Verification Summary Scorecard:{RESET}")
    print(
        f"  Total Claims: {BOLD}{total}{RESET}  │  "
        f"{GREEN}✓ Supported: {supported}{RESET}  │  "
        f"{RED}✗ Refuted: {refuted}{RESET}  │  "
        f"{MAGENTA}! Review Req: {review}{RESET}  │  "
        f"{DIM}— Insufficient: {insufficient}{RESET}"
    )
    print(f"{DIM}{'─' * 91}{RESET}")

    # Render each claim
    for idx, c in enumerate(claims, 1):
        claim_text = c.get("claim", "")
        status = c.get("decision_status") or c.get("verification_status", "UNKNOWN")
        category = c.get("category", "medical")
        confidence = c.get("confidence_score", c.get("confidence", 0.0))
        explanation = c.get("explanation") or c.get("verification_explanation", "")

        badge = format_badge(status)
        conf_bar = format_confidence_bar(confidence)

        print(f"\n  {BOLD}{idx}. Claim:{RESET} \"{WHITE}{claim_text}{RESET}\"")
        print(f"     Status: {badge}   Category: {CYAN}[{category}]{RESET}   Confidence: {conf_bar}")

        # Structural issues
        struct_val = c.get("structural_validation", {})
        if not struct_val.get("is_structurally_valid", True):
            issues = struct_val.get("issues", [])
            for issue in issues:
                print(f"     {MAGENTA}⚠ Structural issue:{RESET} {issue}")

        # Explanation
        if explanation:
            print(f"     {DIM}Rationale:{RESET} {explanation}")

        # Top supporting evidence
        evidence_list = c.get("evidence", [])
        supporting = [e for e in evidence_list if e.get("stance") == "supports"]
        if supporting:
            top_ev = supporting[0]
            src_title = top_ev.get("source", "Medical Reference")
            src_url = top_ev.get("source_url", "")
            ev_text = top_ev.get("text", "").replace("\n", " ").strip()
            if len(ev_text) > 180:
                ev_text = ev_text[:177] + "..."
            print(f"     {GREEN}Supporting Source:{RESET} {BOLD}{src_title}{RESET}")
            if src_url:
                print(f"       {DIM}URL:{RESET} {BLUE}{src_url}{RESET}")
            print(f"       {DIM}Excerpt:{RESET} \"{ev_text}\"")

        # Knowledge Graph Triples
        kg_ev = c.get("knowledge_graph_evidence", [])
        if kg_ev:
            for triple in kg_ev[:2]:
                subj = triple.get("subject", "")
                pred = triple.get("predicate", "")
                obj = triple.get("object", "")
                cit = triple.get("citation", "")
                print(f"     {CYAN}KG Triple:{RESET} ({subj}) —[{pred}]→ ({obj}) {DIM}[{cit}]{RESET}")

    print(f"\n{DIM}{'═' * 91}{RESET}")


def interactive_session() -> None:
    """Main interactive terminal loop."""
    config = get_config()
    os.environ["TRUSTLENS_ENABLE_VERIFICATION"] = "1"

    # Step 1: Health check
    client = RAGClient(config)
    try:
        health = client.health()
    except Exception as exc:
        print(f"\n{RED}[ERROR] Cannot connect to RAG service at {config.base_url}: {exc}{RESET}")
        print(f"{YELLOW}Hint: Start the RAG service first:{RESET}")
        print("  cd AI-Trustlens-RAG && .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n")
        return

    print_banner(health)

    # Initialize pipeline
    pipeline = TrustLensRAGPipeline(config=config)

    last_result: Optional[Dict[str, Any]] = None

    try:
        while True:
            print(f"\n{BOLD}Select a question or type your own:{RESET}")
            for i, q in enumerate(SAMPLE_QUESTIONS, 1):
                print(f"  {CYAN}[{i}]{RESET} {q}")
            print(f"  {DIM}[q] Quit / Exit{RESET}")
            print(f"  {DIM}[j] View raw JSON of last result{RESET}")
            print(f"  {DIM}[s <filename>] Save last result to JSON file{RESET}\n")

            try:
                user_input = input(f"{BOLD}{GREEN}AI-TrustLens ❯ {RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                print(f"\n{CYAN}Exiting session. Goodbye!{RESET}\n")
                break

            if not user_input:
                continue

            if user_input.lower() in ("q", "quit", "exit"):
                print(f"\n{CYAN}Exiting session. Goodbye!{RESET}\n")
                break

            if user_input.lower() == "j":
                if last_result:
                    print(json.dumps(last_result, indent=2, ensure_ascii=False))
                else:
                    print(f"{YELLOW}No previous result available yet.{RESET}")
                continue

            if user_input.lower().startswith("s "):
                if not last_result:
                    print(f"{YELLOW}No previous result available to save.{RESET}")
                    continue
                path = user_input[2:].strip()
                try:
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(last_result, f, indent=2, ensure_ascii=False)
                    print(f"{GREEN}Saved result to {path}{RESET}")
                except Exception as e:
                    print(f"{RED}Could not save file: {e}{RESET}")
                continue

            # Determine query text
            if user_input.isdigit() and 1 <= int(user_input) <= len(SAMPLE_QUESTIONS):
                question = SAMPLE_QUESTIONS[int(user_input) - 1]
            else:
                question = user_input

            print(f"\n{BOLD}Processing Question:{RESET} {CYAN}{question}{RESET}")

            spinner = Spinner("Querying local Ollama (qwen3:8b) & verifying claims...").start()
            t0 = time.time()
            try:
                result = pipeline.process_question(question, verify=True)
                spinner.stop(f"Pipeline finished successfully in {time.time() - t0:.1f}s")
                last_result = result
                print_verification_report(result)
            except RAGConnectionError as e:
                spinner.stop()
                print(f"\n{RED}[RAG Connection Error] {e}{RESET}")
            except RAGClientError as e:
                spinner.stop()
                print(f"\n{RED}[RAG API Error] {e}{RESET}")
            except Exception as e:
                spinner.stop()
                print(f"\n{RED}[Error] Pipeline execution failed: {e}{RESET}")

    finally:
        pipeline.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="AI TrustLens Interactive Clinical QA Session")
    parser.add_argument("--question", "-q", help="Run a single question and exit immediately")
    parser.add_argument("--answer", "-a", help="Verify answer text directly without calling Ollama QA")
    parser.add_argument("--answer-file", dest="answer_file", help="Verify answer text from file")
    parser.add_argument("--out", "-o", help="Save the JSON output to file")
    args = parser.parse_args()

    if args.question or args.answer or args.answer_file:
        config = get_config()
        pipeline = TrustLensRAGPipeline(config=config)
        try:
            if args.question:
                spinner = Spinner("Processing question with Ollama & verifying...").start()
                res = pipeline.process_question(args.question, verify=True)
            else:
                if args.answer_file:
                    with open(args.answer_file, "r", encoding="utf-8") as f:
                        text = f.read()
                else:
                    text = args.answer or ""
                spinner = Spinner("Extracting claims & verifying...").start()
                res = pipeline.process_answer("", text, verify=True)

            spinner.stop("Done.")
            print_verification_report(res)
            if args.out:
                with open(args.out, "w", encoding="utf-8") as f:
                    json.dump(res, f, indent=2, ensure_ascii=False)
                print(f"{GREEN}Saved output to {args.out}{RESET}")
            return 0
        finally:
            pipeline.close()
    else:
        interactive_session()
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
