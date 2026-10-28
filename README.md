# AI TrustLens — Medical QA, Claim Extraction & Trust Verification Platform

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-green.svg)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Qwen3--8B-orange.svg)](https://ollama.ai/)
[![Tests](https://img.shields.io/badge/Tests-134%2F134%20Passing-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-Academic%20Project-lightgrey.svg)]()

> **Final-Year B.Tech Major Project**  
> An end-to-end medical hallucination detection, claim extraction, hybrid evidence retrieval, and factual verification platform running entirely on local infrastructure.
<!--  -->
---

## Table of Contents
1. [Overview & Architecture](#overview--architecture)
2. [Key Subsystems](#key-subsystems)
3. [Project Directory Structure](#project-directory-structure)
4. [Prerequisites & Environments](#prerequisites--environments)
5. [Installation & Setup](#installation--setup)
   - [A. Ollama & Local LLM](#a-ollama--local-llm)
   - [B. RAG Microservice Setup & Document Ingestion](#b-rag-microservice-setup--document-ingestion)
   - [C. Root Verification Environment Setup](#c-root-verification-environment-setup)
6. [Running the Application](#running-the-application)
   - [1. Interactive Web Dashboard UI (Recommended for Presentations)](#1-interactive-web-dashboard-ui-recommended-for-presentations)
   - [2. Interactive Clinical QA Terminal CLI](#2-interactive-clinical-qa-terminal-cli)
   - [3. Pipeline CLI (Automated QA & File Verification)](#3-pipeline-cli-automated-qa--file-verification)
   - [4. Baseline Terminal QA Loop](#4-baseline-terminal-qa-loop)
   - [5. Standalone Claim Extractor](#5-standalone-claim-extractor)
7. [Testing & Quality Assurance](#testing--quality-assurance)
8. [Clinical Safety Disclaimer](#clinical-safety-disclaimer)

---

## Overview & Architecture

Modern Large Language Models (LLMs) hallucinate clinically inaccurate information with high linguistic confidence. **AI TrustLens** introduces a multi-stage audit pipeline that separates generation from verification:

```
                                 ┌────────────────────────────────────────────────────────┐
                                 │                   USER QUESTION                        │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │         Stage 1: Baseline QA (Ollama / Qwen3-8B)       │
                                 │      Generates raw, unverified medical answer text     │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │       Stage 2: Clinical Claim Extractor & Splitter     │
                                 │   • Decomposes answer into atomic medical assertions   │
                                 │   • Classifies across 22 standardized medical categories│
                                 │   • Validates structural fidelity & detects fragments  │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │            Stage 3: Hybrid RAG Evidence Engine         │
                                 │   • Vector Search: MiniLM-L6-v2 embeddings + Qdrant    │
                                 │   • Lexical Search: Local BM25 ranking                 │
                                 │   • Graph Store: Medical entity/relation triples       │
                                 │   • Fusion: Reciprocal Rank Fusion (RRF, k=60)         │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │        Stage 4: Factual Verification & Scoring         │
                                 │   • Evidence stance classification (Support/Refute)    │
                                 │   • Multi-factor confidence score formulation          │
                                 │   • Knowledge Graph corroboration check                │
                                 │   • Decision: SUPPORTED | REFUTED | REVIEW REQUIRED    │
                                 └──────────────────────────┬─────────────────────────────┘
                                                            ▼
                                 ┌────────────────────────────────────────────────────────┐
                                 │             Interactive Report / Structured JSON       │
                                 │       Verification scorecard, confidence meters & URLs │
                                 └────────────────────────────────────────────────────────┘
```

---

## Key Subsystems

| Subsystem | Components | Technology Stack |
| :--- | :--- | :--- |
| **Baseline QA** | `app.py` | Local Ollama, Qwen3-8B, greedy sampling (`temp=0`, `max_tokens=200`) |
| **Claim Extractor** | `claimextractor.py` | RegEx segmentation, 22-category taxonomy, span alignment, hedge/qualifier detection |
| **Integration Bridge** | `rag_integration/` | Asynchronous HTTP client (`httpx`), schema adapters, error boundaries |
| **RAG Evidence Service** | `AI-Trustlens-RAG/` | FastAPI, Qdrant (persistent vector DB), BM25, SentenceTransformers, Graph Store (Neo4j / JSON snapshot) |
| **Verification Engine** | `verification/` | MedicalClaimVerifier, stance analyzer, semantic overlap, confidence scoring, KG corroboration |
| **Interactive Terminal** | `interactive_trustlens.py` | Live threaded progress spinner, color-coded status badges, scorecard, KG triple viewer |
| **Pipeline CLI** | `run_trustlens_with_rag.py` | End-to-end command-line runner supporting question mode, manual answer mode, and file input |

---

## Project Directory Structure

```
Claim verification/
├── app.py                          # Stage 1: Baseline terminal QA loop (Ollama + Qwen3-8B)
├── claimextractor.py               # Medical claim extractor, splitter & structural validator
├── test_claimextractor.py          # Unit tests for claim extractor (19 tests)
├── interactive_trustlens.py        # Interactive clinical QA terminal session with live scorecard
├── run_trustlens_with_rag.py       # End-to-end CLI entrypoint for question/answer verification
├── web_app.py                      # FastAPI Web Dashboard server with async task polling
├── sample.txt                      # Sample clinical answer for testing (fever & infection case)
├── requirements.txt                # Consolidated root dependencies (Ollama, FastAPI, Uvicorn, pytest)
├── .env.example                    # Environment variable template
│
├── web/                            # Modern Web Dashboard assets
│   └── static/                     # Glassmorphic UI with dynamic claim highlights
│       ├── index.html              # Single-page application structure
│       ├── css/style.css           # Vanilla CSS design system (clinical dark theme)
│       └── js/app.js               # Reactive frontend controller with async polling
│
├── rag_integration/                # Bridge connecting Stage 1 to the RAG Subsystem
│   ├── config.py                   # Lazy env-backed integration configuration
│   ├── rag_client.py               # Resilient httpx client for /retrieve and /health
│   ├── claim_processor.py          # Adapter for claimextractor.py (keeps baseline untouched)
│   ├── evidence_formatter.py       # Converts RAG response into structured evidence models
│   ├── pipeline.py                 # Pipeline orchestrator: Answer -> Claims -> Evidence -> Verify
│   ├── model_config.py             # Ollama model name resolver
│   └── requirements-integration.txt# Integration dependency manifest
│
├── verification/                   # Method 2: Trusted KB + Knowledge Graph Verifier
│   ├── verifier.py                 # MedicalClaimVerifier (Deterministic / LLM / Hybrid modes)
│   ├── knowledge_base.py           # Trusted document BM25 retriever & source registry
│   ├── knowledge_graph.py          # Medical entity/relation graph & corroboration checks
│   └── scoring.py                  # Multi-factor confidence formula & decision status mapping
│
├── tests/                          # Root integration & verification test suite (70 tests)
│   ├── test_claim_verification.py  # Claim processing and structural validation tests
│   ├── test_rag_integration.py     # HTTP client, pipeline, and error handling tests
│   └── test_trusted_verification.py# Verification engine, scoring, and stance classification tests
│
└── AI-Trustlens-RAG/               # Standalone RAG & Knowledge Graph evidence retrieval microservice
    ├── app/
    │   ├── main.py                 # FastAPI application with startup health checks
    │   ├── config.py               # Service configuration (Qdrant, Neo4j, BM25 paths)
    │   ├── api/                    # Endpoints (/health, /retrieve, /retrieve/batch) & schemas
    │   ├── embeddings/             # SentenceTransformers (MiniLM-L6-v2) singleton
    │   ├── vector_store/           # Qdrant persistent storage engine
    │   ├── graph_store/            # Neo4j client with local JSON memory-fallback
    │   ├── retrieval/              # Vector, BM25, Graph, and Hybrid (RRF) retrievers
    │   └── ingestion/              # Ingestion service, cleaning, chunking & document loaders
    ├── documents/                  # Reference medical corpus (WHO, CDC, AHA guidelines)
    │   ├── medical/                # Antibiotics, diabetes, hypertension, fever
    │   └── cancer/                 # Breast, cervical, colorectal, lung, cancer overview
    ├── scripts/ingest_documents.py # Ingestion CLI script
    └── tests/                      # RAG microservice unit & integration tests (45 tests)
```

---

## Prerequisites & Environments

The project employs **two clean virtual environments**:
1. **Root Virtual Environment (`./.venv`)**: Handles the baseline QA, claim extraction, verification engine, and interactive CLI.
2. **RAG Virtual Environment (`AI-Trustlens-RAG/.venv`)**: Dedicated to the heavy scientific RAG stack (FastAPI, Qdrant client, PyTorch, SentenceTransformers).

### System Prerequisites
- **Operating System:** Linux (Ubuntu/Debian tested) or macOS / Windows WSL2.
- **Python:** Python 3.10, 3.11, or 3.12+
- **Ollama:** Installed and running locally ([ollama.ai](https://ollama.ai/))

---

## Installation & Setup

### A. Ollama & Local LLM

1. Ensure Ollama is running:
   ```bash
   ollama list
   ```
2. Pull the default medical QA model (`qwen3:8b`):
   ```bash
   ollama pull qwen3:8b
   ```

---

### B. RAG Microservice Setup & Document Ingestion

1. Navigate to the RAG directory:
   ```bash
   cd "AI-Trustlens-RAG"
   ```
2. Create and activate the RAG virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
   *(On Windows PowerShell: `.\.venv\Scripts\Activate.ps1`)*

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Configure local environment variables:
   ```bash
   cp .env.example .env
   ```

5. Ingest reference medical documents (WHO, CDC, AHA):
   ```bash
   python scripts/ingest_documents.py --path documents/medical --domain Medical
   python scripts/ingest_documents.py --path documents/cancer --domain Medical
   ```

6. Start the RAG FastAPI microservice:
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
   ```
   *Keep this service running in a background terminal. It serves hybrid retrieval at `http://127.0.0.1:8000`.*

---

### C. Root Verification Environment Setup

In a new terminal, navigate to the project root:

```bash
cd "Claim verification"
```

1. Create and activate the root virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. (Optional) Configure environment variables:
   ```bash
   cp .env.example .env
   ```

---

## Running the Application

### 1. Interactive Web Dashboard UI (Recommended for Presentations)

Launch the visual web dashboard with real-time claim highlights, reliability gauges, and source cards:

```bash
python web_app.py
```
Open **[http://localhost:8080](http://localhost:8080)** in your browser:
- **Interactive Question Answering:** Enter a medical query or click clinical presets (Hypertension, Diabetes, Antibiotics, Cancer).
- **Direct Text Auditing:** Paste any text/answer to audit and verify individual claims immediately.
- **Visual Claim Highlights:** Click any inline color-coded claim in the generated answer to jump directly to its verification card.
- **Reliability Gauges & Scorecard:** Visual metrics for Supported, Refuted, Review Required, and Insufficient Evidence assertions.
- **Evidence & Knowledge Graph Cards:** Direct hyperlinks to WHO/CDC/AHA guidelines and relational triples.
- **Asynchronous Task Queue:** Built-in background worker execution and polling eliminates proxy timeouts when sharing through public tunnels.

#### Sharing with Teammates:
- **On the same Wi-Fi / Local Network:** Teammates can connect directly via `http://<your-local-ip>:8080` (e.g. `http://172.20.10.3:8080`).
- **Over the Internet via Cloudflare Tunnel:**
  ```bash
  ./cloudflared tunnel --url http://localhost:8080
  ```
  Share the generated `https://*.trycloudflare.com` URL with your team.

---

### 2. Interactive Clinical QA Terminal CLI

Run the interactive clinical verification terminal console:

```bash
python interactive_trustlens.py
```

- Choose from 5 preset medical queries or type your own custom question.
- Watch live token generation with real-time progress timers.
- Inspect color-coded verification badges:
  - `[ SUPPORTED ]` (Green) — Corroborated by WHO/CDC/AHA guidelines.
  - `[ REFUTED ]` (Red) — Contradicted by medical literature.
  - `[ REVIEW REQUIRED ]` (Magenta) — Incomplete sentence fragment or structural issue.
  - `[ NOT ENOUGH EVIDENCE ]` (Gray) — Not found in current knowledge base.
- View matched Knowledge Graph relation triples and cited source excerpts.
- Type `j` to view raw JSON, `s <filename>` to save output, or `q` to quit.

---

### 3. Pipeline CLI (Automated QA & File Verification)

Run the integrated pipeline in one-shot mode via `run_trustlens_with_rag.py`:

- **Health check of the RAG service:**
  ```bash
  python run_trustlens_with_rag.py --health
  ```

- **Generate baseline answer via Ollama + verify claims:**
  ```bash
  python run_trustlens_with_rag.py \
    --question "What are the first-line medications for hypertension?" \
    --verify
  ```

- **Verify an existing answer text file (e.g. `sample.txt`):**
  ```bash
  python run_trustlens_with_rag.py --answer-file sample.txt --verify
  ```

- **Direct text string verification:**
  ```bash
  python run_trustlens_with_rag.py \
    --answer "Metformin is used to treat type 2 diabetes. Hypertension is high blood pressure." \
    --verify
  ```

- **Save verification output to JSON:**
  ```bash
  python run_trustlens_with_rag.py --answer-file sample.txt --verify --out results.json
  ```

---

### 4. Baseline Terminal QA Loop

To interact directly with the unverified Stage 1 baseline model:

```bash
python app.py
```
*(Enter medical questions consecutively; type `exit` or `quit` to close.)*

---

### 5. Standalone Claim Extractor

To extract atomic claims and inspect structural validation without RAG retrieval:

```bash
python claimextractor.py sample.txt
```
*(Optionally append `--refine` to enable LLM-assisted refinement.)*

---

## Testing & Quality Assurance

The codebase includes an exhaustive test suite covering unit tests, HTTP integration, failure recovery, vector store persistence, and factual verification:

| Test Suite | Command | Number of Tests | Status |
| :--- | :--- | :--- | :--- |
| **Root Integration & Verification** | `pytest tests/` | **70 tests** | ✅ 100% Passing |
| **RAG Hybrid Retrieval Subsystem** | `cd AI-Trustlens-RAG && pytest tests/` | **45 tests** | ✅ 100% Passing |
| **Claim Extractor Unit Tests** | `python -m unittest test_claimextractor.py` | **19 tests** | ✅ 100% Passing |
| **Total Automated Test Suite** | Full Project | **134 automated tests** | ✅ 100% Passing |

To run the complete test suite:

```bash
# 1. Run root integration & verification tests (70 tests)
pytest tests/

# 2. Run claim extractor unit tests (19 tests)
python -m unittest test_claimextractor.py

# 3. Run RAG hybrid retrieval subsystem tests (45 tests)
cd AI-Trustlens-RAG && pytest tests/
```

---

## Clinical Safety Disclaimer

> **IMPORTANT CLINICAL NOTICE**  
> AI TrustLens is an academic research prototype developed for hallucination mitigation and claim verification benchmarking. Outputs generated by local language models and automated retrieval components are intended for research, auditing, and experimental validation only. They **MUST NOT** be used for clinical diagnosis, prescribing, treatment planning, or direct patient care without human physician oversight.
