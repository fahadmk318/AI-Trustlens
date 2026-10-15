# AI TrustLens — Stage 1: Baseline Medical QA & Claim Extraction

## Overview
**AI TrustLens** is a final-year B.Tech project focusing on medical question-answering, claim decomposition, and hallucination detection/mitigation.

**Stage 1** establishes the core foundational prototype:
1. **Baseline Medical Question-Answering (`app.py`):** Direct interaction with a locally hosted **Qwen3-8B** model via the **Ollama** Python SDK with deterministic greedy decoding and token bounding.
2. **Medical Claim Extractor & Validator (`claimextractor.py`):** An automated pipeline that extracts atomic factual claims from generated medical answers, decomposes compound statements, assigns standardized medical categories, and validates claims for semantic fidelity and internal consistency.
3. **Unit & Regression Test Suite (`test_claimextractor.py`):** Comprehensive automated tests ensuring semantic integrity, proper claim decomposition, and error handling.

> **Stage 1 Scope Note:** This stage establishes local generation, claim decomposition, and rule-based validation. External Retrieval-Augmented Generation (RAG), external knowledge graphs, and cross-source verification belong to subsequent project stages.

---

## Technical Specifications

- **Target Model:** `qwen3:8b` (hosted locally via Ollama)
- **Runtime Environment:** Python 3.10+ on Linux / Ubuntu
- **Dependencies:** `ollama>=0.6.0` (zero paid/cloud API dependencies)
- **Baseline Inference Parameters:**
  - `think=False`: Suppresses chain-of-thought tokens if supported.
  - `temperature=0`: Deterministic, greedy generation.
  - `num_predict=200`: Caps generated output length to ~200 tokens.
- **Claim Extractor Taxonomy:**
  - 22 standardized medical categories (e.g., `definition`, `symptom`, `disease`, `treatment`, `dosage`, `contraindication`, `risk_factor`, `prevention`, etc.).
  - Automated detection of hedge qualifiers, absolute qualifiers, and negations.
  - Strict atomic sentence decomposition and span-alignment validation.

---

## Medical Safety Disclaimer

> **IMPORTANT:** Answers and claims generated or processed by this application are raw, baseline outputs from a local large language model. They are **NOT medically verified**, may contain factual errors or hallucinations, and **MUST NEVER** be used for clinical diagnosis, treatment decisions, or professional medical advice.

---

## Setup & Installation

### 1. Prerequisites
- Linux / Ubuntu environment
- Local Ollama daemon running (`ollama serve` or system service)
- Ensure the `qwen3:8b` model is downloaded locally:
  ```bash
  ollama list
  # If qwen3:8b is not listed, pull it once:
  ollama pull qwen3:8b
  ```

### 2. Virtual Environment Setup
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

---

## Execution Guide

### 1. Running the Baseline Medical QA System
Launch the interactive terminal-based Q&A loop:
```bash
python app.py
```
- Enter medical questions at the `Question>` prompt.
- Enter multiple questions consecutively.
- Type `exit`, `quit`, or press `Ctrl+C` / `Ctrl+D` to end the session.

---

### 2. Running the Medical Claim Extractor & Validator
Extract, decompose, and validate atomic claims from medical text:

- **Using a text file input:**
  ```bash
  python claimextractor.py sample.txt
  ```

- **With optional LLM-assisted refinement (`--refine`):**
  ```bash
  python claimextractor.py sample.txt --refine
  ```

- **Using piped standard input:**
  ```bash
  cat sample.txt | python claimextractor.py
  ```

- **Interactive input mode:**
  ```bash
  python claimextractor.py
  # Paste or type medical text, then type 'EOF' on a new line or press Ctrl+D
  ```

Output is returned as structured JSON containing atomic claims, categories, qualifiers, source text spans, and validation statuses.

---

### 3. Running the Test Suite
Execute the automated regression and unit test suite:
```bash
python -m unittest test_claimextractor.py
```
All unit tests should pass with status `OK`.

---

## Project Structure

```
Stage 1/
├── app.py                  # Baseline interactive medical QA application (Ollama + Qwen3-8B)
├── claimextractor.py       # Atomic medical claim extraction, decomposition & validation engine
├── test_claimextractor.py  # Automated unit and regression test suite (19 test cases)
├── sample.txt              # Sample medical text for testing claim extraction (fever case)
├── requirements.txt        # Python package dependencies (ollama)
├── .gitignore              # Git ignore rules (.venv, __pycache__, IDE configs)
└── README.md               # Project documentation and execution instructions
```

---

## Collaborative Workflow (Git & GitHub)

When collaborating or uploading to GitHub:
- **Upload:** `app.py`, `claimextractor.py`, `test_claimextractor.py`, `sample.txt`, `requirements.txt`, `.gitignore`, `README.md`.
- **Do NOT upload:** `.venv/` (machine-dependent) or `__pycache__/` (auto-generated). These are ignored by `.gitignore`.
