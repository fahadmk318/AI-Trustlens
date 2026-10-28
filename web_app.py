#!/usr/bin/env python3
"""AI TrustLens — Modern Web Dashboard Backend.

Provides a fast, asynchronous FastAPI application serving:
- REST API for clinical QA, claim extraction, and verification.
- Live status endpoint for RAG, Ollama, and vector/graph stores.
- Static dashboard interface with glassmorphism and real-time claim annotation.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

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

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "web" / "static"

app = FastAPI(
    title="AI TrustLens Dashboard",
    description="Medical Question Answering & Autonomous Claim Verification Interface",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure verification is enabled in environment
os.environ["TRUSTLENS_ENABLE_VERIFICATION"] = "1"

# Global pipeline instance
_pipeline: Optional[TrustLensRAGPipeline] = None


def get_pipeline() -> TrustLensRAGPipeline:
    global _pipeline
    if _pipeline is None:
        config = get_config()
        _pipeline = TrustLensRAGPipeline(config=config)
    return _pipeline


# -----------------------------------------------------------------------------
# Request & Response Models
# -----------------------------------------------------------------------------

class VerificationRequest(BaseModel):
    mode: str = Field("question", description="'question' (runs Ollama QA) or 'answer' (verifies text directly)")
    text: str = Field(..., description="The medical question or medical answer text")
    domain: Optional[str] = Field("Medical", description="Corpus domain filter")
    top_k: Optional[int] = Field(5, description="Evidence chunks to retrieve per claim")


# Preset Medical Questions for Instant Demos
PRESETS = [
    {
        "id": "hypertension",
        "title": "Hypertension First-Line Medications",
        "category": "treatment",
        "question": "What are the recommended first-line medications for hypertension?",
    },
    {
        "id": "diabetes",
        "title": "Diabetes Management & Metformin",
        "category": "medication",
        "question": "Is Metformin used as a first-line treatment for type 2 diabetes?",
    },
    {
        "id": "antibiotics",
        "title": "Antibiotics & Viral Infections",
        "category": "contraindication",
        "question": "Can antibiotics be used to treat viral infections like the common cold or flu?",
    },
    {
        "id": "cancer",
        "title": "Cancer Overview & Metastasis",
        "category": "disease",
        "question": "What is cancer and how does it spread to other parts of the body?",
    },
    {
        "id": "fever",
        "title": "Fever Definition & Causes (sample.txt)",
        "category": "symptom",
        "mode": "answer",
        "text": (
            "A fever is a temporary increase in body temperature, usually as a response to an illness or infection. "
            "Normal body temperature is typically between 97°F (36.1°C) and 99°F (37.2°C). "
            "A fever is generally considered to be a body temperature of 100.4°F (38°C) or higher. "
            "Common causes of fever include infections, inflammation, cancer, vaccinations, and heat exhaustion."
        ),
    },
]


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.get("/api/health")
def api_health() -> Dict[str, Any]:
    """Check health of RAG microservice and local model."""
    config = get_config()
    client = RAGClient(config)
    rag_info = {}
    rag_ok = False
    try:
        rag_info = client.health()
        rag_ok = rag_info.get("status") == "ok"
    except Exception as exc:
        rag_info = {"status": "error", "error": str(exc)}

    return {
        "status": "ok" if rag_ok else "degraded",
        "rag_ok": rag_ok,
        "rag": rag_info,
        "model": "qwen3:8b",
        "base_url": config.base_url,
    }


@app.get("/api/presets")
def api_presets() -> List[Dict[str, Any]]:
    """Return pre-configured clinical questions for one-click demos."""
    return PRESETS


import threading
import uuid

# In-memory task registry for asynchronous execution
_tasks: Dict[str, Dict[str, Any]] = {}


@app.post("/api/verify")
def api_verify(req: VerificationRequest) -> Dict[str, Any]:
    """Launch asynchronous verification job and return task ID immediately."""
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty.")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {
        "status": "processing",
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    def _worker():
        try:
            pipeline = get_pipeline()
            start_time = time.time()
            if req.mode == "question":
                result = pipeline.process_question(
                    question=req.text.strip(),
                    top_k=req.top_k,
                    domain=req.domain,
                    verify=True,
                )
            else:
                result = pipeline.process_answer(
                    question="",
                    answer=req.text.strip(),
                    top_k=req.top_k,
                    domain=req.domain,
                    verify=True,
                )
            result["latency_seconds"] = round(time.time() - start_time, 2)
            _tasks[task_id]["result"] = result
            _tasks[task_id]["status"] = "completed"
        except RAGConnectionError as exc:
            _tasks[task_id]["error"] = f"Cannot reach RAG service at {get_config().base_url}. Ensure uvicorn is running on port 8000."
            _tasks[task_id]["status"] = "failed"
        except Exception as exc:
            _tasks[task_id]["error"] = str(exc)
            _tasks[task_id]["status"] = "failed"

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()

    return {"task_id": task_id, "status": "processing"}


@app.get("/api/task/{task_id}")
def api_task_status(task_id: str) -> Dict[str, Any]:
    """Poll status of an asynchronous verification task."""
    if task_id not in _tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    task = _tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "error": task["error"],
        "result": task["result"] if task["status"] == "completed" else None,
        "elapsed_seconds": round(time.time() - task["created_at"], 1),
    }


# -----------------------------------------------------------------------------
# Static Dashboard Mount
# -----------------------------------------------------------------------------

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def serve_dashboard():
    """Serve the single-page application dashboard."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse(
        content={"message": "AI TrustLens API online. Static files initializing."},
        status_code=200,
    )


def start_server(host: str = "0.0.0.0", port: int = 8080):
    print(f"\n[AI TrustLens] Starting Web Dashboard on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    host = os.environ.get("TRUSTLENS_HOST", "0.0.0.0")
    port = int(os.environ.get("TRUSTLENS_PORT", "8080"))
    start_server(host=host, port=port)
