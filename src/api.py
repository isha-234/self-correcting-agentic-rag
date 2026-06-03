"""
api.py — FastAPI Server
========================
Exposes the multi-agent pipeline as a REST API.
Logs every query to W&B (if WANDB_PROJECT is set).

Run:
    uvicorn api:app --reload --port 8000
"""

import os
import time
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# W&B is optional — only import if project is configured
WANDB_PROJECT = os.getenv("WANDB_PROJECT")
wandb = None
if WANDB_PROJECT:
    try:
        import wandb as _wandb
        wandb = _wandb
    except ImportError:
        print("⚠️  wandb not installed — skipping W&B logging")

from rag_graph import MultiAgentRAGPipeline

pipeline: Optional[MultiAgentRAGPipeline] = None
COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    print("🚀  Starting Multi-Agent RAG API (Fireworks AI)...")

    if wandb and WANDB_PROJECT:
        wandb.init(
            project=WANDB_PROJECT,
            name=f"api-run-{int(time.time())}",
            config={
                "collection":       COLLECTION,
                "model_retriever":  os.getenv("MODEL_RETRIEVER"),
                "model_answerer":   os.getenv("MODEL_ANSWERER"),
                "model_critic":     os.getenv("MODEL_CRITIC"),
                "model_orchestrator": os.getenv("MODEL_ORCHESTRATOR"),
                "embed_model":      os.getenv("EMBEDDING_MODEL"),
            },
        )

    try:
        pipeline = MultiAgentRAGPipeline(collection_name=COLLECTION)
    except FileNotFoundError as e:
        print(f"⚠️  ChromaDB not found — run ingest.py first\n   {e}")

    yield

    if wandb and wandb.run:
        wandb.finish()


app = FastAPI(
    title="Multi-Agent RAG API",
    description="4-agent RAG: Retriever → Answerer → Critic → Orchestrator (Fireworks AI)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Models ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query:   str
    top_k:   int  = 5
    verbose: bool = False


class QueryResponse(BaseModel):
    query:              str
    final_answer:       str
    confidence:         float
    retrieval_attempts: int
    retrieval_score:    float
    critic_verdict:     str
    revision_count:     int
    quality_flags:      list
    latency_seconds:    float
    node_timings:       dict


class IngestRequest(BaseModel):
    docs_dir: str  = "./docs"
    reset:    bool = False


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok" if pipeline else "degraded",
        "message":   "ready" if pipeline else "run ingest.py first",
        "provider":  "Fireworks AI",
        "collection": COLLECTION,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not ready. Run ingest.py first.")

    t0 = time.perf_counter()
    try:
        state = pipeline.run(query=request.query, verbose=request.verbose)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency  = time.perf_counter() - t0
    final    = state.get("final_result", {})
    retr     = state.get("retriever_result", {})
    critic   = state.get("critic_result", {})
    timings  = state.get("node_timings", {})

    response = QueryResponse(
        query=request.query,
        final_answer=final.get("final_answer", ""),
        confidence=final.get("confidence", 0.0),
        retrieval_attempts=retr.get("attempts", 1),
        retrieval_score=retr.get("avg_score", 0.0),
        critic_verdict=critic.get("verdict", "unknown"),
        revision_count=state.get("revision_count", 0),
        quality_flags=final.get("quality_flags", []),
        latency_seconds=round(latency, 3),
        node_timings=timings,
    )

    if wandb and wandb.run:
        wandb.log({
            "query":              request.query,
            "confidence":         response.confidence,
            "retrieval_score":    response.retrieval_score,
            "retrieval_attempts": response.retrieval_attempts,
            "critic_approved":    1 if response.critic_verdict == "approve" else 0,
            "revision_count":     response.revision_count,
            "latency_total":      response.latency_seconds,
            "latency_retriever":  timings.get("retriever", 0),
            "latency_answerer":   timings.get("answerer_r0", 0),
            "latency_critic":     timings.get("critic", 0),
            "latency_orchestrator": timings.get("orchestrator", 0),
        })

    return response


@app.post("/ingest")
async def ingest(request: IngestRequest, background_tasks: BackgroundTasks):
    if not Path(request.docs_dir).exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.docs_dir}")

    def run_ingestion():
        import subprocess
        cmd = ["python", "ingest.py", "--docs_dir", request.docs_dir, "--collection", COLLECTION]
        if request.reset:
            cmd.append("--reset")
        subprocess.run(cmd, check=True)
        global pipeline
        pipeline = MultiAgentRAGPipeline(collection_name=COLLECTION)
        print("✅  Pipeline reloaded")

    background_tasks.add_task(run_ingestion)
    return {"status": "ingestion_started", "docs_dir": request.docs_dir}
