"""
QuantumEmbeddings API -
FastAPI + sentence-transformers + API Key Auth
Author: Bapan Sardar (QuantumHash)
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import List, Optional, Union

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, Security, Depends, Request
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger("bge-api")

model_store: dict = {}
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Loading QuantumEmbeddings model...")
    start = time.time()
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
    if device == "cuda":
        model.half()
        logger.info("FP16 mode enabled")
    model_store["bge"] = model
    model_store["device"] = device
    elapsed = time.time() - start
    logger.info(f"✅ Model loaded in {elapsed:.2f}s")
    yield
    logger.info("🛑 Shutting down...")
    model_store.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


app = FastAPI(
    title="QuantumEmbeddings API",
    description="Production embedding API using BAAI/bge-large-en-v1.5. Built by QuantumHash.",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


def get_valid_api_keys() -> List[str]:
    raw = os.environ.get("BGE_API_KEYS", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        logger.warning("⚠️  No BGE_API_KEYS set! API is OPEN.")
    return keys


async def verify_api_key(api_key: str = Security(api_key_header)):
    valid_keys = get_valid_api_keys()
    if not valid_keys:
        return "anonymous"
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API Key. Pass it as X-API-Key header.",
        )
    return api_key


class EmbedRequest(BaseModel):
    texts: Union[str, List[str]] = Field(..., example=["What is machine learning?"])
    is_query: bool = Field(
        default=False, description="True for search queries, False for documents."
    )
    normalize: bool = Field(default=True)
    batch_size: int = Field(default=32, ge=1, le=256)

    @validator("texts")
    def validate_texts(cls, v):
        if isinstance(v, str):
            return [v]
        if len(v) == 0:
            raise ValueError("texts list empty nahi ho sakti")
        if len(v) > 1000:
            raise ValueError("Max 1000 texts per request")
        return v


class EmbedResponse(BaseModel):
    embeddings: List[List[float]]
    model: str = "BAAI/bge-large-en-v1.5"
    dimension: int = 1024
    total_texts: int
    is_query: bool
    processing_time_ms: float


class SimilarityRequest(BaseModel):
    queries: Union[str, List[str]]
    passages: List[str]

    @validator("queries")
    def normalize_queries(cls, v):
        return [v] if isinstance(v, str) else v


class SimilarityResponse(BaseModel):
    scores: List[List[float]]
    queries_count: int
    passages_count: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    gpu_available: bool
    gpu_name: Optional[str]
    vram_used_mb: Optional[float]


@app.get("/", tags=["Root"])
async def root():
    return {
        "message": "QuantumEmbeddings API — QuantumHash",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    vram_used = None
    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        vram_used = torch.cuda.memory_allocated(0) / 1024 / 1024
    return HealthResponse(
        status="ok" if "bge" in model_store else "model_not_loaded",
        model_loaded="bge" in model_store,
        device=model_store.get("device", "unknown"),
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        vram_used_mb=round(vram_used, 2) if vram_used else None,
    )


@app.post("/v1/embeddings", response_model=EmbedResponse, tags=["Embeddings"])
async def create_embeddings(
    request: EmbedRequest, api_key: str = Depends(verify_api_key)
):
    if "bge" not in model_store:
        raise HTTPException(status_code=503, detail="Model load nahi hua.")
    model = model_store["bge"]
    start = time.perf_counter()
    try:
        texts_to_encode = (
            [QUERY_INSTRUCTION + t for t in request.texts]
            if request.is_query
            else request.texts
        )
        embeddings_np = model.encode(
            texts_to_encode,
            batch_size=request.batch_size,
            normalize_embeddings=request.normalize,
            show_progress_bar=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        return EmbedResponse(
            embeddings=embeddings_np.tolist(),
            total_texts=len(request.texts),
            is_query=request.is_query,
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")


@app.post("/v1/similarity", response_model=SimilarityResponse, tags=["Embeddings"])
async def compute_similarity(
    request: SimilarityRequest, api_key: str = Depends(verify_api_key)
):
    if "bge" not in model_store:
        raise HTTPException(status_code=503, detail="Model load nahi hua.")
    model = model_store["bge"]
    start = time.perf_counter()
    try:
        q_texts = [QUERY_INSTRUCTION + q for q in request.queries]
        q_emb = model.encode(
            q_texts, normalize_embeddings=True, show_progress_bar=False
        )
        p_emb = model.encode(
            request.passages, normalize_embeddings=True, show_progress_bar=False
        )
        scores = (q_emb @ p_emb.T).tolist()
        elapsed_ms = (time.perf_counter() - start) * 1000
        return SimilarityResponse(
            scores=scores,
            queries_count=len(request.queries),
            passages_count=len(request.passages),
            processing_time_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error(f"Similarity error: {e}")
        raise HTTPException(status_code=500, detail=f"Similarity failed: {str(e)}")


@app.post("/v1/rerank", tags=["Reranking"])
async def rerank_documents(
    query: str,
    documents: List[str],
    top_k: int = 5,
    api_key: str = Depends(verify_api_key),
):
    if "bge" not in model_store:
        raise HTTPException(status_code=503, detail="Model load nahi hua.")
    top_k = min(top_k, len(documents))
    model = model_store["bge"]
    start = time.perf_counter()
    try:
        q_emb = model.encode(
            [QUERY_INSTRUCTION + query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        d_emb = model.encode(
            documents, normalize_embeddings=True, show_progress_bar=False
        )
        scores = (q_emb @ d_emb.T)[0].tolist()
        ranked = sorted(
            [
                {"index": i, "document": doc, "score": round(scores[i], 4)}
                for i, doc in enumerate(documents)
            ],
            key=lambda x: x["score"],
            reverse=True,
        )[:top_k]
        elapsed_ms = (time.perf_counter() - start) * 1000
        return {
            "query": query,
            "top_k": top_k,
            "results": ranked,
            "processing_time_ms": round(elapsed_ms, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"detail": "Route nahi mili."})


@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(status_code=500, content={"detail": "Internal server error."})
