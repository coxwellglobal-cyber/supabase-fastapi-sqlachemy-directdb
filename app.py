import os
import re
import time
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, HttpUrl

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from supabase import create_client

# OpenAI (v1+)
from openai import OpenAI


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("coxwell-ai-api")


# -----------------------------
# Env vars (Render injects these)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REX_API_KEY = os.getenv("REX_API_KEY", "").strip()
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour").strip()

# Mixedbread / OpenAI / Supabase
MIXEDBREAD_API_KEY = os.getenv("MIXEDBREAD_API_KEY", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# Safety knobs
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "100"))

# DB timeouts (ms)
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "8000"))
IDLE_TX_TIMEOUT_MS = int(os.getenv("IDLE_TX_TIMEOUT_MS", "8000"))

# Ingestion knobs
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
MAX_PDF_BYTES = int(os.getenv("MAX_PDF_BYTES", str(25 * 1024 * 1024)))  # 25MB
PARSING_POLL_SECONDS = int(os.getenv("PARSING_POLL_SECONDS", "45"))     # max wait
PARSING_POLL_INTERVAL = float(os.getenv("PARSING_POLL_INTERVAL", "1.5"))

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Coxwell AI Query API (Read-only + Ingestion)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Rate limiting (SlowAPI)
# -----------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


# -----------------------------
# DB Engine (pooler-friendly)
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)


def apply_session_safety(conn) -> None:
    """Defense-in-depth: read-only + timeouts"""
    conn.exec_driver_sql("SET default_transaction_read_only = on")
    conn.exec_driver_sql(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'")
    conn.exec_driver_sql(f"SET idle_in_transaction_session_timeout = '{IDLE_TX_TIMEOUT_MS}ms'")


def clamp_limit(user_limit: Optional[int]) -> int:
    if user_limit is None or user_limit < 1:
        return DEFAULT_LIMIT
    if user_limit > MAX_LIMIT:
        return MAX_LIMIT
    return user_limit


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}


# -----------------------------
# AI Query Contract
# -----------------------------
class AIQueryRequest(BaseModel):
    api_key: str = Field(..., description="REX API key")
    question: str = Field(..., description="User question in plain English")
    limit: Optional[int] = Field(None, description="Optional row limit (capped)")
    mode: Optional[str] = Field("products", description="Supported: 'products'")


def build_products_sql(question: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")
    ql = q.lower()

    thickness = None
    m = re.search(r"\b(\d{1,2})\s*mm\b", ql)
    if m:
        thickness = int(m.group(1))

    like_tokens: List[str] = []
    for w in ["multicell", "multiwall", "solid", "corrugated", "vivid", "snapwall", "prism", "standing seam"]:
        if w in ql:
            like_tokens.append(w)

    color_token = None
    for c in ["clear", "opal", "bronze", "blue", "green", "grey", "smoke", "red", "yellow"]:
        if c in ql:
            color_token = c
            break

    finish_token = None
    for f in ["uv", "anti-glare", "ir", "softlite", "anti-reflective", "anti reflective"]:
        if f in ql:
            finish_token = f.replace("anti reflective", "anti-reflective")
            break

    where = []
    params: Dict[str, Any] = {"limit": limit}

    sql = """
SELECT id, product_name, thickness, product_class, color, finish
FROM public.v_product_intelligence
"""

    if thickness is not None:
        where.append("thickness = :thickness")
        params["thickness"] = thickness

    if like_tokens:
        ors = []
        for i, tok in enumerate(like_tokens):
            key = f"tok{i}"
            params[key] = f"%{tok}%"
            ors.append(f"(LOWER(product_name) LIKE :{key} OR LOWER(product_class) LIKE :{key})")
        where.append("(" + " OR ".join(ors) + ")")

    if color_token:
        params["color_tok"] = f"%{color_token}%"
        where.append("LOWER(color) LIKE :color_tok")

    if finish_token:
        params["finish_tok"] = f"%{finish_token}%"
        where.append("LOWER(finish) LIKE :finish_tok")

    if not where:
        where.append("1=1")

    sql += " WHERE " + " AND ".join(where) + " ORDER BY thickness DESC, product_name ASC LIMIT :limit"
    return sql.strip(), params


@app.post("/ai_query")
@limiter.limit(RATE_LIMIT)
async def ai_query(payload: AIQueryRequest, request: Request) -> Any:
    if payload.api_key != REX_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    limit = clamp_limit(payload.limit)
    mode = (payload.mode or "products").strip().lower()
    if mode != "products":
        raise HTTPException(status_code=400, detail="Unsupported mode. Use mode='products'.")

    sql, params = build_products_sql(payload.question, limit)

    start = time.time()
    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            cols = list(result.keys())

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]
        return {
            "mode": mode,
            "question": payload.question,
            "row_limit": limit,
            "row_count": len(data),
            "rows": data,
            "latency_ms": int((time.time() - start) * 1000),
        }

    except SQLAlchemyError:
        logger.exception("Database error")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Unexpected error")


# -----------------------------
# NEW: Ingest PDF by URL (Cloud Agent friendly)
# -----------------------------
class IngestPdfUrlRequest(BaseModel):
    api_key: str
    title: str
    product_family: str
    tags: List[str] = Field(default_factory=list)
    pdf_url: HttpUrl


def _require_ingestion_env() -> None:
    if not MIXEDBREAD_API_KEY:
        raise HTTPException(status_code=500, detail="MIXEDBREAD_API_KEY is required for ingestion")
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required for ingestion")
    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(status_code=500, detail="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for ingestion")


def _download_pdf(pdf_url: str) -> bytes:
    try:
        r = requests.get(pdf_url, timeout=30, stream=True)
        r.raise_for_status()

        content_type = (r.headers.get("content-type") or "").lower()
        if "pdf" not in content_type and not pdf_url.lower().endswith(".pdf"):
            # still allow, but warn
            logger.warning(f"PDF URL content-type not PDF: {content_type}")

        data = r.content
        if len(data) > MAX_PDF_BYTES:
            raise HTTPException(status_code=400, detail=f"PDF too large (> {MAX_PDF_BYTES} bytes)")
        return data
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download pdf_url: {str(e)}")


def _mixedbread_upload_file(filename: str, pdf_bytes: bytes) -> str:
    # POST https://api.mixedbread.com/v1/files (multipart)
    headers = {"Authorization": f"Bearer {MIXEDBREAD_API_KEY}"}
    files = {"file": (filename, pdf_bytes, "application/pdf")}
    resp = requests.post("https://api.mixedbread.com/v1/files", headers=headers, files=files, timeout=60)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mixedbread upload failed: {resp.status_code} {resp.text}")
    return resp.json()["id"]


def _mixedbread_create_parse_job(file_id: str) -> str:
    # POST https://api.mixedbread.com/v1/parsing/jobs
    headers = {"Authorization": f"Bearer {MIXEDBREAD_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "file_id": file_id,
        # Safe defaults
        "chunking_strategy": "page",
        "return_format": "markdown",
    }
    resp = requests.post("https://api.mixedbread.com/v1/parsing/jobs", headers=headers, json=payload, timeout=60)
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mixedbread create job failed: {resp.status_code} {resp.text}")
    return resp.json()["id"]


def _mixedbread_wait_for_result(job_id: str) -> Dict[str, Any]:
    headers = {"Authorization": f"Bearer {MIXEDBREAD_API_KEY}"}
    deadline = time.time() + PARSING_POLL_SECONDS

    while time.time() < deadline:
        resp = requests.get(f"https://api.mixedbread.com/v1/parsing/jobs/{job_id}", headers=headers, timeout=30)
        if resp.status_code >= 400:
            raise HTTPException(status_code=502, detail=f"Mixedbread job fetch failed: {resp.status_code} {resp.text}")

        job = resp.json()
        status_ = job.get("status")

        if status_ == "completed":
            return job
        if status_ == "failed":
            raise HTTPException(status_code=502, detail=f"Mixedbread parsing failed: {job.get('error')}")

        time.sleep(PARSING_POLL_INTERVAL)

    raise HTTPException(status_code=504, detail="Mixedbread parsing timed out")


def _openai_embed(texts: List[str]) -> List[List[float]]:
    client = OpenAI(api_key=OPENAI_API_KEY)
    # OpenAI embeddings API supports batching
    res = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in res.data]


def _supabase_client():
    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


@app.post("/ingest_pdf_url")
@limiter.limit(RATE_LIMIT)
async def ingest_pdf_url(payload: IngestPdfUrlRequest, request: Request) -> Any:
    if payload.api_key != REX_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    _require_ingestion_env()

    # 1) Download PDF
    pdf_bytes = _download_pdf(str(payload.pdf_url))
    filename = f"{payload.title.strip().replace(' ', '_')}.pdf"

    # 2) Upload to Mixedbread + Parse
    file_id = _mixedbread_upload_file(filename, pdf_bytes)
    job_id = _mixedbread_create_parse_job(file_id)
    job = _mixedbread_wait_for_result(job_id)

    # Extract chunks
    result = (job.get("result") or {})
    chunks = result.get("chunks") or []
    if not chunks:
        raise HTTPException(status_code=502, detail="No chunks returned from Mixedbread parsing")

    # 3) Prepare chunk texts
    chunk_texts: List[str] = []
    for c in chunks:
        # prefer content_to_embed if available, else content
        chunk_texts.append((c.get("content_to_embed") or c.get("content") or "").strip())

    chunk_texts = [t for t in chunk_texts if t]
    if not chunk_texts:
        raise HTTPException(status_code=502, detail="Mixedbread returned empty chunk content")

    # 4) Create embeddings (OpenAI)
    embeddings: List[List[float]] = _openai_embed(chunk_texts)

    # 5) Insert into Supabase tables
    sb = _supabase_client()

    # Insert document
    doc_payload = {
        "title": payload.title,
        "product_family": payload.product_family,
        "tags": payload.tags,
        "source_url": str(payload.pdf_url),
        "mixedbread_file_id": file_id,
        "mixedbread_job_id": job_id,
    }

    doc_res = sb.table("pi_documents").insert(doc_payload).execute()
    if not doc_res.data:
        raise HTTPException(status_code=500, detail="Failed to insert into pi_documents")
    document_id = doc_res.data[0].get("id")

    # Insert chunks
    chunk_rows = []
    for i, (text_, emb_) in enumerate(zip(chunk_texts, embeddings)):
        chunk_rows.append(
            {
                "document_id": document_id,
                "chunk_index": i,
                "content": text_,
                "embedding": emb_,  # may fail if your column name/type differs
            }
        )

    # Try inserting with embeddings; if schema differs, retry without embeddings
    try:
        chunk_res = sb.table("pi_document_chunks").insert(chunk_rows).execute()
        if not chunk_res.data:
            raise HTTPException(status_code=500, detail="Failed to insert into pi_document_chunks")
        inserted = len(chunk_res.data)
    except Exception as e:
        logger.warning(f"Chunk insert with embeddings failed, retrying without embeddings: {e}")
        chunk_rows_no_emb = [
            {k: v for k, v in row.items() if k != "embedding"} for row in chunk_rows
        ]
        chunk_res = sb.table("pi_document_chunks").insert(chunk_rows_no_emb).execute()
        if not chunk_res.data:
            raise HTTPException(status_code=500, detail="Failed to insert chunks (without embeddings)")
        inserted = len(chunk_res.data)

    return {
        "status": "ok",
        "document_id": document_id,
        "title": payload.title,
        "chunks_parsed": len(chunk_texts),
        "chunks_inserted": inserted,
        "mixedbread_file_id": file_id,
        "mixedbread_job_id": job_id,
    }
