import os
import re
import time
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Request, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("coxwell-ai-api")


# -----------------------------
# Env vars (Render injects these)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REX_API_KEY = os.getenv("REX_API_KEY", "").strip()
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour").strip()

# Safety knobs
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "100"))

# DB timeouts (ms)
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "8000"))  # 8s
IDLE_TX_TIMEOUT_MS = int(os.getenv("IDLE_TX_TIMEOUT_MS", "8000"))      # 8s

# -----------------------------
# NEW: Ingestion env vars (for Mixedbread + Supabase inserts)
# -----------------------------
MIXEDBREAD_API_KEY = os.getenv("MIXEDBREAD_API_KEY", "").strip()
MIXEDBREAD_PARSE_URL = os.getenv("MIXEDBREAD_PARSE_URL", "https://api.mixedbread.com/v1/parsing").strip()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()

# Optional ingestion rate-limit (separate from /ai_query)
INGEST_RATE_LIMIT = os.getenv("INGEST_RATE_LIMIT", "10/hour").strip()

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
    allow_origins=["*"],  # tighten later (your domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Rate limiting (SlowAPI)
# -----------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)  # ✅ IMPORTANT


@app.exception_handler(RateLimitExceeded)
async def custom_rate_limit_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


# -----------------------------
# DB Engine (pooler-friendly) - READ ONLY for /ai_query
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)


def apply_session_safety(conn) -> None:
    """
    Defense-in-depth:
    - read-only transaction
    - query timeouts
    """
    conn.exec_driver_sql("SET default_transaction_read_only = on")
    conn.exec_driver_sql(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'")
    conn.exec_driver_sql(f"SET idle_in_transaction_session_timeout = '{IDLE_TX_TIMEOUT_MS}ms'")


def clamp_limit(user_limit: Optional[int]) -> int:
    if user_limit is None:
        return DEFAULT_LIMIT
    if user_limit < 1:
        return DEFAULT_LIMIT
    if user_limit > MAX_LIMIT:
        return MAX_LIMIT
    return user_limit


def mask_key(k: str) -> str:
    if not k:
        return ""
    if len(k) <= 8:
        return "****"
    return k[:4] + "****" + k[-4:]


# -----------------------------
# NEW: Supabase client (for inserts)
# -----------------------------
def get_supabase():
    """
    Uses supabase-py (REST) with service role key for inserts.
    This avoids touching the read-only SQLAlchemy engine.
    """
    try:
        from supabase import create_client  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: supabase. Add `supabase` to requirements.txt"
        )

    if not SUPABASE_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise HTTPException(
            status_code=500,
            detail="SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required for ingestion"
        )

    return create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


# -----------------------------
# NEW: OpenAI embeddings
# -----------------------------
def embed_texts_openai(texts: List[str]) -> List[List[float]]:
    """
    Returns embeddings for texts using OpenAI.
    Uses model text-embedding-3-small (1536 dims) to match vector(1536).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required for ingestion")

    try:
        from openai import OpenAI  # type: ignore
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Missing dependency: openai. Add `openai` to requirements.txt"
        )

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [d.embedding for d in resp.data]


# -----------------------------
# NEW: Mixedbread parse helper
# -----------------------------
def mixedbread_parse_pdf(file_bytes: bytes, filename: str, metadata: Dict[str, Any], strategy: str = "fast") -> List[Dict[str, Any]]:
    """
    Sends PDF to Mixedbread and returns normalized chunks:
    [
      {"chunk_index": 0, "content": "...", "page_number": 1, "meta": {...}},
      ...
    ]
    """
    if not MIXEDBREAD_API_KEY:
        raise HTTPException(status_code=500, detail="MIXEDBREAD_API_KEY is required for ingestion")

    headers = {"Authorization": f"Bearer {MIXEDBREAD_API_KEY}"}
    files = {"file": (filename, file_bytes, "application/pdf")}
    data = {
        "strategy": strategy,
        "metadata": json.dumps(metadata),
    }

    try:
        r = requests.post(MIXEDBREAD_PARSE_URL, headers=headers, files=files, data=data, timeout=180)
    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Mixedbread request failed: {str(e)}")

    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Mixedbread parse failed: {r.status_code} {r.text}")

    payload = r.json()

    # Try common shapes
    chunks = payload.get("chunks") or payload.get("data") or payload.get("results") or []
    if not chunks:
        if payload.get("text"):
            chunks = [{"text": payload["text"], "page_number": None, "metadata": metadata}]
        else:
            raise HTTPException(status_code=502, detail=f"Mixedbread returned no chunks. Keys={list(payload.keys())}")

    normalized: List[Dict[str, Any]] = []
    for i, ch in enumerate(chunks):
        t = ch.get("text") or ch.get("content") or ""
        t = (t or "").strip()
        if not t:
            continue
        normalized.append(
            {
                "chunk_index": i,
                "content": t,
                "page_number": ch.get("page_number") or ch.get("page") or None,
                "meta": ch.get("metadata") or ch.get("meta") or metadata,
            }
        )

    if not normalized:
        raise HTTPException(status_code=502, detail="Mixedbread chunks were empty after normalization")

    return normalized


# -----------------------------
# Health endpoint
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}


# -----------------------------
# AI Query Contract (existing)
# -----------------------------
class AIQueryRequest(BaseModel):
    api_key: str = Field(..., description="REX API key")
    question: str = Field(..., description="User question in plain English")
    limit: Optional[int] = Field(None, description="Optional row limit (capped)")
    mode: Optional[str] = Field(
        "products",
        description="Currently supported: 'products' (queries v_product_intelligence)",
    )


def build_products_sql(question: str, limit: int) -> Tuple[str, Dict[str, Any]]:
    """
    Converts plain question -> safe SQL against ONLY:
      public.v_product_intelligence
    """
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    ql = q.lower()

    # Extract thickness like "25mm", "25 mm"
    thickness = None
    m = re.search(r"\b(\d{1,2})\s*mm\b", ql)
    if m:
        thickness = int(m.group(1))

    # Keywords (safe LIKE)
    like_tokens: List[str] = []
    for w in ["multicell", "multiwall", "solid", "corrugated", "vivid", "snapwall", "prism", "standing seam"]:
        if w in ql:
            like_tokens.append(w)

    # Color hints
    color_token = None
    for c in ["clear", "opal", "bronze", "blue", "green", "grey", "smoke", "red", "yellow"]:
        if c in ql:
            color_token = c
            break

    # Finish hints
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

    # Fallback
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

    client_ip = get_remote_address(request)
    start = time.time()

    logger.info(f"AI_QUERY IP={client_ip} key={mask_key(payload.api_key)} mode={mode}")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            cols = list(result.keys())

        duration_ms = int((time.time() - start) * 1000)
        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]

        return {
            "mode": mode,
            "question": payload.question,
            "row_limit": limit,
            "row_count": len(data),
            "rows": data,
            "latency_ms": duration_ms
        }

    except SQLAlchemyError:
        logger.exception("Database error")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Unexpected error")


# -----------------------------
# NEW: PDF Ingestion Endpoint
# -----------------------------
@app.post("/ingest_pdf_file")
@limiter.limit(INGEST_RATE_LIMIT)
async def ingest_pdf_file(
    request: Request,
    api_key: str = Form(...),
    title: str = Form(...),
    product_family: str = Form(...),
    tags: Optional[str] = Form(None),  # comma-separated
    file: UploadFile = File(...)
) -> Dict[str, Any]:
    """
    Upload PDF -> Mixedbread parse -> OpenAI embeddings -> Insert into Supabase tables:
    - public.pi_documents
    - public.pi_document_chunks
    """
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    if not file:
        raise HTTPException(status_code=400, detail="file is required")

    if file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    pdf_bytes = await file.read()
    if not pdf_bytes or len(pdf_bytes) < 200:
        raise HTTPException(status_code=400, detail="Uploaded file looks empty")

    tag_list: List[str] = []
    if tags:
        tag_list = [t.strip() for t in tags.split(",") if t.strip()]

    client_ip = get_remote_address(request)
    logger.info(f"INGEST_PDF IP={client_ip} key={mask_key(api_key)} title={title} family={product_family}")

    # 1) Parse with Mixedbread
    meta = {
        "company": "Coxwell",
        "source": "mixedbread",
        "document_type": "pdf",
        "title": title,
        "product_family": product_family,
        "tags": tag_list,
        "filename": file.filename or "document.pdf",
    }

    chunks = mixedbread_parse_pdf(
        file_bytes=pdf_bytes,
        filename=file.filename or "document.pdf",
        metadata=meta,
        strategy="fast",
    )

    # 2) Embeddings (batch)
    texts = [c["content"] for c in chunks]
    embeddings = embed_texts_openai(texts)

    if len(embeddings) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    # 3) Insert into Supabase
    sb = get_supabase()

    # Insert into pi_documents
    doc_row = {
        "title": title,
        "product_family": product_family,
        "tags": tag_list,
        "meta": meta,
        "source_url": None,
    }

    doc_insert = sb.table("pi_documents").insert(doc_row).execute()
    if not getattr(doc_insert, "data", None):
        raise HTTPException(status_code=500, detail="Failed to insert into pi_documents")

    document_id = doc_insert.data[0]["id"]

    # Insert chunks in batches
    chunk_rows = []
    for i, c in enumerate(chunks):
        chunk_rows.append({
            "document_id": document_id,
            "chunk_index": c["chunk_index"],
            "content": c["content"],
            "page_number": c["page_number"],
            "meta": c["meta"],
            "embedding": embeddings[i],
        })

    BATCH = 50
    inserted = 0
    for start in range(0, len(chunk_rows), BATCH):
        batch = chunk_rows[start:start + BATCH]
        res = sb.table("pi_document_chunks").insert(batch).execute()
        if not getattr(res, "data", None):
            raise HTTPException(status_code=500, detail="Failed inserting some chunk batches")
        inserted += len(res.data)

    return {
        "status": "ok",
        "document_id": document_id,
        "chunks_inserted": inserted,
        "filename": file.filename,
        "title": title,
        "product_family": product_family,
        "tags": tag_list
    }
