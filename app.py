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

# Mixedbread + OpenAI for ingestion
MIXEDBREAD_API_KEY = os.getenv("MIXEDBREAD_API_KEY", os.getenv("MIXEDBREAD_API_KEY".upper(), "")).strip()
# (Some people set MIXEDBREAD_API_KEY, you showed MIXEDBREAD_API_KEY in Render. Good.)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

# If Mixedbread ever changes, you can override without code change:
MIXEDBREAD_PARSE_URL = os.getenv("MIXEDBREAD_PARSE_URL", "https://api.mixedbread.com/v1/parse").strip()

# Embedding model used for 1536 dims
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small").strip()

# Safety knobs
DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "100"))

# DB timeouts (ms)
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "8000"))  # 8s
IDLE_TX_TIMEOUT_MS = int(os.getenv("IDLE_TX_TIMEOUT_MS", "8000"))      # 8s

# Chunking knobs
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1400"))   # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

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
# DB Engine
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)


def apply_readonly_session_safety(conn) -> None:
    """
    Defense-in-depth for READ endpoints:
    - force read-only
    - set query timeouts
    """
    conn.exec_driver_sql("SET default_transaction_read_only = on")
    conn.exec_driver_sql(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}ms'")
    conn.exec_driver_sql(f"SET idle_in_transaction_session_timeout = '{IDLE_TX_TIMEOUT_MS}ms'")


def apply_write_session_safety(conn) -> None:
    """
    Safety for WRITE endpoints:
    - do NOT set read-only
    - still set timeouts
    """
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
    mode: Optional[str] = Field(
        "products",
        description="Supported: 'products' (queries v_product_intelligence)",
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

    client_ip = get_remote_address(request)
    start = time.time()
    logger.info(f"AI_QUERY IP={client_ip} key={mask_key(payload.api_key)} mode={mode}")

    try:
        with engine.connect() as conn:
            apply_readonly_session_safety(conn)
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


# =========================================================
# Ingestion helpers (Mixedbread parse -> chunk -> embed -> DB)
# =========================================================

def chunk_text(text_in: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    t = (text_in or "").strip()
    if not t:
        return []
    chunks = []
    i = 0
    n = len(t)
    while i < n:
        j = min(i + chunk_size, n)
        chunks.append(t[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return chunks


def openai_embed(texts: List[str]) -> List[List[float]]:
    """
    Uses OpenAI Embeddings via HTTP (no extra SDK needed).
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required for ingestion")
    if not texts:
        return []

    resp = requests.post(
        "https://api.openai.com/v1/embeddings",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": OPENAI_EMBED_MODEL,
            "input": texts
        },
        timeout=60,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI embeddings failed: {resp.text}")

    data = resp.json().get("data", [])
    # keep same order
    return [row["embedding"] for row in data]


def vector_literal(vec: List[float]) -> str:
    # pgvector accepts: '[1,2,3]'
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def mixedbread_parse_from_url(pdf_url: str) -> str:
    """
    Calls Mixedbread parse with a URL.
    Mixedbread endpoint is configurable via MIXEDBREAD_PARSE_URL.
    """
    if not MIXEDBREAD_API_KEY:
        raise HTTPException(status_code=500, detail="MIXEDBREAD_API_KEY is required for ingestion")

    # IMPORTANT: https + correct endpoint (your earlier error showed http://api.mixedbread.com/v1/parsing)
    resp = requests.post(
        MIXEDBREAD_PARSE_URL,
        headers={
            "Authorization": f"Bearer {MIXEDBREAD_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "source": {"type": "url", "url": pdf_url},
            "strategy": "fast"
        },
        timeout=90,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Mixedbread parse failed: {resp.status_code} {resp.text}")

    parsed = resp.json()

    # Try common structures safely
    if isinstance(parsed, dict):
        if "text" in parsed and isinstance(parsed["text"], str):
            return parsed["text"]

        blocks = parsed.get("blocks") or parsed.get("content") or parsed.get("data") or []
        if isinstance(blocks, list) and blocks:
            parts = []
            for b in blocks:
                if isinstance(b, dict) and "text" in b and isinstance(b["text"], str):
                    parts.append(b["text"])
                elif isinstance(b, str):
                    parts.append(b)
            return "\n".join(parts).strip()

    # fallback
    return json.dumps(parsed)[:20000]


def save_document_and_chunks(
    title: str,
    product_family: str,
    tags: List[str],
    source_url: Optional[str],
    full_text: str
) -> Dict[str, Any]:
    """
    Inserts into:
      public.pi_documents
      public.pi_document_chunks (with embedding vector)
    """
    chunks = chunk_text(full_text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text extracted to ingest")

    embeddings = openai_embed(chunks)

    if len(embeddings) != len(chunks):
        raise HTTPException(status_code=500, detail="Embedding count mismatch")

    try:
        with engine.begin() as conn:
            apply_write_session_safety(conn)

            doc_row = conn.execute(
                text("""
                    INSERT INTO public.pi_documents (title, source_url, product_family, tags, meta)
                    VALUES (:title, :source_url, :product_family, :tags, :meta)
                    RETURNING id
                """),
                {
                    "title": title,
                    "source_url": source_url,
                    "product_family": product_family,
                    "tags": tags,
                    "meta": {}
                }
            ).fetchone()

            document_id = str(doc_row[0])

            # Insert chunks
            for idx, (c, emb) in enumerate(zip(chunks, embeddings), start=1):
                conn.execute(
                    text("""
                        INSERT INTO public.pi_document_chunks
                        (chunk_index, document_id, content, page_number, meta, embedding)
                        VALUES (:chunk_index, :document_id, :content, :page_number, :meta, :embedding::vector)
                    """),
                    {
                        "chunk_index": idx,
                        "document_id": document_id,
                        "content": c,
                        "page_number": None,
                        "meta": {},
                        "embedding": vector_literal(emb),
                    }
                )

        return {
            "status": "success",
            "document_id": document_id,
            "chunks_inserted": len(chunks),
        }

    except SQLAlchemyError:
        logger.exception("DB write failed")
        raise HTTPException(status_code=500, detail="Database error during ingestion")
    except Exception:
        logger.exception("Unexpected ingestion error")
        raise HTTPException(status_code=500, detail="Unexpected error during ingestion")


# =========================================================
# NEW ✅ Cloud-Agent friendly endpoint (JSON)
# =========================================================
class IngestPdfUrlRequest(BaseModel):
    api_key: str
    title: str
    product_family: str
    tags: List[str] = Field(default_factory=list)
    pdf_url: str


@app.post("/ingest_pdf_url")
@limiter.limit(RATE_LIMIT)
async def ingest_pdf_url(payload: IngestPdfUrlRequest, request: Request) -> Any:
    if payload.api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not payload.pdf_url.lower().startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="pdf_url must be a public http(s) URL")

    client_ip = get_remote_address(request)
    logger.info(f"INGEST_PDF_URL IP={client_ip} key={mask_key(payload.api_key)} title={payload.title}")

    full_text = mixedbread_parse_from_url(payload.pdf_url)

    return save_document_and_chunks(
        title=payload.title,
        product_family=payload.product_family,
        tags=payload.tags,
        source_url=payload.pdf_url,
        full_text=full_text
    )


# =========================================================
# Existing file-upload endpoint (multipart)
# NOTE: Cloud Agent often fails with file upload. Use /ingest_pdf_url instead.
# =========================================================
@app.post("/ingest_pdf_file")
@limiter.limit(RATE_LIMIT)
async def ingest_pdf_file(
    request: Request,
    api_key: str = Form(...),
    title: str = Form(...),
    product_family: str = Form(...),
    tags: str = Form(""),
    file: UploadFile = File(...)
) -> Any:
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not MIXEDBREAD_API_KEY:
        raise HTTPException(status_code=500, detail="MIXEDBREAD_API_KEY is required for ingestion")

    # Parse tags from comma string
    tag_list = [t.strip() for t in (tags or "").split(",") if t.strip()]

    client_ip = get_remote_address(request)
    logger.info(f"INGEST_PDF_FILE IP={client_ip} key={mask_key(api_key)} title={title} filename={file.filename}")

    # Read file bytes
    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    # Mixedbread parse (multipart)
    # If Mixedbread expects a different file field name, change "file" here.
    resp = requests.post(
        MIXEDBREAD_PARSE_URL,
        headers={"Authorization": f"Bearer {MIXEDBREAD_API_KEY}"},
        files={"file": (file.filename or "document.pdf", pdf_bytes, file.content_type or "application/pdf")},
        data={"strategy": "fast"},
        timeout=120,
    )

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Mixedbread parse failed: {resp.status_code} {resp.text}")

    parsed = resp.json()
    if isinstance(parsed, dict) and "text" in parsed and isinstance(parsed["text"], str):
        full_text = parsed["text"]
    else:
        blocks = parsed.get("blocks", []) if isinstance(parsed, dict) else []
        full_text = "\n".join([b.get("text", "") for b in blocks if isinstance(b, dict)]).strip()

    return save_document_and_chunks(
        title=title,
        product_family=product_family,
        tags=tag_list,
        source_url=None,
        full_text=full_text
    )
