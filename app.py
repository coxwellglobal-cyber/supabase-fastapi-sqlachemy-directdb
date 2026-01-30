import os
import re
import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded


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

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Coxwell AI Query API (Read-only)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten later (your domain)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Rate limiting
# -----------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter


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
    """
    Defense-in-depth:
    - read-only transaction
    - query timeouts
    """
    conn.exec_driver_sql("SET default_transaction_read_only = on")
    conn.exec_driver_sql(f"SET statement_timeout = '{STATEMENT_TIMEOUT_MS}'")
    conn.exec_driver_sql(f"SET idle_in_transaction_session_timeout = '{IDLE_TX_TIMEOUT_MS}'")


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
# Health endpoint (kept)
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}


# -----------------------------
# AI Query Contract (main)
# -----------------------------
class AIQueryRequest(BaseModel):
    api_key: str = Field(..., description="REX API key")
    question: str = Field(..., description="User question in plain English")
    limit: Optional[int] = Field(None, description="Optional row limit (capped)")
    mode: Optional[str] = Field(
        "products",
        description="Currently supported: 'products' (queries v_product_intelligence)",
    )


def build_products_sql(question: str, limit: int) -> (str, Dict[str, Any]):
    """
    Converts plain question -> safe SQL against ONLY:
      public.v_product_intelligence

    We do NOT allow arbitrary SQL here.
    """
    q = (question or "").strip()
    if not q:
        raise HTTPException(status_code=400, detail="question is required")

    ql = q.lower()

    # Extract thickness like "25mm", "25 mm", "25"
    thickness = None
    m = re.search(r"\b(\d{1,2})\s*mm\b", ql)
    if m:
        thickness = int(m.group(1))
    else:
        # allow "25 mm multicell" without mm? (optional)
        m2 = re.search(r"\b(\d{1,2})\b", ql)
        if m2 and any(word in ql for word in ["thickness", "mm"]):
            thickness = int(m2.group(1))

    # Keyword filters (safe LIKE)
    # You can extend this later with more rules
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

    # IMPORTANT: only this view/table
    sql = """
SELECT id, product_name, thickness, product_class, color, finish
FROM public.v_product_intelligence
"""

    if thickness is not None:
        where.append("thickness = :thickness")
        params["thickness"] = thickness

    if like_tokens:
        # match on product_name OR product_class
        # (use OR inside a grouped clause)
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

    # Generic fallback: if no structured matches, do a safe broad search
    if not where:
        # search words in product_name
        params["q"] = f"%{ql[:80]}%"
        where.append("LOWER(product_name) LIKE :q")

    sql += " WHERE " + " AND ".join(where) + " ORDER BY thickness DESC, product_name ASC LIMIT :limit"
    return sql.strip(), params


@app.post("/ai_query")
@limiter.limit(RATE_LIMIT)
async def ai_query(payload: AIQueryRequest, request: Request) -> Any:
    # Auth
    if payload.api_key != REX_API_KEY:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    limit = clamp_limit(payload.limit)
    mode = (payload.mode or "products").strip().lower()

    if mode != "products":
        raise HTTPException(status_code=400, detail="Unsupported mode. Use mode='products'.")

    # Build SAFE SQL (no raw SQL accepted from user/AI)
    sql, params = build_products_sql(payload.question, limit)

    client_ip = get_remote_address(request)
    start = time.time()

    logger.info(f"AI_QUERY IP={client_ip} key={mask_key(payload.api_key)} mode={mode} sql={sql[:250]}")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            cols = list(result.keys())

        duration_ms = int((time.time() - start) * 1000)
        logger.info(f"AI_QUERY OK IP={client_ip} ms={duration_ms} rows={len(rows)}")

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]

        return {
            "mode": mode,
            "question": payload.question,
            "row_limit": limit,
            "rows": data,
        }

    except SQLAlchemyError:
        logger.exception("Database error")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail="Unexpected error")
