import os
import re
import time
import logging
from typing import Any, List, Dict, Optional

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
logger = logging.getLogger("supabase-api")

# -----------------------------
# Env vars
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REX_API_KEY = os.getenv("REX_API_KEY", "")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")

DEFAULT_LIMIT = int(os.getenv("DEFAULT_LIMIT", "50"))
MAX_LIMIT = int(os.getenv("MAX_LIMIT", "200"))

STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "8000"))
IDLE_TX_TIMEOUT_MS = int(os.getenv("IDLE_TX_TIMEOUT_MS", "8000"))

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Coxwell Supabase Read-only SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# DB Engine
# -----------------------------
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

def apply_session_safety(conn) -> None:
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

# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}

# -----------------------------
# EXISTING SQL endpoint (optional keep)
# -----------------------------
DANGEROUS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|vacuum|analyze)\b",
    re.IGNORECASE,
)
BAD_SELECT = re.compile(r"\b(pg_sleep|for\s+update|for\s+share)\b", re.IGNORECASE)

def is_select_only(sql: str) -> bool:
    s = (sql or "").strip()
    if not s.lower().startswith("select"):
        return False
    if ";" in s[:-1]:
        return False
    if DANGEROUS.search(s):
        return False
    if BAD_SELECT.search(s):
        return False
    return True

@app.get("/sqlquery_alchemy/")
@limiter.limit(RATE_LIMIT)
async def sqlquery_alchemy(sqlquery: str, api_key: str, request: Request) -> Any:
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    if not is_select_only(sqlquery):
        raise HTTPException(status_code=400, detail="Only single SELECT allowed")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sqlquery))
            rows = result.fetchall()
            cols = list(result.keys())
        return [dict(zip(cols, row)) for row in rows]
    except Exception:
        logger.exception("sqlquery_alchemy failed")
        raise HTTPException(status_code=500, detail="Database error")

# ============================================================
# ✅ NEW: AI CONTRACT ENDPOINT (NO RAW SQL FROM AI)
# ============================================================

ALLOWED_VIEW = 'public.v_product_intelligence'
ALLOWED_COLUMNS = ["id", "product_name", "thickness", "product_class", "color", "finish"]

class AIQueryRequest(BaseModel):
    question: str = Field(..., min_length=2)
    api_key: str = Field(..., min_length=5)
    limit: Optional[int] = None

def extract_thickness_mm(q: str) -> Optional[int]:
    # finds 25mm, 25 mm, thickness 25, etc.
    m = re.search(r"(\d{1,2})\s*mm", q.lower())
    if m:
        return int(m.group(1))
    # also allow: "25 mm thickness" without mm? (optional)
    m2 = re.search(r"\bthickness\s*(\d{1,2})\b", q.lower())
    if m2:
        return int(m2.group(1))
    return None

def extract_product_class(q: str) -> Optional[str]:
    s = q.lower()
    # map common words to your product_class values
    mapping = {
        "multicell": "Multicell Polycarbonate",
        "multiwall": "Multiwall Polycarbonate",
        "solid": "Solid Polycarbonate",
        "corrugated": "Corrugated Polycarbonate",
    }
    for key, val in mapping.items():
        if key in s:
            return val
    return None

def extract_color_keyword(q: str) -> Optional[str]:
    # simple: if user says "opal" or "bronze" or "clear"
    s = q.lower()
    for c in ["opal", "bronze", "clear", "blue", "green", "grey", "smoke"]:
        if c in s:
            return c
    return None

def build_safe_product_sql(question: str, limit: int) -> (str, Dict[str, Any]):
    """
    Builds SQL ONLY from allowed view + allowed columns + allowed filters.
    """
    filters = []
    params: Dict[str, Any] = {"limit": limit}

    thickness = extract_thickness_mm(question)
    if thickness is not None:
        filters.append("thickness = :thickness")
        params["thickness"] = thickness

    pclass = extract_product_class(question)
    if pclass is not None:
        filters.append("product_class = :product_class")
        params["product_class"] = pclass

    color_kw = extract_color_keyword(question)
    if color_kw is not None:
        # color column seems like text, so we do ILIKE match
        filters.append("color ILIKE :color")
        params["color"] = f"%{color_kw}%"

    # generic keyword search in product_name (only if nothing else found)
    # (keeps it useful)
    if thickness is None and pclass is None and color_kw is None:
        filters.append("product_name ILIKE :kw")
        params["kw"] = f"%{question.strip()}%"

    where_clause = ""
    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    cols = ", ".join(ALLOWED_COLUMNS)
    sql = f"""
    SELECT {cols}
    FROM {ALLOWED_VIEW}
    {where_clause}
    ORDER BY thickness NULLS LAST, product_name
    LIMIT :limit
    """
    return sql, params

@app.post("/ai_query")
@limiter.limit(RATE_LIMIT)
async def ai_query(payload: AIQueryRequest, request: Request) -> Any:
    # Auth
    if payload.api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    final_limit = clamp_limit(payload.limit)

    sql, params = build_safe_product_sql(payload.question, final_limit)

    logger.info(f"AI_QUERY ip={get_remote_address(request)} q='{payload.question[:120]}' limit={final_limit}")

    try:
        with engine.connect() as conn:
            apply_session_safety(conn)
            result = conn.execute(text(sql), params)
            rows = result.fetchall()
            cols = list(result.keys())

        return [dict(zip(cols, row)) for row in rows]

    except SQLAlchemyError:
        logger.exception("AI query DB error")
        raise HTTPException(status_code=500, detail="Database error")
    except Exception:
        logger.exception("AI query unexpected error")
        raise HTTPException(status_code=500, detail="Unexpected error")
