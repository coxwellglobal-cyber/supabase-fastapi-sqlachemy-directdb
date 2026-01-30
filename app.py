import os
import re
import logging
from typing import Any, List, Dict

from fastapi import FastAPI, HTTPException, Request, status, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

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
# Env vars (Render injects these)
# -----------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
REX_API_KEY = os.getenv("REX_API_KEY", "")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/hour")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable is required")
if not REX_API_KEY:
    raise ValueError("REX_API_KEY environment variable is required")

logger.info(f"Rate limit: {RATE_LIMIT}")

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Coxwell Supabase Read-only SQL API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten later if needed
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

# -----------------------------
# Safety: allow ONLY SELECT, single statement, block dangerous keywords
# -----------------------------
DANGEROUS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|vacuum|analyze)\b",
    re.IGNORECASE,
)

AGG_HINT = re.compile(
    r"\b(group\s+by|having|count\s*\(|sum\s*\(|avg\s*\(|min\s*\(|max\s*\()\b",
    re.IGNORECASE,
)

def is_select_only(sql: str) -> bool:
    s = (sql or "").strip()
    if not s.lower().startswith("select"):
        return False
    # block multi-statement; allow a single trailing ';'
    if ";" in s[:-1]:
        return False
    if DANGEROUS.search(s):
        return False
    return True

def add_limit_if_needed(sql: str, limit: int = 100) -> str:
    s = (sql or "").strip().rstrip(";")

    # don't force LIMIT on aggregations
    if AGG_HINT.search(s):
        return s

    # already has LIMIT
    if re.search(r"\blimit\b", s, flags=re.IGNORECASE):
        return s

    return f"{s} LIMIT {limit}"

# -----------------------------
# Health endpoints
# -----------------------------
@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok", "service": "supabase-sql-api"}

# ✅ PROTECTED dbcheck (requires api_key)
@app.get("/dbcheck")
@limiter.limit(RATE_LIMIT)
async def dbcheck(
    request: Request,
    api_key: str = Query(..., description="API key required"),
) -> Dict[str, Any]:
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        with engine.connect() as conn:
            # 🔒 Enforce DB-level read-only for this session
            conn.exec_driver_sql("SET default_transaction_read_only = on")
            val = conn.execute(text("SELECT 1")).scalar()
        return {"db": "ok", "result": int(val)}
    except Exception as e:
        logger.exception("DB health check failed")
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Main endpoint
# -----------------------------
@app.get("/sqlquery_alchemy/")
@limiter.limit(RATE_LIMIT)
async def sqlquery_alchemy(sqlquery: str, api_key: str, request: Request) -> Any:
    if api_key != REX_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not is_select_only(sqlquery):
        raise HTTPException(
            status_code=400,
            detail="Only single-statement SELECT queries are allowed.",
        )

    safe_sql = add_limit_if_needed(sqlquery, limit=100)
    logger.info(f"Query: {safe_sql}")

    try:
        with engine.connect() as conn:
            # 🔒 Enforce DB-level read-only for this session
            conn.exec_driver_sql("SET default_transaction_read_only = on")

            result = conn.execute(text(safe_sql))
            rows = result.fetchall()
            cols = list(result.keys())

        data: List[Dict[str, Any]] = [dict(zip(cols, row)) for row in rows]
        return data

    except SQLAlchemyError as e:
        logger.exception("Database error")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
